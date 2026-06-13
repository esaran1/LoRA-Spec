from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lora_spec.adapter_props import compute_adapter_properties, read_adapter_metadata
from lora_spec.theory import effective_rank, spectral_analysis
from lora_spec.utils import (
    add_common_args,
    ensure_dir,
    get_config_value,
    load_yaml,
    resolve_torch_dtype,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure effective rank of the adapter-induced logit shift.")
    add_common_args(parser)
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--adapters-config", type=str, default="configs/adapters.yaml")
    parser.add_argument("--model-pair", type=str, default=None)
    parser.add_argument("--adapter-name", action="append", default=[])
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--rank-estimation-mode", type=str, choices=["auto", "exact", "projected"], default="auto")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--max-matrix-gb", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="results/theory")
    parser.add_argument("--plots-dir", type=str, default="results/theory/plots")
    return parser.parse_args()


def _load_prompts(path: str) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _load_base_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    return model, tokenizer


def _apply_lora_scale(model: PreTrainedModel, scale: float) -> None:
    if scale == 1.0:
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.mul_(scale)


def _load_adapted_model(
    base_model_name: str,
    adapter_path: str,
    magnitude_scale: float,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> PreTrainedModel:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    adapted = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
    _apply_lora_scale(adapted, magnitude_scale)
    return adapted


@torch.inference_mode()
def _collect_model_logit_rows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    rows: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits[:, :-1, :].float()
        logits = logits - logits.mean(dim=-1, keepdim=True)
        mask = encoded["attention_mask"][:, 1:].bool()
        for batch_index in range(mask.shape[0]):
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).reshape(-1)
            for position in valid_positions.tolist():
                rows.append(logits[batch_index, position].detach().cpu())
    if not rows:
        raise ValueError("Calibration prompts did not yield any next-token positions")
    return torch.stack(rows)


def _count_next_token_positions(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
) -> int:
    total = 0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        total += int(encoded["attention_mask"][:, 1:].sum().item())
    return total


@torch.inference_mode()
def _collect_projected_logit_rows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
    projection: torch.Tensor,
    row_count: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    projection_device = projection.to(device=device)
    projected_rows = torch.empty((row_count, projection.shape[1]), dtype=torch.float32)
    offset = 0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits[:, :-1, :].float()
        logits = logits - logits.mean(dim=-1, keepdim=True)
        projected = logits @ projection_device
        mask = encoded["attention_mask"][:, 1:].bool()
        for batch_index in range(mask.shape[0]):
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).reshape(-1)
            for position in valid_positions.tolist():
                projected_rows[offset] = projected[batch_index, position].detach().cpu()
                offset += 1
    if offset != row_count:
        raise RuntimeError(f"Projected row count mismatch: expected {row_count}, got {offset}")
    return projected_rows


def _select_experiments(
    models_config: str,
    adapters_config: str,
    selected_model_pair: str | None,
    selected_adapters: list[str],
) -> list[dict[str, Any]]:
    model_pairs = load_yaml(models_config).get("model_pairs", {})
    adapter_payload = load_yaml(adapters_config)
    adapters = adapter_payload.get("adapters", {})
    experiments = adapter_payload.get("experiments", [])
    if not isinstance(model_pairs, dict) or not isinstance(adapters, dict):
        raise ValueError("Invalid model or adapter config payload")

    selected_set = set(selected_adapters)
    if experiments:
        rows = []
        for entry in experiments:
            if selected_model_pair and entry["model_pair"] != selected_model_pair:
                continue
            if selected_set and entry["adapter"] not in selected_set:
                continue
            rows.append(
                {
                    "model_pair_name": entry["model_pair"],
                    "adapter_name": entry["adapter"],
                    "model_config": model_pairs[entry["model_pair"]],
                    "adapter_config": adapters[entry["adapter"]],
                    "notes": entry.get("notes", ""),
                    "tags": entry.get("tags", []),
                }
            )
        return rows

    rows = []
    for model_pair_name, model_config in model_pairs.items():
        if selected_model_pair and model_pair_name != selected_model_pair:
            continue
        for adapter_name, adapter_config in adapters.items():
            if selected_set and adapter_name not in selected_set:
                continue
            rows.append(
                {
                    "model_pair_name": model_pair_name,
                    "adapter_name": adapter_name,
                    "model_config": model_config,
                    "adapter_config": adapter_config,
                    "notes": "",
                    "tags": [],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "measure_logit_shift_rank")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)

    models_config = str(get_config_value(config_data, args, "models_config"))
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    prompts_file = str(get_config_value(config_data, args, "prompts_file"))
    if not prompts_file:
        raise ValueError("prompts_file must be provided")
    model_pair = get_config_value(config_data, args, "model_pair")
    adapter_names = get_config_value(config_data, args, "adapter_name", args.adapter_name)
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    energy_threshold = float(get_config_value(config_data, args, "energy_threshold"))
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    rank_estimation_mode = str(get_config_value(config_data, args, "rank_estimation_mode"))
    projection_dim = int(get_config_value(config_data, args, "projection_dim"))
    max_matrix_gb = float(get_config_value(config_data, args, "max_matrix_gb"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    plots_dir = ensure_dir(str(get_config_value(config_data, args, "plots_dir")))

    prompts = _load_prompts(prompts_file)
    experiments = _select_experiments(
        models_config=models_config,
        adapters_config=adapters_config,
        selected_model_pair=str(model_pair) if model_pair else None,
        selected_adapters=list(adapter_names or []),
    )
    if not experiments:
        raise ValueError("No experiments matched the requested model/adapter selection")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(torch_dtype_name, device=device)
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        model_config = dict(experiment["model_config"])
        adapter_config = dict(experiment["adapter_config"])
        base_model_name = str(model_config["target_model"])
        adapter_path = str(adapter_config["hf_path"])
        magnitude_scale = float(adapter_config.get("magnitude_scale", 1.0))
        compatible_target = adapter_config.get("target_model")
        if compatible_target and str(compatible_target) != base_model_name:
            raise ValueError(
                f"Adapter {experiment['adapter_name']} targets {compatible_target}, not {base_model_name}"
            )
        logger.info(
            "Measuring shift rank for %s x %s (scale=%.3f)",
            experiment["model_pair_name"],
            experiment["adapter_name"],
            magnitude_scale,
        )
        base_model, tokenizer = _load_base_model_and_tokenizer(
            base_model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        row_count = _count_next_token_positions(tokenizer, prompts, batch_size)
        estimated_matrix_gb = (row_count * tokenizer.vocab_size * 4.0) / (1024.0**3)
        use_projected = rank_estimation_mode == "projected" or (
            rank_estimation_mode == "auto" and estimated_matrix_gb > max_matrix_gb
        )
        analysis_input_dim = tokenizer.vocab_size
        if use_projected:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + len(rows))
            projection = torch.randn(
                tokenizer.vocab_size,
                projection_dim,
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(float(projection_dim))
            base_rows = _collect_projected_logit_rows(
                model=base_model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                projection=projection,
                row_count=row_count,
            )
            analysis_input_dim = projection_dim
        else:
            projection = None
            base_rows = _collect_model_logit_rows(
                model=base_model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
            )
        properties = compute_adapter_properties(adapter_path, base_model=base_model)
        adapter_metadata = read_adapter_metadata(adapter_path)
        metadata_rank = adapter_metadata.get("r")
        if metadata_rank is not None and int(metadata_rank) != int(adapter_config["rank"]):
            raise ValueError(
                f"Configured rank {adapter_config['rank']} does not match adapter metadata rank {metadata_rank} "
                f"for {experiment['adapter_name']}"
            )
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        adapted_model = _load_adapted_model(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            magnitude_scale=magnitude_scale,
            device=device,
            torch_dtype=torch_dtype,
        )
        if use_projected:
            if projection is None:
                raise RuntimeError("Projection matrix was not initialized for projected mode")
            adapted_rows = _collect_projected_logit_rows(
                model=adapted_model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                projection=projection,
                row_count=row_count,
            )
        else:
            adapted_rows = _collect_model_logit_rows(
                model=adapted_model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
            )
        shift_matrix = adapted_rows - base_rows
        analysis = spectral_analysis(shift_matrix)
        plot_path = analysis.save_spectrum_plot(
            plots_dir / f"{experiment['model_pair_name']}__{experiment['adapter_name']}__spectrum.png",
            title=f"{experiment['model_pair_name']} / {experiment['adapter_name']}",
        )
        rows.append(
            {
                "model_pair_name": experiment["model_pair_name"],
                "adapter_name": experiment["adapter_name"],
                "adapter_rank": int(adapter_config["rank"]),
                "adapter_domain": str(adapter_config["domain"]),
                "adapter_epochs": adapter_config.get("epochs"),
                "adapter_path": adapter_path,
                "magnitude_scale": magnitude_scale,
                "effective_rank": effective_rank(shift_matrix, threshold=energy_threshold),
                "effective_rank_95": analysis.effective_rank_95,
                "effective_rank_99": analysis.effective_rank_99,
                "stable_rank": analysis.stable_rank,
                "participation_ratio": analysis.participation_ratio,
                "num_rows": int(shift_matrix.shape[0]),
                "vocab_size": int(tokenizer.vocab_size),
                "analysis_dimension": int(analysis_input_dim),
                "rank_estimation_mode": "projected" if use_projected else "exact",
                "spectrum_is_approximate": bool(use_projected),
                "logit_gauge": "row_mean_centered_before_projection",
                "spectral_analysis": {
                    "singular_values": analysis.singular_values,
                    "cumulative_energy": analysis.cumulative_energy,
                },
                "adapter_properties": {
                    "frobenius_norm_sum": properties.frobenius_norm_sum * magnitude_scale,
                    "spectral_norm_sum": properties.spectral_norm_sum * magnitude_scale,
                    "max_spectral_norm": properties.max_spectral_norm * magnitude_scale,
                    "adapted_parameter_count": properties.adapted_parameter_count,
                    "adapted_parameter_fraction": properties.adapted_parameter_fraction,
                },
                "spectrum_plot": str(plot_path),
                "notes": experiment["notes"],
                "tags": experiment["tags"],
            }
        )
        del adapted_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "experiment_type": "measure_logit_shift_rank",
        "rows": rows,
        "summary": {
            "num_experiments": len(rows),
            "energy_threshold": energy_threshold,
            "prompts_file": prompts_file,
            "batch_size": batch_size,
            "logit_gauge": "row_mean_centered_before_projection",
        },
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="measure_logit_shift_rank",
        config={
            "models_config": models_config,
            "adapters_config": adapters_config,
            "model_pair": model_pair,
            "adapter_names": list(adapter_names or []),
            "prompts_file": prompts_file,
            "batch_size": batch_size,
            "energy_threshold": energy_threshold,
            "torch_dtype": torch_dtype_name,
            "rank_estimation_mode": rank_estimation_mode,
            "projection_dim": projection_dim,
            "max_matrix_gb": max_matrix_gb,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved effective-rank measurements to %s", output)


if __name__ == "__main__":
    main()
