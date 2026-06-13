from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lora_spec.artifacts import resolve_artifact_revision
from lora_spec.prompts import load_frozen_prompt_texts, prompt_file_provenance
from lora_spec.theory import dominant_subspace_basis, subspace_overlap_from_bases
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    load_yaml,
    resolve_config,
    resolve_torch_dtype,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure dominant-subspace overlap across adapters.")
    add_common_args(parser)
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--adapters-config", type=str, default="configs/adapters.yaml")
    parser.add_argument("--model-pair", type=str, default=None)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/calibration.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overlap-rank", type=int, default=8)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--rank-estimation-mode", choices=["auto", "exact", "projected"], default="auto")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--max-matrix-gb", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="results/theory")
    return parser.parse_args()


def _load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    revision: str | None = None,
) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def _load_tokenizer(model_name: str, revision: str | None = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _count_rows(tokenizer: PreTrainedTokenizerBase, prompts: list[str], batch_size: int) -> int:
    total = 0
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True)
        total += int(encoded["attention_mask"][:, 1:].sum().item())
    return total


@torch.inference_mode()
def _collect_rows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
    row_count: int,
    projection: torch.Tensor | None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    output_dim = projection.shape[1] if projection is not None else tokenizer.vocab_size
    rows = torch.empty((row_count, output_dim), dtype=torch.float32)
    projection_device = projection.to(device) if projection is not None else None
    offset = 0
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        logits = model(**encoded).logits[:, :-1, :].float()
        logits = logits - logits.mean(dim=-1, keepdim=True)
        if projection_device is not None:
            logits = logits @ projection_device
        mask = encoded["attention_mask"][:, 1:].bool()
        for batch_index in range(mask.shape[0]):
            valid = logits[batch_index, mask[batch_index]].detach().cpu()
            rows[offset : offset + valid.shape[0]] = valid
            offset += valid.shape[0]
    if offset != row_count:
        raise RuntimeError(f"Logit row count mismatch: expected {row_count}, got {offset}")
    return rows


def _apply_lora_scale(model: PreTrainedModel, scale: float) -> None:
    if scale == 1.0:
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.mul_(scale)


def _select_adapters(adapter_payload: dict[str, Any], model_pair: str) -> list[str]:
    adapters = adapter_payload["adapters"]
    experiments = adapter_payload.get("experiments", [])
    selected = sorted({str(entry["adapter"]) for entry in experiments if entry["model_pair"] == model_pair})
    return selected or sorted(adapters)


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "subspace_sharing")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)

    models_config = str(get_config_value(config_data, args, "models_config"))
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    model_pair_value = get_config_value(config_data, args, "model_pair")
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not model_pair_value or not prompts_file_value:
        raise ValueError("model_pair and prompts_file must be provided")
    model_pair = str(model_pair_value)
    prompts_file = str(prompts_file_value)
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    overlap_rank = int(get_config_value(config_data, args, "overlap_rank"))
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    rank_estimation_mode = str(get_config_value(config_data, args, "rank_estimation_mode"))
    projection_dim = int(get_config_value(config_data, args, "projection_dim"))
    max_matrix_gb = float(get_config_value(config_data, args, "max_matrix_gb"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    if overlap_rank < 1:
        raise ValueError("overlap_rank must be positive")
    if projection_dim < overlap_rank:
        raise ValueError("projection_dim must be at least overlap_rank")

    model_pairs = load_yaml(models_config)["model_pairs"]
    adapter_payload = load_yaml(adapters_config)
    if model_pair not in model_pairs:
        raise KeyError(f"Unknown model pair {model_pair}")
    adapters = adapter_payload["adapters"]
    adapter_names = _select_adapters(adapter_payload, model_pair)
    prompts = load_frozen_prompt_texts(prompts_file, expected_split="calibration")
    prompts_provenance = prompt_file_provenance(prompts_file, expected_split="calibration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_torch_dtype(torch_dtype_name, device=device)
    base_model_name = str(model_pairs[model_pair]["target_model"])
    base_artifact = resolve_artifact_revision(
        base_model_name,
        revision=model_pairs[model_pair].get("target_revision"),
    )
    tokenizer = _load_tokenizer(base_model_name, revision=base_artifact.revision_for_loading)
    row_count = _count_rows(tokenizer, prompts, batch_size)
    estimated_dense_gb = row_count * tokenizer.vocab_size * 4.0 / (1024.0**3)
    use_projection = rank_estimation_mode == "projected" or (
        rank_estimation_mode == "auto" and estimated_dense_gb > max_matrix_gb
    )
    projection: torch.Tensor | None = None
    if use_projection:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        projection = torch.randn(
            tokenizer.vocab_size,
            projection_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(projection_dim))

    logger.info("Collecting base logits for %s using %s analysis", model_pair, "projected" if use_projection else "exact")
    base_model = _load_model(
        base_model_name,
        device,
        dtype,
        revision=base_artifact.revision_for_loading,
    )
    base_rows = _collect_rows(base_model, tokenizer, prompts, batch_size, row_count, projection)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bases: dict[str, torch.Tensor] = {}
    adapter_sources: dict[str, str] = {}
    adapter_provenance: dict[str, dict[str, str | None]] = {}
    for adapter_name in adapter_names:
        adapter_config = adapters[adapter_name]
        compatible_target = adapter_config.get("target_model")
        if compatible_target and str(compatible_target) != base_model_name:
            raise ValueError(f"Adapter {adapter_name} targets {compatible_target}, not {base_model_name}")
        logger.info("Computing dominant subspace for %s", adapter_name)
        artifact = resolve_artifact_revision(
            str(adapter_config["hf_path"]),
            revision=adapter_config.get("revision"),
        )
        adapted_model = PeftModel.from_pretrained(
            _load_model(
                base_model_name,
                device,
                dtype,
                revision=base_artifact.revision_for_loading,
            ),
            str(adapter_config["hf_path"]),
            revision=artifact.revision_for_loading,
        ).to(device).eval()
        _apply_lora_scale(adapted_model, float(adapter_config.get("magnitude_scale", 1.0)))
        adapted_rows = _collect_rows(adapted_model, tokenizer, prompts, batch_size, row_count, projection)
        shift = adapted_rows - base_rows
        bases[adapter_name] = dominant_subspace_basis(shift, rank=overlap_rank)
        adapter_sources[adapter_name] = str(adapter_config["hf_path"])
        adapter_provenance[adapter_name] = artifact.to_dict()
        del adapted_rows, shift, adapted_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for adapter_a, adapter_b in itertools.combinations(sorted(bases), 2):
        overlap = subspace_overlap_from_bases(bases[adapter_a], bases[adapter_b])
        rows.append(
            {
                "adapter_a": adapter_a,
                "adapter_b": adapter_b,
                "rank": overlap.rank,
                "mean_cosine": overlap.mean_cosine,
                "chordal_distance": overlap.chordal_distance,
                "principal_angles_degrees": overlap.principal_angles_degrees,
                "same_adapter_source": adapter_sources[adapter_a] == adapter_sources[adapter_b],
            }
        )

    payload = {
        "experiment_type": "subspace_sharing",
        "model_pair": model_pair,
        "rows": rows,
        "rank_estimation_mode": "projected" if use_projection else "exact",
        "subspace_is_approximate": use_projection,
        "analysis_dimension": projection_dim if use_projection else tokenizer.vocab_size,
        "num_context_rows": row_count,
        "estimated_dense_matrix_gb": estimated_dense_gb,
        "logit_gauge": "row_mean_centered_before_projection",
        "unique_adapter_sources": len(set(adapter_sources.values())),
        "artifact_provenance": {
            "base_model": base_artifact.to_dict(),
            "adapters": adapter_provenance,
        },
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="subspace_sharing",
        config={
            "models_config": models_config,
            "adapters_config": adapters_config,
            "model_pair": model_pair,
            "prompts_file": prompts_file,
            "prompts_provenance": prompts_provenance,
            "batch_size": batch_size,
            "overlap_rank": overlap_rank,
            "torch_dtype": torch_dtype_name,
            "rank_estimation_mode": rank_estimation_mode,
            "projection_dim": projection_dim,
            "max_matrix_gb": max_matrix_gb,
            "artifact_provenance": {
                "base_model": base_artifact.to_dict(),
                "adapters": adapter_provenance,
            },
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved subspace-sharing analysis to %s", output)


if __name__ == "__main__":
    main()
