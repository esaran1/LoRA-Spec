from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from lora_spec.artifacts import resolve_artifact_revision
from lora_spec.design import audit_experiment_design
from lora_spec.prompts import load_frozen_prompt_texts, prompt_file_provenance
from lora_spec.theory import (
    ContinuationContextSet,
    build_continuation_contexts,
    effective_rank,
    iter_continuation_context_batches,
    spectral_analysis,
    spectral_sample_size_sensitivity,
)
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
    parser = argparse.ArgumentParser(
        description="Measure effective rank of the adapter-induced logit shift."
    )
    add_common_args(parser)
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--adapters-config", type=str, default="configs/adapters.yaml")
    parser.add_argument("--model-pair", type=str, default=None)
    parser.add_argument("--adapter-name", action="append", default=[])
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--target-revision", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--adapter-revision", type=str, default=None)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-domain", type=str, default="unknown")
    parser.add_argument("--magnitude-scale", type=float, default=1.0)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/calibration.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument(
        "--rank-estimation-mode", type=str, choices=["auto", "exact", "projected"], default="auto"
    )
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-dimensions", type=str, default=None)
    parser.add_argument("--projection-repetitions", type=int, default=3)
    parser.add_argument("--calibration-size-values", type=str, default="4,8,16,32,64")
    parser.add_argument("--calibration-subsample-repetitions", type=int, default=5)
    parser.add_argument("--max-matrix-gb", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="results/theory")
    parser.add_argument("--plots-dir", type=str, default="results/theory/plots")
    parser.add_argument("--require-paper-ready-design", action="store_true")
    parser.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help=(
            "Run a deterministic low-rank synthetic shift through the rank-analysis and "
            "JSON-writing path without importing Transformers, PEFT, or downloading models."
        ),
    )
    return parser.parse_args()


def _load_base_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    revision: str | None = None,
    device_map: str | None = None,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if device_map is None:
        model = model.to(device)
    model = model.eval()
    return model, tokenizer


def _apply_lora_scale(model: Any, scale: float) -> None:
    from lora_spec.adapter_props import scale_plain_lora_adapter

    scale_plain_lora_adapter(model, scale, context="logit-shift rank measurement")


def _parse_projection_dimensions(raw: str | None, primary: int) -> list[int]:
    if raw is None:
        values = sorted({max(2, primary // 2), primary})
    else:
        values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value < 2 for value in values):
        raise ValueError("projection_dimensions must contain integers >= 2")
    if primary not in values:
        values.append(primary)
        values.sort()
    return values


def _parse_calibration_sizes(raw: str | None, prompt_count: int) -> list[int]:
    requested = (
        [int(item.strip()) for item in raw.split(",") if item.strip()] if raw is not None else []
    )
    if any(size < 1 for size in requested):
        raise ValueError("calibration_size_values must contain positive integers")
    return sorted({size for size in requested if size <= prompt_count} | {prompt_count})


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_synthetic_smoke_test(
    args: argparse.Namespace,
    config_data: dict[str, Any],
    seed: int,
    logger: Any,
) -> Path:
    energy_threshold = float(get_config_value(config_data, args, "energy_threshold"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    plots_dir = ensure_dir(str(get_config_value(config_data, args, "plots_dir")))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_rows = 24
    vocab_size = 64
    true_rank = 3
    left = torch.randn(num_rows, true_rank, generator=generator)
    right = torch.randn(true_rank, vocab_size, generator=generator)
    shift_matrix = left @ right
    shift_matrix = shift_matrix - shift_matrix.mean(dim=-1, keepdim=True)

    analysis = spectral_analysis(shift_matrix)
    threshold_rank = effective_rank(shift_matrix, threshold=energy_threshold)
    plot_path = analysis.save_spectrum_plot(
        plots_dir / "synthetic_smoke__spectrum.png",
        title="Synthetic low-rank smoke test",
    )
    payload = {
        "experiment_type": "measure_logit_shift_rank",
        "synthetic_smoke_test": True,
        "rows": [
            {
                "model_pair_name": "synthetic",
                "adapter_name": "synthetic_rank3",
                "adapter_rank": true_rank,
                "adapter_domain": "synthetic",
                "adapter_epochs": None,
                "adapter_path": "synthetic",
                "magnitude_scale": 1.0,
                "effective_rank": int(threshold_rank),
                "effective_rank_95": analysis.effective_rank_95,
                "effective_rank_99": analysis.effective_rank_99,
                "stable_rank": analysis.stable_rank,
                "participation_ratio": analysis.participation_ratio,
                "num_rows": num_rows,
                "vocab_size": vocab_size,
                "analysis_dimension": vocab_size,
                "rank_ceiling": min(num_rows, vocab_size),
                "effective_rank_fraction_of_ceiling": float(
                    threshold_rank / max(min(num_rows, vocab_size), 1)
                ),
                "rank_ceiling_saturated": False,
                "calibration_sample_size_sensitivity": {},
                "sample_size_sensitivity_basis": "synthetic_full_matrix",
                "rank_estimation_mode": "exact",
                "spectrum_is_approximate": False,
                "projection_repetitions": 0,
                "projection_dimension_sensitivity": {},
                "effective_rank_estimates": [int(threshold_rank)],
                "effective_rank_95_estimates": [analysis.effective_rank_95],
                "effective_rank_99_estimates": [analysis.effective_rank_99],
                "effective_rank_99_range": [analysis.effective_rank_99, analysis.effective_rank_99],
                "stable_rank_estimates": [analysis.stable_rank],
                "stable_rank_range": [analysis.stable_rank, analysis.stable_rank],
                "participation_ratio_estimates": [analysis.participation_ratio],
                "participation_ratio_range": [
                    analysis.participation_ratio,
                    analysis.participation_ratio,
                ],
                "logit_gauge": "row_mean_centered",
                "spectral_analysis": {
                    "singular_values": analysis.singular_values,
                    "cumulative_energy": analysis.cumulative_energy,
                },
                "adapter_properties": {
                    "frobenius_norm_sum": float(torch.linalg.matrix_norm(shift_matrix).item()),
                    "spectral_norm_sum": float(analysis.singular_values[0]),
                    "max_spectral_norm": float(analysis.singular_values[0]),
                    "adapted_parameter_count": true_rank * (num_rows + vocab_size),
                    "adapted_parameter_fraction": float(true_rank * (num_rows + vocab_size))
                    / float(num_rows * vocab_size),
                },
                "artifact_provenance": {
                    "base_model": {
                        "source": "synthetic",
                        "repository_type": "local",
                        "requested_revision": None,
                        "resolved_revision": "synthetic",
                    },
                    "adapter": {
                        "source": "synthetic",
                        "repository_type": "local",
                        "requested_revision": None,
                        "resolved_revision": "synthetic",
                    },
                },
                "continuation_contexts_sha256": "synthetic",
                "spectrum_plot": str(plot_path),
                "notes": "MacBook-safe synthetic smoke test",
                "tags": ["smoke", "synthetic"],
            }
        ],
        "summary": {
            "num_experiments": 1,
            "energy_threshold": energy_threshold,
            "prompts_file": "synthetic",
            "prompts_provenance": {"source": "synthetic"},
            "batch_size": 0,
            "continuation_tokens": 0,
            "trajectory_policy": "synthetic_low_rank_matrix",
            "logit_gauge": "row_mean_centered",
            "design_report": None,
            "continuation_contexts_sha256_by_experiment": {
                "synthetic::synthetic_rank3": "synthetic"
            },
        },
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="measure_logit_shift_rank_smoke",
        config={
            "synthetic_smoke_test": True,
            "seed": seed,
            "num_rows": num_rows,
            "vocab_size": vocab_size,
            "true_rank": true_rank,
            "energy_threshold": energy_threshold,
        },
        cwd=Path.cwd(),
    )
    logger.info(
        "Saved synthetic effective-rank smoke test to %s (effective_rank=%d)",
        output,
        threshold_rank,
    )
    return output


def _load_adapted_model(
    base_model_name: str,
    adapter_path: str,
    magnitude_scale: float,
    device: torch.device,
    torch_dtype: torch.dtype,
    base_revision: str | None = None,
    adapter_revision: str | None = None,
    device_map: str | None = None,
) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        revision=base_revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if device_map is None:
        base_model = base_model.to(device)
    base_model = base_model.eval()
    adapted = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        revision=adapter_revision,
    )
    if device_map is None:
        adapted = adapted.to(device)
    adapted = adapted.eval()
    _apply_lora_scale(adapted, magnitude_scale)
    return adapted


@torch.inference_mode()
def _collect_model_logit_rows(
    model: Any,
    tokenizer: Any,
    contexts: ContinuationContextSet,
    batch_size: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    rows: list[torch.Tensor] = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires a pad or EOS token")
    for start, input_ids, attention_mask in iter_continuation_context_batches(
        contexts, batch_size, pad_token_id
    ):
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits.float()
        logits = logits - logits.mean(dim=-1, keepdim=True)
        for batch_index in range(input_ids.shape[0]):
            prompt_index = start + batch_index
            first = contexts.prompt_lengths[prompt_index] - 1
            count = contexts.continuation_lengths[prompt_index]
            rows.extend(logits[batch_index, first : first + count].detach().cpu())
    if not rows:
        raise ValueError("Calibration prompts did not yield any next-token positions")
    return torch.stack(rows)


@torch.inference_mode()
def _collect_projected_logit_rows(
    model: Any,
    tokenizer: Any,
    contexts: ContinuationContextSet,
    batch_size: int,
    projection: torch.Tensor,
    row_count: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    projected_rows = torch.empty((row_count, projection.shape[1]), dtype=torch.float32)
    projection_device: torch.Tensor | None = None
    projection_output_device: torch.device | None = None
    offset = 0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires a pad or EOS token")
    for start, input_ids, attention_mask in iter_continuation_context_batches(
        contexts, batch_size, pad_token_id
    ):
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits.float()
        logits = logits - logits.mean(dim=-1, keepdim=True)
        if projection_device is None or projection_output_device != logits.device:
            projection_device = projection.to(device=logits.device)
            projection_output_device = logits.device
        projected = logits @ projection_device
        for batch_index in range(input_ids.shape[0]):
            prompt_index = start + batch_index
            first = contexts.prompt_lengths[prompt_index] - 1
            count = contexts.continuation_lengths[prompt_index]
            valid = projected[batch_index, first : first + count].detach().cpu()
            projected_rows[offset : offset + count] = valid
            offset += count
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
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)
    if args.synthetic_smoke_test:
        _run_synthetic_smoke_test(args, config_data, seed, logger)
        return

    models_config = str(get_config_value(config_data, args, "models_config"))
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not prompts_file_value:
        raise ValueError("prompts_file must be provided")
    prompts_file = str(prompts_file_value)
    model_pair = get_config_value(config_data, args, "model_pair")
    adapter_names = get_config_value(config_data, args, "adapter_name", args.adapter_name)
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    continuation_tokens = int(get_config_value(config_data, args, "continuation_tokens"))
    energy_threshold = float(get_config_value(config_data, args, "energy_threshold"))
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    device_map_value = get_config_value(config_data, args, "device_map")
    device_map = str(device_map_value) if device_map_value else None
    rank_estimation_mode = str(get_config_value(config_data, args, "rank_estimation_mode"))
    projection_dim = int(get_config_value(config_data, args, "projection_dim"))
    projection_dimensions = _parse_projection_dimensions(
        get_config_value(config_data, args, "projection_dimensions"),
        projection_dim,
    )
    projection_repetitions = int(get_config_value(config_data, args, "projection_repetitions"))
    max_matrix_gb = float(get_config_value(config_data, args, "max_matrix_gb"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    plots_dir = ensure_dir(str(get_config_value(config_data, args, "plots_dir")))
    require_paper_ready = bool(
        get_config_value(
            config_data,
            args,
            "require_paper_ready_design",
            args.require_paper_ready_design,
        )
    )

    prompts = load_frozen_prompt_texts(prompts_file, expected_split="calibration")
    calibration_sizes = _parse_calibration_sizes(
        get_config_value(config_data, args, "calibration_size_values"),
        len(prompts),
    )
    calibration_subsample_repetitions = int(
        get_config_value(config_data, args, "calibration_subsample_repetitions")
    )
    if calibration_subsample_repetitions < 1:
        raise ValueError("calibration_subsample_repetitions must be positive")
    prompts_provenance = prompt_file_provenance(prompts_file, expected_split="calibration")
    target_model_value = get_config_value(config_data, args, "target_model")
    adapter_path_value = get_config_value(config_data, args, "adapter_path")
    if bool(target_model_value) != bool(adapter_path_value):
        raise ValueError("target_model and adapter_path must be provided together for direct mode")
    if target_model_value and adapter_path_value:
        design_report = None
        adapter_rank = get_config_value(config_data, args, "adapter_rank")
        if adapter_rank is None:
            raise ValueError("adapter_rank is required in direct mode")
        experiments = [
            {
                "model_pair_name": str(model_pair or "direct"),
                "adapter_name": str((adapter_names or ["direct"])[0]),
                "model_config": {
                    "target_model": str(target_model_value),
                    "target_revision": get_config_value(config_data, args, "target_revision"),
                },
                "adapter_config": {
                    "rank": int(adapter_rank),
                    "domain": str(get_config_value(config_data, args, "adapter_domain")),
                    "epochs": None,
                    "hf_path": str(adapter_path_value),
                    "revision": get_config_value(config_data, args, "adapter_revision"),
                    "target_model": str(target_model_value),
                    "magnitude_scale": float(
                        get_config_value(config_data, args, "magnitude_scale")
                    ),
                },
                "notes": "direct CLI experiment",
                "tags": ["direct"],
            }
        ]
    else:
        design_report = audit_experiment_design(
            load_yaml(adapters_config),
            load_yaml(models_config),
        )
        if require_paper_ready and not design_report.paper_ready:
            raise ValueError(
                "Adapter configuration is not paper-ready; run "
                "scripts/validate_experiment_design.py for details"
            )
        experiments = _select_experiments(
            models_config=models_config,
            adapters_config=adapters_config,
            selected_model_pair=str(model_pair) if model_pair else None,
            selected_adapters=list(adapter_names or []),
        )
    if not experiments:
        raise ValueError("No experiments matched the requested model/adapter selection")
    if projection_dim < 2 or projection_repetitions < 2:
        raise ValueError("projection_dim and projection_repetitions must both be at least 2")
    maximum_projection_dim = max(projection_dimensions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(torch_dtype_name, device=device)
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        model_config = dict(experiment["model_config"])
        adapter_config = dict(experiment["adapter_config"])
        base_model_name = str(model_config["target_model"])
        adapter_path = str(adapter_config["hf_path"])
        base_artifact = resolve_artifact_revision(
            base_model_name,
            revision=model_config.get("target_revision"),
        )
        adapter_artifact = resolve_artifact_revision(
            adapter_path,
            revision=adapter_config.get("revision"),
        )
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
            revision=base_artifact.revision_for_loading,
            device_map=device_map,
        )
        contexts = build_continuation_contexts(
            base_model,
            tokenizer,
            prompts,
            max_new_tokens=continuation_tokens,
        )
        row_count = contexts.num_positions
        vocabulary_size = int(base_model.config.vocab_size)
        estimated_matrix_gb = (row_count * vocabulary_size * 4.0) / (1024.0**3)
        use_projected = rank_estimation_mode == "projected" or (
            rank_estimation_mode == "auto" and estimated_matrix_gb > max_matrix_gb
        )
        analysis_input_dim = vocabulary_size
        if use_projected:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            projection = torch.randn(
                vocabulary_size,
                maximum_projection_dim * projection_repetitions,
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(float(maximum_projection_dim))
            base_rows = _collect_projected_logit_rows(
                model=base_model,
                tokenizer=tokenizer,
                contexts=contexts,
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
                contexts=contexts,
                batch_size=batch_size,
            )
        from lora_spec.adapter_props import compute_adapter_properties, read_adapter_metadata

        properties = compute_adapter_properties(
            adapter_path,
            base_model=base_model,
            revision=adapter_artifact.revision_for_loading,
        )
        adapter_metadata = read_adapter_metadata(
            adapter_path,
            revision=adapter_artifact.revision_for_loading,
        )
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
            base_revision=base_artifact.revision_for_loading,
            adapter_revision=adapter_artifact.revision_for_loading,
            device_map=device_map,
        )
        if use_projected:
            if projection is None:
                raise RuntimeError("Projection matrix was not initialized for projected mode")
            adapted_rows = _collect_projected_logit_rows(
                model=adapted_model,
                tokenizer=tokenizer,
                contexts=contexts,
                batch_size=batch_size,
                projection=projection,
                row_count=row_count,
            )
        else:
            adapted_rows = _collect_model_logit_rows(
                model=adapted_model,
                tokenizer=tokenizer,
                contexts=contexts,
                batch_size=batch_size,
            )
        combined_shift = adapted_rows - base_rows
        if use_projected:
            maximum_blocks = list(torch.split(combined_shift, maximum_projection_dim, dim=1))
            shift_matrices = [
                block[:, :projection_dim]
                * math.sqrt(float(maximum_projection_dim) / float(projection_dim))
                for block in maximum_blocks
            ]
            analyses = [spectral_analysis(matrix) for matrix in shift_matrices]
            threshold_ranks = [
                effective_rank(matrix, threshold=energy_threshold) for matrix in shift_matrices
            ]
            median_index = sorted(
                range(len(analyses)),
                key=lambda index: threshold_ranks[index],
            )[len(analyses) // 2]
            shift_matrix = shift_matrices[median_index]
            analysis = analyses[median_index]
        else:
            shift_matrix = combined_shift
            analyses = [spectral_analysis(shift_matrix)]
            threshold_ranks = [effective_rank(shift_matrix, threshold=energy_threshold)]
        rank_95_estimates = [item.effective_rank_95 for item in analyses]
        rank_99_estimates = [item.effective_rank_99 for item in analyses]
        stable_rank_estimates = [item.stable_rank for item in analyses]
        participation_ratio_estimates = [item.participation_ratio for item in analyses]
        prompt_indices = [
            prompt_index
            for prompt_index, continuation_length in enumerate(contexts.continuation_lengths)
            for _ in range(continuation_length)
        ]
        sample_size_sensitivity = spectral_sample_size_sensitivity(
            shift_matrix,
            prompt_indices,
            calibration_sizes,
            seed=seed,
            threshold=energy_threshold,
            repetitions=calibration_subsample_repetitions,
        )
        rank_ceiling = min(shift_matrix.shape)
        median_effective_rank = int(round(float(torch.tensor(threshold_ranks).median().item())))
        projection_sensitivity: dict[str, dict[str, list[float] | list[int]]] = {}
        if use_projected:
            for sensitivity_dimension in projection_dimensions:
                dimension_matrices = [
                    block[:, :sensitivity_dimension]
                    * math.sqrt(float(maximum_projection_dim) / float(sensitivity_dimension))
                    for block in maximum_blocks
                ]
                dimension_analyses = [spectral_analysis(matrix) for matrix in dimension_matrices]
                projection_sensitivity[str(sensitivity_dimension)] = {
                    "effective_rank_95": [item.effective_rank_95 for item in dimension_analyses],
                    "effective_rank_99": [item.effective_rank_99 for item in dimension_analyses],
                    "stable_rank": [item.stable_rank for item in dimension_analyses],
                    "participation_ratio": [
                        item.participation_ratio for item in dimension_analyses
                    ],
                }
        plot_path = analysis.save_spectrum_plot(
            plots_dir
            / f"{experiment['model_pair_name']}__{experiment['adapter_name']}__spectrum.png",
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
                "effective_rank": median_effective_rank,
                "effective_rank_95": analysis.effective_rank_95,
                "effective_rank_99": analysis.effective_rank_99,
                "stable_rank": analysis.stable_rank,
                "participation_ratio": analysis.participation_ratio,
                "num_rows": int(shift_matrix.shape[0]),
                "vocab_size": vocabulary_size,
                "analysis_dimension": int(analysis_input_dim),
                "rank_ceiling": int(rank_ceiling),
                "effective_rank_fraction_of_ceiling": float(
                    median_effective_rank / max(rank_ceiling, 1)
                ),
                "rank_ceiling_saturated": bool(median_effective_rank >= 0.9 * rank_ceiling),
                "calibration_sample_size_sensitivity": sample_size_sensitivity,
                "sample_size_sensitivity_basis": (
                    "representative_projection_sketch" if use_projected else "exact_shift_matrix"
                ),
                "rank_estimation_mode": "projected" if use_projected else "exact",
                "spectrum_is_approximate": bool(use_projected),
                "projection_repetitions": projection_repetitions if use_projected else 0,
                "projection_dimension_sensitivity": projection_sensitivity,
                "effective_rank_estimates": threshold_ranks,
                "effective_rank_95_estimates": rank_95_estimates,
                "effective_rank_99_estimates": rank_99_estimates,
                "effective_rank_99_range": [min(rank_99_estimates), max(rank_99_estimates)],
                "stable_rank_estimates": stable_rank_estimates,
                "stable_rank_range": [
                    min(stable_rank_estimates),
                    max(stable_rank_estimates),
                ],
                "participation_ratio_estimates": participation_ratio_estimates,
                "participation_ratio_range": [
                    min(participation_ratio_estimates),
                    max(participation_ratio_estimates),
                ],
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
                "artifact_provenance": {
                    "base_model": base_artifact.to_dict(),
                    "adapter": adapter_artifact.to_dict(),
                },
                "continuation_contexts_sha256": contexts.sha256(),
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
            "prompts_provenance": prompts_provenance,
            "batch_size": batch_size,
            "continuation_tokens": continuation_tokens,
            "trajectory_policy": f"greedy_max_new_tokens_{continuation_tokens}",
            "logit_gauge": "row_mean_centered_before_projection",
            "design_report": design_report.to_dict() if design_report else None,
            "continuation_contexts_sha256_by_experiment": {
                f"{row['model_pair_name']}::{row['adapter_name']}": row[
                    "continuation_contexts_sha256"
                ]
                for row in rows
            },
        },
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="measure_logit_shift_rank",
        config={
            "models_config": models_config,
            "adapters_config": adapters_config,
            "models_config_sha256": _file_sha256(models_config),
            "adapters_config_sha256": _file_sha256(adapters_config),
            "model_pair": model_pair,
            "adapter_names": list(adapter_names or []),
            "prompts_file": prompts_file,
            "prompts_provenance": prompts_provenance,
            "batch_size": batch_size,
            "continuation_tokens": continuation_tokens,
            "energy_threshold": energy_threshold,
            "torch_dtype": torch_dtype_name,
            "device_map": device_map,
            "rank_estimation_mode": rank_estimation_mode,
            "projection_dim": projection_dim,
            "projection_dimensions": projection_dimensions,
            "projection_repetitions": projection_repetitions,
            "calibration_size_values": calibration_sizes,
            "calibration_subsample_repetitions": calibration_subsample_repetitions,
            "max_matrix_gb": max_matrix_gb,
            "require_paper_ready_design": require_paper_ready,
            "design_report": design_report.to_dict() if design_report else None,
            "continuation_contexts_sha256_by_experiment": {
                f"{row['model_pair_name']}::{row['adapter_name']}": row[
                    "continuation_contexts_sha256"
                ]
                for row in rows
            },
            "seed": seed,
            "experiment_definitions": [
                {
                    "model_pair_name": row["model_pair_name"],
                    "adapter_name": row["adapter_name"],
                    "adapter_rank": row["adapter_rank"],
                    "adapter_domain": row["adapter_domain"],
                    "adapter_epochs": row["adapter_epochs"],
                    "magnitude_scale": row["magnitude_scale"],
                }
                for row in rows
            ],
            "artifact_provenance_by_experiment": [
                {
                    "model_pair_name": row["model_pair_name"],
                    "adapter_name": row["adapter_name"],
                    "artifact_provenance": row["artifact_provenance"],
                }
                for row in rows
            ],
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved effective-rank measurements to %s", output)


if __name__ == "__main__":
    main()
