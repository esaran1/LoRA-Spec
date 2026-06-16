from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from lora_spec.adapter_props import scale_plain_lora_adapter
from lora_spec.artifacts import resolve_artifact_revision
from lora_spec.prompts import file_sha256, load_frozen_prompt_texts, prompt_file_provenance
from lora_spec.theory import (
    ContinuationContextSet,
    build_continuation_contexts,
    dominant_subspace_basis,
    iter_continuation_context_batches,
    subspace_overlap_from_bases,
)
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
    parser = argparse.ArgumentParser(
        description="Measure dominant-subspace overlap across adapters."
    )
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
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--overlap-rank", type=int, default=8)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument(
        "--rank-estimation-mode", choices=["auto", "exact", "projected"], default="auto"
    )
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
    return (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )


def _load_tokenizer(model_name: str, revision: str | None = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.inference_mode()
def _collect_rows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    contexts: ContinuationContextSet,
    batch_size: int,
    row_count: int,
    projection: torch.Tensor | None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    output_dim = projection.shape[1] if projection is not None else int(model.config.vocab_size)
    rows = torch.empty((row_count, output_dim), dtype=torch.float32)
    projection_device = projection.to(device) if projection is not None else None
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
        if projection_device is not None:
            logits = logits @ projection_device
        for batch_index in range(input_ids.shape[0]):
            prompt_index = start + batch_index
            first = contexts.prompt_lengths[prompt_index] - 1
            count = contexts.continuation_lengths[prompt_index]
            valid = logits[batch_index, first : first + count].detach().cpu()
            rows[offset : offset + valid.shape[0]] = valid
            offset += valid.shape[0]
    if offset != row_count:
        raise RuntimeError(f"Logit row count mismatch: expected {row_count}, got {offset}")
    return rows


def _apply_lora_scale(model: PreTrainedModel, scale: float) -> None:
    scale_plain_lora_adapter(model, scale, context="adapter subspace measurement")


def _select_adapters(adapter_payload: dict[str, Any], model_pair: str) -> list[str]:
    adapters = adapter_payload["adapters"]
    experiments = adapter_payload.get("experiments", [])
    selected = sorted(
        {str(entry["adapter"]) for entry in experiments if entry["model_pair"] == model_pair}
    )
    return selected or sorted(adapters)


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "subspace_sharing")
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)

    models_config = str(get_config_value(config_data, args, "models_config"))
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    model_pair_value = get_config_value(config_data, args, "model_pair")
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not model_pair_value or not prompts_file_value:
        raise ValueError("model_pair and prompts_file must be provided")
    model_pair = str(model_pair_value)
    prompts_file = str(prompts_file_value)
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    continuation_tokens = int(get_config_value(config_data, args, "continuation_tokens"))
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
    base_model = _load_model(
        base_model_name,
        device,
        dtype,
        revision=base_artifact.revision_for_loading,
    )
    contexts = build_continuation_contexts(
        base_model,
        tokenizer,
        prompts,
        max_new_tokens=continuation_tokens,
    )
    row_count = contexts.num_positions
    vocabulary_size = int(base_model.config.vocab_size)
    estimated_dense_gb = row_count * vocabulary_size * 4.0 / (1024.0**3)
    use_projection = rank_estimation_mode == "projected" or (
        rank_estimation_mode == "auto" and estimated_dense_gb > max_matrix_gb
    )
    projection: torch.Tensor | None = None
    if use_projection:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projection = torch.randn(
            vocabulary_size,
            projection_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(projection_dim))
    logger.info(
        "Collecting base logits for %s using %s analysis",
        model_pair,
        "projected" if use_projection else "exact",
    )
    base_rows = _collect_rows(base_model, tokenizer, contexts, batch_size, row_count, projection)
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
            raise ValueError(
                f"Adapter {adapter_name} targets {compatible_target}, not {base_model_name}"
            )
        logger.info("Computing dominant subspace for %s", adapter_name)
        artifact = resolve_artifact_revision(
            str(adapter_config["hf_path"]),
            revision=adapter_config.get("revision"),
        )
        adapted_model = (
            PeftModel.from_pretrained(
                _load_model(
                    base_model_name,
                    device,
                    dtype,
                    revision=base_artifact.revision_for_loading,
                ),
                str(adapter_config["hf_path"]),
                revision=artifact.revision_for_loading,
            )
            .to(device)
            .eval()
        )
        _apply_lora_scale(adapted_model, float(adapter_config.get("magnitude_scale", 1.0)))
        adapted_rows = _collect_rows(
            adapted_model, tokenizer, contexts, batch_size, row_count, projection
        )
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
                "rank_a": overlap.rank_a,
                "rank_b": overlap.rank_b,
                "mean_cosine": overlap.mean_cosine,
                "chordal_distance": overlap.chordal_distance,
                "overlap_fraction_a": overlap.overlap_fraction_a,
                "overlap_fraction_b": overlap.overlap_fraction_b,
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
        "analysis_dimension": projection_dim if use_projection else vocabulary_size,
        "num_context_rows": row_count,
        "continuation_contexts_sha256": contexts.sha256(),
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
            "models_config_sha256": file_sha256(models_config),
            "adapters_config_sha256": file_sha256(adapters_config),
            "model_pair": model_pair,
            "prompts_file": prompts_file,
            "prompts_provenance": prompts_provenance,
            "batch_size": batch_size,
            "continuation_tokens": continuation_tokens,
            "trajectory_policy": contexts.generation_policy,
            "continuation_contexts_sha256": contexts.sha256(),
            "overlap_rank": overlap_rank,
            "torch_dtype": torch_dtype_name,
            "rank_estimation_mode": rank_estimation_mode,
            "projection_dim": projection_dim,
            "max_matrix_gb": max_matrix_gb,
            "seed": seed,
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
