from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_spec.adapter_props import scale_plain_lora_adapter
from lora_spec.artifacts import resolve_artifact_revision, tokenizers_are_equivalent
from lora_spec.correction import LowRankCorrection
from lora_spec.metrics import simulate_speculative_decoding
from lora_spec.prompts import load_frozen_prompt_texts, prompt_file_provenance
from lora_spec.statistical import cluster_bootstrap_mean, fit_continuous_piecewise_linear
from lora_spec.theory import (
    ContinuationContextSet,
    build_continuation_contexts,
    center_logit_shift_rows,
    collect_context_model_outputs,
    collect_logit_shift_dataset,
    compute_logit_shift_matrix,
    first_order_logit_shift,
    nonlinearity_residual,
    parameter_delta_from_models,
    spectral_analysis,
)
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    resolve_torch_dtype,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep adapter magnitude to detect analytical-correction phase transitions."
    )
    add_common_args(parser)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--base-revision", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--draft-revision", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--adapter-revision", type=str, default=None)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/calibration.jsonl",
    )
    parser.add_argument(
        "--eval-prompts-file",
        type=str,
        default="data/prompts/pilot_v1/evaluation.jsonl",
    )
    parser.add_argument(
        "--magnitude-values", type=str, default="0.0,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0"
    )
    parser.add_argument("--correction-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--jvp-max-tangent-mb", type=int, default=512)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="results/theory")
    return parser.parse_args()


def _parse_values(raw: str) -> list[float]:
    values = sorted(float(piece.strip()) for piece in raw.split(",") if piece.strip())
    if not values:
        raise ValueError("magnitude_values must contain at least one float")
    if any(value < 0.0 for value in values):
        raise ValueError("magnitude_values must be non-negative")
    if len(set(values)) != len(values):
        raise ValueError("magnitude_values must be unique")
    return values


def _load_model(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    revision: str | None = None,
) -> Any:
    return (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )


def _load_tokenizer(model_name: str, revision: str | None = None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _apply_lora_scale(model: Any, scale: float) -> None:
    scale_plain_lora_adapter(model, scale, context="phase-transition sweep")


def _load_scaled_adapted_model(
    base_model_name: str,
    adapter_path: str,
    scale: float,
    device: torch.device,
    torch_dtype: torch.dtype,
    base_revision: str | None = None,
    adapter_revision: str | None = None,
) -> Any:
    base_model = _load_model(
        base_model_name,
        device,
        torch_dtype=torch_dtype,
        revision=base_revision,
    )
    adapted = (
        PeftModel.from_pretrained(
            base_model,
            adapter_path,
            revision=adapter_revision,
        )
        .to(device)
        .eval()
    )
    _apply_lora_scale(adapted, scale)
    return adapted


def _prepare_prompt_input_ids(tokenizer: Any, prompts: list[str]) -> list[torch.Tensor]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    sequences: list[torch.Tensor] = []
    for index in range(encoded["input_ids"].shape[0]):
        length = int(encoded["attention_mask"][index].sum().item())
        if length >= 1:
            sequences.append(encoded["input_ids"][index, :length].clone())
    return sequences


def _heldout_normalized_logit_error(
    correction: Any,
    draft_model: Any,
    adapted_model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    continuation_contexts: ContinuationContextSet,
) -> tuple[float, list[float], list[int]]:
    _ = prompts
    draft_logits, _, prompt_indices, _ = collect_context_model_outputs(
        draft_model,
        tokenizer,
        continuation_contexts,
        batch_size=batch_size,
    )
    adapted_logits, _, adapted_prompt_indices, _ = collect_context_model_outputs(
        adapted_model,
        tokenizer,
        continuation_contexts,
        batch_size=batch_size,
    )
    if prompt_indices != adapted_prompt_indices:
        raise RuntimeError("Draft and adapted held-out rows are not prompt-aligned")
    adjusted_logits = correction.apply(draft_logits)
    baseline_delta = center_logit_shift_rows(draft_logits - adapted_logits)
    corrected_delta = center_logit_shift_rows(adjusted_logits - adapted_logits)
    global_error = float(
        torch.linalg.matrix_norm(corrected_delta, ord="fro").item()
        / max(torch.linalg.matrix_norm(baseline_delta, ord="fro").item(), 1e-12)
    )
    per_prompt_errors: list[float] = []
    unique_prompt_indices = sorted(set(prompt_indices))
    prompt_index_tensor = torch.tensor(prompt_indices, dtype=torch.long)
    for prompt_index in unique_prompt_indices:
        mask = prompt_index_tensor == prompt_index
        numerator = torch.linalg.matrix_norm(corrected_delta[mask], ord="fro").item()
        denominator = torch.linalg.matrix_norm(baseline_delta[mask], ord="fro").item()
        per_prompt_errors.append(float(numerator / max(denominator, 1e-12)))
    return global_error, per_prompt_errors, unique_prompt_indices


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "phase_transition_sweep")
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)

    base_model_value = get_config_value(config_data, args, "base_model")
    draft_model_value = get_config_value(config_data, args, "draft_model")
    adapter_path_value = get_config_value(config_data, args, "adapter_path")
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not all([base_model_value, draft_model_value, adapter_path_value, prompts_file_value]):
        raise ValueError("base_model, draft_model, adapter_path, and prompts_file must be provided")
    base_model_name = str(base_model_value)
    draft_model_name = str(draft_model_value)
    adapter_path = str(adapter_path_value)
    prompts_file = str(prompts_file_value)
    base_revision = get_config_value(config_data, args, "base_revision")
    draft_revision = get_config_value(config_data, args, "draft_revision")
    adapter_revision = get_config_value(config_data, args, "adapter_revision")
    eval_prompts_file_value = get_config_value(config_data, args, "eval_prompts_file")
    if not eval_prompts_file_value:
        raise ValueError("eval_prompts_file must be provided separately from prompts_file")
    eval_prompts_file = str(eval_prompts_file_value)
    magnitude_values = _parse_values(str(get_config_value(config_data, args, "magnitude_values")))
    correction_rank = int(get_config_value(config_data, args, "correction_rank"))
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    speculation_length = int(get_config_value(config_data, args, "speculation_length"))
    max_new_tokens = int(get_config_value(config_data, args, "max_new_tokens"))
    continuation_tokens = int(get_config_value(config_data, args, "continuation_tokens"))
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    jvp_max_tangent_mb = int(get_config_value(config_data, args, "jvp_max_tangent_mb"))
    bootstrap_repetitions = int(get_config_value(config_data, args, "bootstrap_repetitions"))
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    calibration_prompts = load_frozen_prompt_texts(prompts_file, expected_split="calibration")
    eval_prompts = load_frozen_prompt_texts(eval_prompts_file, expected_split="evaluation")
    prompts_provenance = prompt_file_provenance(prompts_file, expected_split="calibration")
    eval_prompts_provenance = prompt_file_provenance(
        eval_prompts_file,
        expected_split="evaluation",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(torch_dtype_name, device=device)
    base_artifact = resolve_artifact_revision(base_model_name, revision=base_revision)
    draft_artifact = resolve_artifact_revision(draft_model_name, revision=draft_revision)
    adapter_artifact = resolve_artifact_revision(adapter_path, revision=adapter_revision)
    tokenizer = _load_tokenizer(base_model_name, revision=base_artifact.revision_for_loading)
    draft_tokenizer = _load_tokenizer(
        draft_model_name, revision=draft_artifact.revision_for_loading
    )
    if not tokenizers_are_equivalent(
        tokenizer, draft_tokenizer, calibration_prompts + eval_prompts
    ):
        raise ValueError("draft_model tokenizer must be compatible with base_model tokenizer")
    base_model = _load_model(
        base_model_name,
        device,
        torch_dtype=torch_dtype,
        revision=base_artifact.revision_for_loading,
    )
    draft_model = _load_model(
        draft_model_name,
        torch.device("cpu"),
        torch_dtype=torch_dtype,
        revision=draft_artifact.revision_for_loading,
    )
    calibration_contexts = build_continuation_contexts(
        base_model,
        tokenizer,
        calibration_prompts,
        max_new_tokens=continuation_tokens,
    )
    evaluation_contexts = build_continuation_contexts(
        base_model,
        tokenizer,
        eval_prompts,
        max_new_tokens=continuation_tokens,
    )
    prompt_input_ids = [
        sequence.to(device) for sequence in _prepare_prompt_input_ids(draft_tokenizer, eval_prompts)
    ]

    rows: list[dict[str, Any]] = []
    prompt_weighted_errors: list[float] = []
    for scale in magnitude_values:
        logger.info("Evaluating magnitude scale %.3f", scale)
        adapted_model = _load_scaled_adapted_model(
            base_model_name,
            adapter_path,
            scale,
            device,
            torch_dtype=torch_dtype,
            base_revision=base_artifact.revision_for_loading,
            adapter_revision=adapter_artifact.revision_for_loading,
        )
        shift_matrix = compute_logit_shift_matrix(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=eval_prompts,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            continuation_contexts=evaluation_contexts,
        )
        calibration_dataset = collect_logit_shift_dataset(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=calibration_prompts,
            tokenizer=tokenizer,
            batch_size=batch_size,
            continuation_contexts=calibration_contexts,
        )
        delta_W = parameter_delta_from_models(base_model, adapted_model)
        first_order_matrix = first_order_logit_shift(
            base_model=base_model,
            delta_W=delta_W,
            calibration_prompts=eval_prompts,
            tokenizer=tokenizer,
            batch_size=1,
            max_tangent_bytes=jvp_max_tangent_mb * 1024 * 1024,
            continuation_contexts=evaluation_contexts,
        )
        residual = nonlinearity_residual(shift_matrix, first_order_matrix)
        del delta_W, first_order_matrix
        base_model.to("cpu")
        draft_model.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        feature_logits, _, feature_prompt_indices, feature_token_positions = (
            collect_context_model_outputs(
                draft_model,
                draft_tokenizer,
                calibration_contexts,
                batch_size=batch_size,
            )
        )
        if (
            feature_prompt_indices != calibration_dataset.prompt_indices
            or feature_token_positions != calibration_dataset.token_positions
        ):
            raise RuntimeError("Draft correction features are not aligned with calibration labels")
        correction = LowRankCorrection(rank=correction_rank).calibrate_from_dataset(
            calibration_dataset,
            feature_logits,
        )
        approximation = correction.approximation_error()
        heldout_error, per_prompt_errors, heldout_prompt_indices = _heldout_normalized_logit_error(
            correction=correction,
            draft_model=draft_model,
            adapted_model=adapted_model,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            batch_size=batch_size,
            continuation_contexts=evaluation_contexts,
        )
        heldout_interval = cluster_bootstrap_mean(
            per_prompt_errors,
            heldout_prompt_indices,
            repetitions=bootstrap_repetitions,
            seed=seed + len(rows),
        )
        prompt_weighted_errors.append(heldout_interval.estimate)
        corrected_proxy = simulate_speculative_decoding(
            draft_model=draft_model,
            target_model=adapted_model,
            prompt_input_ids=prompt_input_ids,
            speculation_length=speculation_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=draft_tokenizer.eos_token_id,
            correction=correction,
        )
        baseline_proxy = simulate_speculative_decoding(
            draft_model=draft_model,
            target_model=adapted_model,
            prompt_input_ids=prompt_input_ids,
            speculation_length=speculation_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=draft_tokenizer.eos_token_id,
            correction=None,
        )
        analysis = spectral_analysis(center_logit_shift_rows(shift_matrix))
        rows.append(
            {
                "magnitude_scale": scale,
                "effective_rank_99": analysis.effective_rank_99,
                "stable_rank": analysis.stable_rank,
                "participation_ratio": analysis.participation_ratio,
                "spectral_tail_relative_frobenius": approximation.spectral_tail_relative_frobenius,
                "centered_shift_reconstruction_relative_frobenius": (
                    approximation.centered_shift_reconstruction_relative_frobenius
                ),
                "operator_calibration_relative_frobenius": approximation.operator_calibration_relative_frobenius,
                "end_to_end_calibration_relative_frobenius": approximation.end_to_end_calibration_relative_frobenius,
                "coefficient_regression_relative_frobenius": (
                    approximation.coefficient_regression_relative_frobenius
                ),
                "predicted_centered_operator_relative_frobenius": (
                    approximation.predicted_centered_operator_relative_frobenius
                ),
                "centered_operator_relative_frobenius": approximation.centered_operator_relative_frobenius,
                "heldout_normalized_logit_error": heldout_error,
                "heldout_prompt_weighted_normalized_logit_error": heldout_interval.estimate,
                "heldout_prompt_cluster_bootstrap": asdict(heldout_interval),
                "nonlinearity_frobenius_fraction": residual.frobenius_fraction,
                "nonlinearity_relative_row_mean": residual.relative_row_mean,
                "greedy_proxy_acceptance_baseline": baseline_proxy.acceptance.overall_acceptance_rate,
                "greedy_proxy_acceptance_corrected": corrected_proxy.acceptance.overall_acceptance_rate,
                "greedy_proxy_acceptance_recovery": (
                    corrected_proxy.acceptance.overall_acceptance_rate
                    - baseline_proxy.acceptance.overall_acceptance_rate
                ),
            }
        )
        draft_model.to("cpu")
        del adapted_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base_model.to(device)

    breakpoint_analysis: dict[str, Any] | None = None
    if len(magnitude_values) >= 6:
        segmented_fit = fit_continuous_piecewise_linear(
            magnitude_values,
            prompt_weighted_errors,
        )
        breakpoint_analysis = {
            **asdict(segmented_fit),
            "status": "exploratory",
            "interpretation": (
                "Positive BIC improvement favors a continuous segmented trend over one "
                "straight line; it does not establish a thermodynamic phase transition."
            ),
            "response": "heldout_prompt_weighted_normalized_logit_error",
        }

    payload = {
        "experiment_type": "phase_transition_sweep",
        "correction_calibration_shift": "adapted_target_minus_base_target",
        "logit_gauge": "row_mean_centered",
        "acceptance_metric": "greedy_sequence_proxy_not_vllm_rejection_sampling",
        "rows": rows,
        "breakpoint_analysis": breakpoint_analysis,
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="phase_transition_sweep",
        config={
            "base_model": base_model_name,
            "draft_model": draft_model_name,
            "adapter_path": adapter_path,
            "prompts_file": prompts_file,
            "eval_prompts_file": eval_prompts_file,
            "prompts_provenance": prompts_provenance,
            "eval_prompts_provenance": eval_prompts_provenance,
            "artifact_provenance": {
                "base_model": base_artifact.to_dict(),
                "draft_model": draft_artifact.to_dict(),
                "adapter": adapter_artifact.to_dict(),
            },
            "magnitude_values": magnitude_values,
            "correction_rank": correction_rank,
            "batch_size": batch_size,
            "speculation_length": speculation_length,
            "continuation_tokens": continuation_tokens,
            "trajectory_policy": evaluation_contexts.generation_policy,
            "calibration_contexts_sha256": calibration_contexts.sha256(),
            "evaluation_contexts_sha256": evaluation_contexts.sha256(),
            "max_new_tokens": max_new_tokens,
            "torch_dtype": torch_dtype_name,
            "jvp_max_tangent_mb": jvp_max_tangent_mb,
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved phase-transition sweep to %s", output)


if __name__ == "__main__":
    main()
