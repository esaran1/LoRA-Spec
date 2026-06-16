from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from lora_spec.acceptance_theory import (
    acceptance_lower_bound_from_logit_residual,
    empirical_vs_predicted_recovery,
)
from lora_spec.artifacts import resolve_artifact_revision, tokenizers_are_equivalent
from lora_spec.correction import LowRankCorrection, MeanShiftCorrection
from lora_spec.metrics import simulate_speculative_decoding
from lora_spec.prompts import load_frozen_prompt_texts, prompt_file_provenance
from lora_spec.statistical import cluster_bootstrap_mean
from lora_spec.theory import (
    ContinuationContextSet,
    build_continuation_contexts,
    center_logit_shift_rows,
    collect_context_model_outputs,
    collect_logit_shift_dataset,
)
from lora_spec.utils import (
    add_common_args,
    ensure_dir,
    get_config_value,
    resolve_config,
    resolve_torch_dtype,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the low-rank correction theory against acceptance recovery."
    )
    add_common_args(parser)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--base-revision", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--adapted-revision", type=str, default=None)
    parser.add_argument("--adapted-adapter-path", type=str, default=None)
    parser.add_argument("--adapter-revision", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--draft-revision", type=str, default=None)
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
    parser.add_argument("--rank-values", type=str, default="0,1,2,4,8,16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="results/theory")
    return parser.parse_args()


def _parse_rank_values(value: str) -> list[int]:
    ranks = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not ranks:
        raise ValueError("rank_values must contain at least one integer")
    return ranks


def _load_tokenizer(model_name: str, revision: str | None = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    revision: str | None = None,
) -> PreTrainedModel:
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


def _load_adapted_model(
    base_model_name: str,
    adapted_model_name: str | None,
    adapted_adapter_path: str | None,
    device: torch.device,
    torch_dtype: torch.dtype,
    base_revision: str | None = None,
    adapted_revision: str | None = None,
    adapter_revision: str | None = None,
) -> PreTrainedModel:
    if adapted_adapter_path:
        base_model = _load_model(
            base_model_name,
            device,
            torch_dtype=torch_dtype,
            revision=base_revision,
        )
        return (
            PeftModel.from_pretrained(
                base_model,
                adapted_adapter_path,
                revision=adapter_revision,
            )
            .to(device)
            .eval()
        )
    if adapted_model_name:
        return _load_model(
            adapted_model_name,
            device,
            torch_dtype=torch_dtype,
            revision=adapted_revision,
        )
    raise ValueError("Either adapted_model or adapted_adapter_path must be provided")


def _prepare_prompt_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> list[torch.Tensor]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    sequences: list[torch.Tensor] = []
    for index in range(input_ids.shape[0]):
        length = int(attention_mask[index].sum().item())
        if length >= 1:
            sequences.append(input_ids[index, :length].clone())
    if not sequences:
        raise ValueError("No valid prompt token sequences were prepared")
    return sequences


def _evaluate_divergence(
    correction: Any | None,
    draft_model: PreTrainedModel,
    adapted_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
    continuation_contexts: ContinuationContextSet,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
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
        raise RuntimeError("Draft and adapted evaluation rows are not prompt-aligned")
    adjusted_logits = draft_logits if correction is None else correction.apply(draft_logits)
    adjusted_log_probs = F.log_softmax(adjusted_logits, dim=-1)
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    adapted_probs = adapted_log_probs.exp()
    adjusted_probs = adjusted_log_probs.exp()
    kl = torch.sum(adapted_probs * (adapted_log_probs - adjusted_log_probs), dim=-1)
    total_variation = 0.5 * torch.sum(torch.abs(adapted_probs - adjusted_probs), dim=-1)
    rejection_acceptance = torch.sum(torch.minimum(adapted_probs, adjusted_probs), dim=-1)
    residual = adapted_logits - adjusted_logits
    residual_span = residual.amax(dim=-1) - residual.amin(dim=-1)
    logit_acceptance_lower_bound = acceptance_lower_bound_from_logit_residual(
        adapted_logits,
        adjusted_logits,
    )
    baseline_residual = center_logit_shift_rows(draft_logits - adapted_logits)
    corrected_residual = center_logit_shift_rows(adjusted_logits - adapted_logits)
    total_positions = int(draft_logits.shape[0])
    clustered_metrics = {
        "kl_divergence": cluster_bootstrap_mean(
            kl.cpu().numpy(), prompt_indices, bootstrap_repetitions, seed=bootstrap_seed
        ),
        "total_variation": cluster_bootstrap_mean(
            total_variation.cpu().numpy(),
            prompt_indices,
            bootstrap_repetitions,
            seed=bootstrap_seed + 1,
        ),
        "rejection_sampling_acceptance": cluster_bootstrap_mean(
            rejection_acceptance.cpu().numpy(),
            prompt_indices,
            bootstrap_repetitions,
            seed=bootstrap_seed + 2,
        ),
        "residual_logit_span": cluster_bootstrap_mean(
            residual_span.cpu().numpy(),
            prompt_indices,
            bootstrap_repetitions,
            seed=bootstrap_seed + 3,
        ),
        "logit_acceptance_lower_bound": cluster_bootstrap_mean(
            logit_acceptance_lower_bound.cpu().numpy(),
            prompt_indices,
            bootstrap_repetitions,
            seed=bootstrap_seed + 4,
        ),
    }

    return {
        "kl_divergence": float(kl.mean().item()),
        "mean_total_variation": float(total_variation.mean().item()),
        "expected_rejection_sampling_acceptance": float(rejection_acceptance.mean().item()),
        "logit_acceptance_lower_bound": float(logit_acceptance_lower_bound.mean().item()),
        "logit_tv_upper_bound": float(1.0 - logit_acceptance_lower_bound.mean().item()),
        "mean_residual_logit_span": float(residual_span.mean().item()),
        "num_positions": float(total_positions),
        "num_prompt_clusters": len(set(prompt_indices)),
        "prompt_cluster_bootstrap": {
            name: asdict(interval) for name, interval in clustered_metrics.items()
        },
        "heldout_normalized_logit_error": float(
            torch.linalg.matrix_norm(corrected_residual, ord="fro").item()
            / max(torch.linalg.matrix_norm(baseline_residual, ord="fro").item(), 1e-12),
        ),
    }


def _mean_shift_error(dataset: Any) -> float:
    shift = center_logit_shift_rows(dataset.shift_matrix.float())
    centered = shift - shift.mean(dim=0)
    return float(
        torch.linalg.matrix_norm(centered, ord="fro").item()
        / max(torch.linalg.matrix_norm(shift, ord="fro").item(), 1e-12)
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "validate_correction_theory")
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)

    base_model_value = get_config_value(config_data, args, "base_model")
    adapted_model_value = get_config_value(config_data, args, "adapted_model")
    adapted_adapter_path_value = get_config_value(config_data, args, "adapted_adapter_path")
    draft_model_value = get_config_value(config_data, args, "draft_model")
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not base_model_value or not draft_model_value or not prompts_file_value:
        raise ValueError("base_model, draft_model, and prompts_file must be provided")
    base_model_name = str(base_model_value)
    draft_model_name = str(draft_model_value)
    prompts_file = str(prompts_file_value)
    base_revision = get_config_value(config_data, args, "base_revision")
    adapted_revision = get_config_value(config_data, args, "adapted_revision")
    adapter_revision = get_config_value(config_data, args, "adapter_revision")
    draft_revision = get_config_value(config_data, args, "draft_revision")
    eval_prompts_file_value = get_config_value(config_data, args, "eval_prompts_file")
    if not eval_prompts_file_value:
        raise ValueError("eval_prompts_file must be provided separately from prompts_file")
    eval_prompts_file = str(eval_prompts_file_value)
    rank_values = _parse_rank_values(str(get_config_value(config_data, args, "rank_values")))
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    speculation_length = int(get_config_value(config_data, args, "speculation_length"))
    max_new_tokens = int(get_config_value(config_data, args, "max_new_tokens"))
    continuation_tokens = int(get_config_value(config_data, args, "continuation_tokens"))
    bootstrap_repetitions = int(get_config_value(config_data, args, "bootstrap_repetitions"))
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    plots_dir = ensure_dir(Path(output_dir) / "plots")

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
    adapted_artifact = (
        resolve_artifact_revision(str(adapted_model_value), revision=adapted_revision)
        if adapted_model_value
        else None
    )
    adapter_artifact = (
        resolve_artifact_revision(str(adapted_adapter_path_value), revision=adapter_revision)
        if adapted_adapter_path_value
        else None
    )

    tokenizer = _load_tokenizer(base_model_name, revision=base_artifact.revision_for_loading)
    draft_tokenizer = _load_tokenizer(
        draft_model_name, revision=draft_artifact.revision_for_loading
    )
    if not tokenizers_are_equivalent(
        tokenizer, draft_tokenizer, calibration_prompts + eval_prompts
    ):
        raise ValueError("draft_model tokenizer must be compatible with base_model tokenizer")

    logger.info("Loading models on %s", device)
    base_model = _load_model(
        base_model_name,
        device,
        torch_dtype=torch_dtype,
        revision=base_artifact.revision_for_loading,
    )
    adapted_model = _load_adapted_model(
        base_model_name=base_model_name,
        adapted_model_name=str(adapted_model_value) if adapted_model_value else None,
        adapted_adapter_path=str(adapted_adapter_path_value)
        if adapted_adapter_path_value
        else None,
        device=device,
        torch_dtype=torch_dtype,
        base_revision=base_artifact.revision_for_loading,
        adapted_revision=adapted_artifact.revision_for_loading if adapted_artifact else None,
        adapter_revision=adapter_artifact.revision_for_loading if adapter_artifact else None,
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
    calibration_dataset = collect_logit_shift_dataset(
        base_model=base_model,
        adapted_model=adapted_model,
        calibration_prompts=calibration_prompts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        collect_hidden_states=False,
        continuation_contexts=calibration_contexts,
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    draft_model = _load_model(
        draft_model_name,
        device,
        torch_dtype=torch_dtype,
        revision=draft_artifact.revision_for_loading,
    )
    calibration_feature_logits, _, feature_prompt_indices, feature_token_positions = (
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

    prompt_input_ids = [
        sequence.to(device) for sequence in _prepare_prompt_input_ids(draft_tokenizer, eval_prompts)
    ]
    baseline_proxy = simulate_speculative_decoding(
        draft_model=draft_model,
        target_model=adapted_model,
        prompt_input_ids=prompt_input_ids,
        speculation_length=speculation_length,
        max_new_tokens=max_new_tokens,
        eos_token_id=draft_tokenizer.eos_token_id,
        correction=None,
    )
    baseline_divergence = _evaluate_divergence(
        None,
        draft_model,
        adapted_model,
        draft_tokenizer,
        eval_prompts,
        batch_size,
        evaluation_contexts,
        bootstrap_repetitions,
        seed,
    )

    rows: list[dict[str, Any]] = []
    mean_residual_spans: list[float] = []
    empirical_acceptances: list[float] = []
    for rank in rank_values:
        if rank == 0:
            correction = MeanShiftCorrection().calibrate_from_dataset(
                calibration_dataset,
            )
            spectral_tail_error = _mean_shift_error(calibration_dataset)
            reconstruction_error = _mean_shift_error(calibration_dataset)
            retained_energy_fraction = 0.0
        else:
            correction = LowRankCorrection(rank=rank).calibrate_from_dataset(
                calibration_dataset,
                calibration_feature_logits,
            )
            approximation = correction.approximation_error()
            spectral_tail_error = approximation.spectral_tail_relative_frobenius
            reconstruction_error = approximation.centered_shift_reconstruction_relative_frobenius
            operator_calibration_error = approximation.operator_calibration_relative_frobenius
            end_to_end_calibration_error = approximation.end_to_end_calibration_relative_frobenius
            coefficient_regression_error = approximation.coefficient_regression_relative_frobenius
            centered_operator_error = approximation.centered_operator_relative_frobenius
            retained_energy_fraction = approximation.retained_energy_fraction
        if rank == 0:
            operator_calibration_error = float("nan")
            end_to_end_calibration_error = float("nan")
            coefficient_regression_error = float("nan")
            centered_operator_error = float("nan")
        divergence = _evaluate_divergence(
            correction,
            draft_model,
            adapted_model,
            draft_tokenizer,
            eval_prompts,
            batch_size,
            evaluation_contexts,
            bootstrap_repetitions,
            seed + rank * 10,
        )
        proxy = simulate_speculative_decoding(
            draft_model=draft_model,
            target_model=adapted_model,
            prompt_input_ids=prompt_input_ids,
            speculation_length=speculation_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=draft_tokenizer.eos_token_id,
            correction=correction,
        )
        bootstrap_summary = divergence["prompt_cluster_bootstrap"]
        mean_residual_spans.append(float(bootstrap_summary["residual_logit_span"]["estimate"]))
        empirical_acceptances.append(
            float(bootstrap_summary["rejection_sampling_acceptance"]["estimate"])
        )
        rows.append(
            {
                "rank": rank,
                "spectral_tail_relative_frobenius": spectral_tail_error,
                "centered_shift_reconstruction_relative_frobenius": reconstruction_error,
                "operator_calibration_relative_frobenius": operator_calibration_error,
                "end_to_end_calibration_relative_frobenius": end_to_end_calibration_error,
                "coefficient_regression_relative_frobenius": coefficient_regression_error,
                "centered_operator_relative_frobenius": centered_operator_error,
                "base_feature_coefficient_relative_frobenius": (
                    approximation.base_feature_coefficient_relative_frobenius
                    if rank > 0
                    else float("nan")
                ),
                "base_feature_operator_relative_frobenius": (
                    approximation.base_feature_operator_relative_frobenius
                    if rank > 0
                    else float("nan")
                ),
                "heldout_normalized_logit_error": float(
                    divergence["heldout_normalized_logit_error"]
                ),
                "retained_energy_fraction": retained_energy_fraction,
                "kl_divergence": divergence["kl_divergence"],
                "mean_total_variation": divergence["mean_total_variation"],
                "expected_rejection_sampling_acceptance": divergence[
                    "expected_rejection_sampling_acceptance"
                ],
                "logit_acceptance_lower_bound": divergence["logit_acceptance_lower_bound"],
                "logit_tv_upper_bound": divergence["logit_tv_upper_bound"],
                "mean_residual_logit_span": divergence["mean_residual_logit_span"],
                "num_prompt_clusters": divergence["num_prompt_clusters"],
                "prompt_cluster_bootstrap": divergence["prompt_cluster_bootstrap"],
                "greedy_proxy_acceptance_rate": proxy.acceptance.overall_acceptance_rate,
                "acceptance_by_depth": proxy.acceptance.acceptance_by_depth,
                "depth_attempts": proxy.acceptance.depth_attempts,
                "depth_accepted": proxy.acceptance.depth_accepted,
                "tokens_per_target_call": proxy.tokens_per_target_call,
            }
        )

    comparison = empirical_vs_predicted_recovery(
        approximation_error=mean_residual_spans,
        base_acceptance=float(
            baseline_divergence["prompt_cluster_bootstrap"]["rejection_sampling_acceptance"][
                "estimate"
            ]
        ),
        empirical_acceptance=empirical_acceptances,
        error_metric="logit_span",
    )
    plot_path = comparison.save_plot(plots_dir / "correction_theory_validation.png")

    payload = {
        "experiment_type": "validate_correction_theory",
        "baseline_expected_rejection_sampling_acceptance": baseline_divergence[
            "expected_rejection_sampling_acceptance"
        ],
        "baseline_prompt_cluster_bootstrap": baseline_divergence["prompt_cluster_bootstrap"],
        "baseline_greedy_proxy_acceptance": baseline_proxy.acceptance.overall_acceptance_rate,
        "correction_calibration_shift": "adapted_target_minus_base_target",
        "logit_gauge": "row_mean_centered",
        "rows": rows,
        "predicted_vs_empirical": {
            "approximation_error": comparison.approximation_error,
            "acceptance_lower_bound": comparison.acceptance_lower_bound,
            "empirical_acceptance": comparison.empirical_acceptance,
            "guaranteed_recovery_delta": comparison.guaranteed_recovery_delta,
            "empirical_recovery_delta": comparison.empirical_recovery_delta,
            "bound_coverage": comparison.bound_coverage,
            "mean_bound_slack": comparison.mean_bound_slack,
            "maximum_bound_violation": comparison.maximum_bound_violation,
            "error_metric": comparison.error_metric,
            "plot_path": str(plot_path),
        },
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="validate_correction_theory",
        config={
            "base_model": base_model_name,
            "adapted_model": adapted_model_value,
            "adapted_adapter_path": adapted_adapter_path_value,
            "draft_model": draft_model_name,
            "prompts_file": prompts_file,
            "eval_prompts_file": eval_prompts_file,
            "prompts_provenance": prompts_provenance,
            "eval_prompts_provenance": eval_prompts_provenance,
            "artifact_provenance": {
                "base_model": base_artifact.to_dict(),
                "adapted_model": adapted_artifact.to_dict() if adapted_artifact else None,
                "adapter": adapter_artifact.to_dict() if adapter_artifact else None,
                "draft_model": draft_artifact.to_dict(),
            },
            "rank_values": rank_values,
            "batch_size": batch_size,
            "speculation_length": speculation_length,
            "max_new_tokens": max_new_tokens,
            "continuation_tokens": continuation_tokens,
            "bootstrap_repetitions": bootstrap_repetitions,
            "trajectory_policy": evaluation_contexts.generation_policy,
            "calibration_contexts_sha256": calibration_contexts.sha256(),
            "evaluation_contexts_sha256": evaluation_contexts.sha256(),
            "torch_dtype": torch_dtype_name,
            "seed": seed,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved correction-theory validation to %s", output)


if __name__ == "__main__":
    main()
