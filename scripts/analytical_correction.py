from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lora_spec.correction import ContextDependentCorrection, LowRankCorrection, MeanShiftCorrection
from lora_spec.metrics import simulate_speculative_decoding
from lora_spec.prompts import load_frozen_prompt_texts, prompt_file_provenance
from lora_spec.theory import (
    center_logit_shift_rows,
    first_order_logit_shift,
    nonlinearity_residual,
    parameter_delta_from_models,
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
    parser = argparse.ArgumentParser(description="Evaluate theory-grounded analytical corrections.")
    add_common_args(parser)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--adapted-adapter-path", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
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
    parser.add_argument("--low-rank-k", type=int, default=8)
    parser.add_argument("--context-rank", type=int, default=8)
    parser.add_argument("--context-hidden-dim", type=int, default=64)
    parser.add_argument("--context-epochs", type=int, default=150)
    parser.add_argument("--context-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--skip-speculative-proxy", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/correction")
    return parser.parse_args()


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def _load_adapted_model(
    base_model_name: str,
    adapted_model_name: str | None,
    adapted_adapter_path: str | None,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> PreTrainedModel:
    if adapted_adapter_path:
        base_model = _load_model(base_model_name, device, torch_dtype=torch_dtype)
        return PeftModel.from_pretrained(base_model, adapted_adapter_path).to(device).eval()
    if adapted_model_name:
        return _load_model(adapted_model_name, device, torch_dtype=torch_dtype)
    raise ValueError("Either adapted_model or adapted_adapter_path must be provided")


def _load_draft_model(
    draft_model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = _load_tokenizer(draft_model_name)
    model = _load_model(draft_model_name, device, torch_dtype=torch_dtype)
    return model, tokenizer


def _tokenizer_is_compatible(
    reference_tokenizer: PreTrainedTokenizerBase,
    candidate_tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> bool:
    if reference_tokenizer.vocab_size != candidate_tokenizer.vocab_size:
        return False
    for prompt in prompts[: min(4, len(prompts))]:
        if reference_tokenizer(prompt, add_special_tokens=True)["input_ids"] != candidate_tokenizer(
            prompt,
            add_special_tokens=True,
        )["input_ids"]:
            return False
    return True


def _batch_prompts(prompts: list[str], batch_size: int) -> list[list[str]]:
    return [prompts[index : index + batch_size] for index in range(0, len(prompts), batch_size)]


def _evaluate_against_adapted(
    correction: Any | None,
    draft_model: PreTrainedModel,
    adapted_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
) -> dict[str, float]:
    device = next(draft_model.parameters()).device
    total_positions = 0
    kl_sum = 0.0
    js_sum = 0.0
    baseline_residual_sum = 0.0
    corrected_residual_sum = 0.0

    for batch_prompts in _batch_prompts(prompts, batch_size):
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        requires_hidden = bool(getattr(correction, "requires_hidden_state", False)) if correction is not None else False
        with torch.no_grad():
            draft_outputs = draft_model(**encoded, output_hidden_states=requires_hidden)
            adapted_logits = adapted_model(**encoded).logits[:, :-1, :].float()
        draft_logits = draft_outputs.logits[:, :-1, :].float()
        hidden_state = None
        if requires_hidden:
            hidden_states = getattr(draft_outputs, "hidden_states", None)
            if not hidden_states:
                raise ValueError("Draft model did not return hidden states required by the correction")
            hidden_state = hidden_states[-1][:, :-1, :].float()

        adjusted_logits = draft_logits if correction is None else correction.apply(draft_logits, hidden_state=hidden_state)
        adjusted_log_probs = F.log_softmax(adjusted_logits, dim=-1)
        adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
        adjusted_probs = adjusted_log_probs.exp()
        adapted_probs = adapted_log_probs.exp()

        mask = encoded["attention_mask"][:, 1:].bool()
        positions = int(mask.sum().item())
        if positions == 0:
            continue

        kl = torch.sum(adapted_probs * (adapted_log_probs - adjusted_log_probs), dim=-1)
        midpoint = 0.5 * (adapted_probs + adjusted_probs)
        midpoint_log = torch.log(midpoint.clamp_min(1e-12))
        js_left = torch.sum(adapted_probs * (adapted_log_probs - midpoint_log), dim=-1)
        js_right = torch.sum(adjusted_probs * (adjusted_log_probs - midpoint_log), dim=-1)
        js = 0.5 * (js_left + js_right)
        expanded_mask = mask.unsqueeze(-1).expand_as(adapted_logits)
        baseline_delta = center_logit_shift_rows(
            (draft_logits - adapted_logits).reshape(-1, adapted_logits.shape[-1]),
        ).reshape_as(adapted_logits)
        corrected_delta = center_logit_shift_rows(
            (adjusted_logits - adapted_logits).reshape(-1, adapted_logits.shape[-1]),
        ).reshape_as(adapted_logits)
        baseline_residual = baseline_delta.masked_select(expanded_mask)
        corrected_residual = corrected_delta.masked_select(expanded_mask)

        total_positions += positions
        kl_sum += float(kl.masked_select(mask).sum().item())
        js_sum += float(js.masked_select(mask).sum().item())
        baseline_residual_sum += float(torch.sum(baseline_residual.square()).item())
        corrected_residual_sum += float(torch.sum(corrected_residual.square()).item())

    if total_positions == 0:
        raise ValueError("Prompt set did not yield any valid next-token positions")
    return {
        "kl_divergence": kl_sum / total_positions,
        "js_divergence": js_sum / total_positions,
        "num_positions": float(total_positions),
        "heldout_normalized_logit_error": float(
            (corrected_residual_sum ** 0.5) / max(baseline_residual_sum ** 0.5, 1e-12),
        ),
    }


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
        if length < 1:
            continue
        sequences.append(input_ids[index, :length].clone())
    if not sequences:
        raise ValueError("No valid prompt token sequences were prepared")
    return sequences


def _proxy_metrics_to_dict(metrics: Any) -> dict[str, Any]:
    return {
        "acceptance_rate_overall": metrics.acceptance.overall_acceptance_rate,
        "acceptance_rate_per_position": metrics.acceptance.per_position_acceptance_rate,
        "per_position_attempts": metrics.acceptance.per_position_attempts,
        "per_position_accepted": metrics.acceptance.per_position_accepted,
        "acceptance_by_depth": metrics.acceptance.acceptance_by_depth,
        "accepted_drafted_tokens": metrics.acceptance.accepted_drafted_tokens,
        "total_drafted_tokens": metrics.acceptance.total_drafted_tokens,
        "bonus_tokens": metrics.acceptance.bonus_tokens,
        "speculative_steps": metrics.acceptance.speculative_steps,
        "emitted_tokens": metrics.emitted_tokens,
        "target_model_calls": metrics.target_model_calls,
        "draft_model_calls": metrics.draft_model_calls,
        "tokens_per_target_call": metrics.tokens_per_target_call,
        "draft_tokens_per_call": metrics.draft_tokens_per_call,
        "target_call_reduction_vs_autoregressive": metrics.target_call_reduction_vs_autoregressive,
    }


def _correction_bundle(
    base_model: PreTrainedModel,
    adapted_model: PreTrainedModel,
    feature_model: PreTrainedModel,
    prompts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    low_rank_k: int,
    context_rank: int,
    context_hidden_dim: int,
    context_epochs: int,
    context_lr: float,
    seed: int,
) -> dict[str, Any]:
    mean_shift = MeanShiftCorrection().calibrate(base_model, adapted_model, prompts, tokenizer=tokenizer)
    low_rank = LowRankCorrection(rank=low_rank_k).calibrate(base_model, adapted_model, prompts, tokenizer=tokenizer)
    context = ContextDependentCorrection(
        rank=context_rank,
        hidden_dim=context_hidden_dim,
        epochs=context_epochs,
        lr=context_lr,
        seed=seed,
    ).calibrate(
        base_model,
        adapted_model,
        prompts,
        tokenizer=tokenizer,
        feature_model=feature_model,
    )
    return {
        "baseline": None,
        "mean_shift": mean_shift,
        "low_rank": low_rank,
        "context_dependent": context,
    }


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "analytical_correction")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)

    base_model_value = get_config_value(config_data, args, "base_model")
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not base_model_value or not prompts_file_value:
        raise ValueError("base_model and prompts_file must be provided")
    adapted_model_value = get_config_value(config_data, args, "adapted_model")
    adapted_adapter_path_value = get_config_value(config_data, args, "adapted_adapter_path")
    if not adapted_model_value and not adapted_adapter_path_value:
        raise ValueError("Either adapted_model or adapted_adapter_path must be provided")

    base_model_name = str(base_model_value)
    adapted_model_name = str(adapted_model_value) if adapted_model_value else None
    adapted_adapter_path = str(adapted_adapter_path_value) if adapted_adapter_path_value else None
    draft_model_value = get_config_value(config_data, args, "draft_model")
    prompts_file = str(prompts_file_value)
    eval_prompts_file_value = get_config_value(config_data, args, "eval_prompts_file")
    if not eval_prompts_file_value:
        raise ValueError("eval_prompts_file must be provided separately from prompts_file")
    eval_prompts_file = str(eval_prompts_file_value)
    low_rank_k = int(get_config_value(config_data, args, "low_rank_k"))
    context_rank = int(get_config_value(config_data, args, "context_rank"))
    context_hidden_dim = int(get_config_value(config_data, args, "context_hidden_dim"))
    context_epochs = int(get_config_value(config_data, args, "context_epochs"))
    context_lr = float(get_config_value(config_data, args, "context_lr"))
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    speculation_length = int(get_config_value(config_data, args, "speculation_length"))
    max_new_tokens = int(get_config_value(config_data, args, "max_new_tokens"))
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    skip_speculative_proxy = bool(get_config_value(config_data, args, "skip_speculative_proxy", args.skip_speculative_proxy))
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

    logger.info("Loading base and adapted models on %s", device)
    tokenizer = _load_tokenizer(base_model_name)
    adapted_model = _load_adapted_model(
        base_model_name=base_model_name,
        adapted_model_name=adapted_model_name,
        adapted_adapter_path=adapted_adapter_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    base_model = _load_model(base_model_name, device=device, torch_dtype=torch_dtype)
    draft_model = None
    draft_tokenizer = None
    if not skip_speculative_proxy:
        if not draft_model_value:
            raise ValueError("draft_model must be provided unless --skip-speculative-proxy is set")
        draft_model, draft_tokenizer = _load_draft_model(
            str(draft_model_value),
            device=device,
            torch_dtype=torch_dtype,
        )
        if not _tokenizer_is_compatible(tokenizer, draft_tokenizer, calibration_prompts + eval_prompts):
            raise ValueError("Draft tokenizer must be tokenization-compatible with the base/adapted tokenizer")
        correction_feature_model = draft_model
    else:
        correction_feature_model = base_model

    logger.info("Calibrating corrections on %d prompts", len(calibration_prompts))
    corrections = _correction_bundle(
        base_model=base_model,
        adapted_model=adapted_model,
        feature_model=correction_feature_model,
        prompts=calibration_prompts,
        tokenizer=tokenizer,
        low_rank_k=low_rank_k,
        context_rank=context_rank,
        context_hidden_dim=context_hidden_dim,
        context_epochs=context_epochs,
        context_lr=context_lr,
        seed=args.seed,
    )

    logger.info("Evaluating corrected draft logits against adapted target on %d prompts", len(eval_prompts))
    divergence_results = {
        name: _evaluate_against_adapted(
            correction=correction,
            draft_model=draft_model if draft_model is not None else base_model,
            adapted_model=adapted_model,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            batch_size=batch_size,
        )
        for name, correction in corrections.items()
    }

    logger.info("Computing first-order residual diagnostics")
    true_shift_matrix = None
    first_order_matrix = None
    residual_report = None
    try:
        from lora_spec.theory import compute_logit_shift_matrix

        delta_W = parameter_delta_from_models(base_model, adapted_model)
        true_shift_matrix = compute_logit_shift_matrix(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=eval_prompts,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        first_order_matrix = first_order_logit_shift(
            base_model=base_model,
            delta_W=delta_W,
            calibration_prompts=eval_prompts,
            tokenizer=tokenizer,
            batch_size=1,
        )
        residual_report = nonlinearity_residual(true_shift_matrix, first_order_matrix)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        logger.warning("First-order residual diagnostic failed: %s", exc)

    proxy_payload: dict[str, Any] = {}
    if draft_model is not None and draft_tokenizer is not None:
        logger.info("Simulating speculative decoding recovery proxy on %d prompts", len(eval_prompts))
        prompt_input_ids = [sequence.to(device) for sequence in _prepare_prompt_input_ids(draft_tokenizer, eval_prompts)]
        for name, correction in corrections.items():
            proxy_metrics = simulate_speculative_decoding(
                draft_model=draft_model,
                target_model=adapted_model,
                prompt_input_ids=prompt_input_ids,
                speculation_length=speculation_length,
                max_new_tokens=max_new_tokens,
                eos_token_id=draft_tokenizer.eos_token_id,
                correction=correction,
            )
            proxy_payload[name] = _proxy_metrics_to_dict(proxy_metrics)

    low_rank_report = corrections["low_rank"].approximation_error()
    payload = {
        **divergence_results,
        "correction_calibration_shift": "adapted_target_minus_base_target",
        "logit_gauge": "row_mean_centered",
        "low_rank_approximation": {
            "spectral_tail_relative_frobenius": low_rank_report.spectral_tail_relative_frobenius,
            "centered_shift_reconstruction_relative_frobenius": (
                low_rank_report.centered_shift_reconstruction_relative_frobenius
            ),
            "coefficient_regression_relative_frobenius": (
                low_rank_report.coefficient_regression_relative_frobenius
            ),
            "predicted_centered_operator_relative_frobenius": (
                low_rank_report.predicted_centered_operator_relative_frobenius
            ),
            "centered_operator_relative_frobenius": (
                low_rank_report.centered_operator_relative_frobenius
            ),
            "operator_calibration_relative_frobenius": low_rank_report.operator_calibration_relative_frobenius,
            "retained_energy_fraction": low_rank_report.retained_energy_fraction,
            "selected_rank": low_rank_report.selected_rank,
            "apply_overhead_ms": corrections["low_rank"].measure_overhead_ms(device=device),
        },
        "mean_shift_overhead_ms": corrections["mean_shift"].measure_overhead_ms(device=device),
        "context_dependent_overhead_ms": (
            corrections["context_dependent"].measure_overhead_ms(
                device=device,
                hidden_state=torch.zeros(
                    1,
                    int(corrections["context_dependent"].hidden_size),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            if getattr(corrections["context_dependent"], "hidden_size", None) is not None
            else None
        ),
        "speculative_proxy": proxy_payload,
        "first_order_residual": (
            None
            if residual_report is None
            else {
                "frobenius_fraction": residual_report.frobenius_fraction,
                "relative_row_mean": residual_report.relative_row_mean,
                "cosine_similarity_mean": residual_report.cosine_similarity_mean,
            }
        ),
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="analytical_correction",
        config={
            "base_model": base_model_name,
            "adapted_model": adapted_model_name,
            "adapted_adapter_path": adapted_adapter_path,
            "draft_model": str(draft_model_value) if draft_model_value else None,
            "prompts_file": prompts_file,
            "eval_prompts_file": eval_prompts_file,
            "prompts_provenance": prompts_provenance,
            "eval_prompts_provenance": eval_prompts_provenance,
            "low_rank_k": low_rank_k,
            "context_rank": context_rank,
            "context_hidden_dim": context_hidden_dim,
            "context_epochs": context_epochs,
            "context_lr": context_lr,
            "batch_size": batch_size,
            "speculation_length": speculation_length,
            "max_new_tokens": max_new_tokens,
            "torch_dtype": torch_dtype_name,
            "skip_speculative_proxy": skip_speculative_proxy,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved analytical correction evaluation to %s", output)


if __name__ == "__main__":
    main()
