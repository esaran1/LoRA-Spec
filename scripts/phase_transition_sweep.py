from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_spec.correction import LowRankCorrection
from lora_spec.metrics import simulate_speculative_decoding
from lora_spec.theory import (
    center_logit_shift_rows,
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
    parser = argparse.ArgumentParser(description="Sweep adapter magnitude to detect analytical-correction phase transitions.")
    add_common_args(parser)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--eval-prompts-file", type=str, default=None)
    parser.add_argument("--magnitude-values", type=str, default="0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--correction-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/theory")
    return parser.parse_args()


def _load_prompts(path: str) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _parse_values(raw: str) -> list[float]:
    values = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError("magnitude_values must contain at least one float")
    return values


def _load_model(model_name: str, device: torch.device, torch_dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _apply_lora_scale(model: Any, scale: float) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.mul_(scale)


def _load_scaled_adapted_model(
    base_model_name: str,
    adapter_path: str,
    scale: float,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    base_model = _load_model(base_model_name, device, torch_dtype=torch_dtype)
    adapted = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
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


def _tokenizer_is_compatible(reference_tokenizer: Any, candidate_tokenizer: Any, prompts: list[str]) -> bool:
    if reference_tokenizer.vocab_size != candidate_tokenizer.vocab_size:
        return False
    for prompt in prompts[: min(4, len(prompts))]:
        if reference_tokenizer(prompt, add_special_tokens=True)["input_ids"] != candidate_tokenizer(
            prompt,
            add_special_tokens=True,
        )["input_ids"]:
            return False
    return True


def _heldout_normalized_logit_error(
    correction: Any,
    draft_model: Any,
    adapted_model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
) -> float:
    device = next(draft_model.parameters()).device
    baseline_residual_sum = 0.0
    corrected_residual_sum = 0.0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            draft_logits = draft_model(**encoded).logits[:, :-1, :].float()
            adapted_logits = adapted_model(**encoded).logits[:, :-1, :].float()
        adjusted_logits = correction.apply(draft_logits)
        mask = encoded["attention_mask"][:, 1:].bool().unsqueeze(-1).expand_as(adapted_logits)
        baseline_delta = center_logit_shift_rows(
            (draft_logits - adapted_logits).reshape(-1, adapted_logits.shape[-1]),
        ).reshape_as(adapted_logits)
        corrected_delta = center_logit_shift_rows(
            (adjusted_logits - adapted_logits).reshape(-1, adapted_logits.shape[-1]),
        ).reshape_as(adapted_logits)
        baseline_residual = baseline_delta.masked_select(mask)
        corrected_residual = corrected_delta.masked_select(mask)
        baseline_residual_sum += float(torch.sum(baseline_residual.square()).item())
        corrected_residual_sum += float(torch.sum(corrected_residual.square()).item())
    return float((corrected_residual_sum ** 0.5) / max(baseline_residual_sum ** 0.5, 1e-12))


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "phase_transition_sweep")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)

    base_model_name = str(get_config_value(config_data, args, "base_model"))
    draft_model_name = str(get_config_value(config_data, args, "draft_model"))
    adapter_path = str(get_config_value(config_data, args, "adapter_path"))
    prompts_file = str(get_config_value(config_data, args, "prompts_file"))
    eval_prompts_file = str(get_config_value(config_data, args, "eval_prompts_file") or prompts_file)
    magnitude_values = _parse_values(str(get_config_value(config_data, args, "magnitude_values")))
    correction_rank = int(get_config_value(config_data, args, "correction_rank"))
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    speculation_length = int(get_config_value(config_data, args, "speculation_length"))
    max_new_tokens = int(get_config_value(config_data, args, "max_new_tokens"))
    torch_dtype_name = str(get_config_value(config_data, args, "torch_dtype"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    if not all([base_model_name, draft_model_name, adapter_path, prompts_file]):
        raise ValueError("base_model, draft_model, adapter_path, and prompts_file must be provided")

    calibration_prompts = _load_prompts(prompts_file)
    eval_prompts = _load_prompts(eval_prompts_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(torch_dtype_name, device=device)
    tokenizer = _load_tokenizer(base_model_name)
    draft_tokenizer = _load_tokenizer(draft_model_name)
    if not _tokenizer_is_compatible(tokenizer, draft_tokenizer, calibration_prompts + eval_prompts):
        raise ValueError("draft_model tokenizer must be compatible with base_model tokenizer")
    base_model = _load_model(base_model_name, device, torch_dtype=torch_dtype)
    draft_model = _load_model(draft_model_name, device, torch_dtype=torch_dtype)
    prompt_input_ids = [sequence.to(device) for sequence in _prepare_prompt_input_ids(draft_tokenizer, eval_prompts)]

    rows: list[dict[str, Any]] = []
    for scale in magnitude_values:
        logger.info("Evaluating magnitude scale %.3f", scale)
        adapted_model = _load_scaled_adapted_model(
            base_model_name,
            adapter_path,
            scale,
            device,
            torch_dtype=torch_dtype,
        )
        shift_matrix = compute_logit_shift_matrix(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=eval_prompts,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
        )
        delta_W = parameter_delta_from_models(base_model, adapted_model)
        first_order_matrix = first_order_logit_shift(
            base_model=base_model,
            delta_W=delta_W,
            calibration_prompts=eval_prompts,
            tokenizer=tokenizer,
            batch_size=1,
        )
        residual = nonlinearity_residual(shift_matrix, first_order_matrix)
        correction = LowRankCorrection(rank=correction_rank).calibrate(
            base_model,
            adapted_model,
            calibration_prompts,
            tokenizer=draft_tokenizer,
        )
        approximation = correction.approximation_error()
        heldout_error = _heldout_normalized_logit_error(
            correction=correction,
            draft_model=draft_model,
            adapted_model=adapted_model,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            batch_size=batch_size,
        )
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
                "coefficient_regression_relative_frobenius": (
                    approximation.coefficient_regression_relative_frobenius
                ),
                "predicted_centered_operator_relative_frobenius": (
                    approximation.predicted_centered_operator_relative_frobenius
                ),
                "centered_operator_relative_frobenius": approximation.centered_operator_relative_frobenius,
                "heldout_normalized_logit_error": heldout_error,
                "nonlinearity_frobenius_fraction": residual.frobenius_fraction,
                "nonlinearity_relative_row_mean": residual.relative_row_mean,
                "greedy_proxy_acceptance_baseline": baseline_proxy.acceptance.overall_acceptance_rate,
                "greedy_proxy_acceptance_corrected": corrected_proxy.acceptance.overall_acceptance_rate,
                "greedy_proxy_acceptance_recovery": (
                    corrected_proxy.acceptance.overall_acceptance_rate - baseline_proxy.acceptance.overall_acceptance_rate
                ),
            }
        )
        del adapted_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "experiment_type": "phase_transition_sweep",
        "correction_calibration_shift": "adapted_target_minus_base_target",
        "logit_gauge": "row_mean_centered",
        "acceptance_metric": "greedy_sequence_proxy_not_vllm_rejection_sampling",
        "rows": rows,
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
            "magnitude_values": magnitude_values,
            "correction_rank": correction_rank,
            "batch_size": batch_size,
            "speculation_length": speculation_length,
            "max_new_tokens": max_new_tokens,
            "torch_dtype": torch_dtype_name,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved phase-transition sweep to %s", output)


if __name__ == "__main__":
    main()
