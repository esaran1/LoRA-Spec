from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lora_spec.correction import DistributionOffsetCorrection, JacobianCorrection, LowRankCorrection
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate analytical correction baselines.")
    add_common_args(parser)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--adapted-adapter-path", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--eval-prompts-file", type=str, default=None)
    parser.add_argument("--low-rank-k", type=int, default=8)
    parser.add_argument("--jacobian-probe-count", type=int, default=8)
    parser.add_argument("--jacobian-max-params", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="results/correction")
    return parser.parse_args()


def _load_prompts(path: str) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _load_base_model_and_tokenizer(
    base_model_name: str,
    device: torch.device,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device).eval()
    return model, tokenizer


def _load_adapted_model(
    base_model_name: str,
    adapted_model_name: str | None,
    adapted_adapter_path: str | None,
    device: torch.device,
) -> PreTrainedModel:
    if adapted_adapter_path:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device).eval()
        return PeftModel.from_pretrained(base_model, adapted_adapter_path).to(device).eval()
    if adapted_model_name:
        return AutoModelForCausalLM.from_pretrained(adapted_model_name).to(device).eval()
    raise ValueError("Either adapted_model or adapted_adapter_path must be provided")


def _batch_prompts(prompts: list[str], batch_size: int) -> list[list[str]]:
    return [prompts[index : index + batch_size] for index in range(0, len(prompts), batch_size)]


def _evaluate_against_adapted(
    correction: Any | None,
    base_model: PreTrainedModel,
    adapted_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
) -> dict[str, float]:
    device = next(base_model.parameters()).device
    total_positions = 0
    kl_sum = 0.0
    js_sum = 0.0

    for batch_prompts in _batch_prompts(prompts, batch_size):
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            base_logits = base_model(**encoded).logits[:, :-1, :].float()
            adapted_logits = adapted_model(**encoded).logits[:, :-1, :].float()

        adjusted_logits = correction.apply(base_logits) if correction is not None else base_logits
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

        total_positions += positions
        kl_sum += float(kl.masked_select(mask).sum().item())
        js_sum += float(js.masked_select(mask).sum().item())

    if total_positions == 0:
        raise ValueError("Prompt set did not yield any valid next-token positions")
    return {
        "kl_divergence": kl_sum / total_positions,
        "js_divergence": js_sum / total_positions,
        "num_positions": float(total_positions),
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
    prompts_file = str(prompts_file_value)
    eval_prompts_file_value = get_config_value(config_data, args, "eval_prompts_file")
    eval_prompts_file = str(eval_prompts_file_value) if eval_prompts_file_value else prompts_file
    low_rank_k = int(get_config_value(config_data, args, "low_rank_k"))
    jacobian_probe_count = int(get_config_value(config_data, args, "jacobian_probe_count"))
    jacobian_max_params = int(get_config_value(config_data, args, "jacobian_max_params"))
    batch_size = int(get_config_value(config_data, args, "batch_size"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    calibration_prompts = _load_prompts(prompts_file)
    eval_prompts = _load_prompts(eval_prompts_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading base and adapted models on %s", device)
    base_model, tokenizer = _load_base_model_and_tokenizer(base_model_name, device=device)
    adapted_model = _load_adapted_model(
        base_model_name=base_model_name,
        adapted_model_name=adapted_model_name,
        adapted_adapter_path=adapted_adapter_path,
        device=device,
    )

    logger.info("Calibrating correction baselines on %d prompts", len(calibration_prompts))
    offset = DistributionOffsetCorrection().calibrate(
        base_model,
        adapted_model,
        calibration_prompts,
        tokenizer=tokenizer,
    )
    low_rank = LowRankCorrection(rank=low_rank_k).calibrate(
        base_model,
        adapted_model,
        calibration_prompts,
        tokenizer=tokenizer,
    )
    jacobian = JacobianCorrection(
        probe_count=jacobian_probe_count,
        max_params=jacobian_max_params,
        seed=args.seed,
    ).calibrate(
        base_model,
        adapted_model,
        calibration_prompts,
        tokenizer=tokenizer,
    )

    logger.info("Evaluating divergence reduction on %d prompts", len(eval_prompts))
    baseline_metrics = _evaluate_against_adapted(
        correction=None,
        base_model=base_model,
        adapted_model=adapted_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        batch_size=batch_size,
    )
    offset_metrics = _evaluate_against_adapted(
        correction=offset,
        base_model=base_model,
        adapted_model=adapted_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        batch_size=batch_size,
    )
    low_rank_metrics = _evaluate_against_adapted(
        correction=low_rank,
        base_model=base_model,
        adapted_model=adapted_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        batch_size=batch_size,
    )
    jacobian_metrics = _evaluate_against_adapted(
        correction=jacobian,
        base_model=base_model,
        adapted_model=adapted_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        batch_size=batch_size,
    )

    payload = {
        "baseline": baseline_metrics,
        "distribution_offset": offset_metrics,
        "low_rank": low_rank_metrics,
        "jacobian": jacobian_metrics,
        "improvements": {
            "distribution_offset_kl_reduction": baseline_metrics["kl_divergence"] - offset_metrics["kl_divergence"],
            "low_rank_kl_reduction": baseline_metrics["kl_divergence"] - low_rank_metrics["kl_divergence"],
            "jacobian_kl_reduction": baseline_metrics["kl_divergence"] - jacobian_metrics["kl_divergence"],
        },
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="analytical_correction",
        config={
            "base_model": base_model_name,
            "adapted_model": adapted_model_name,
            "adapted_adapter_path": adapted_adapter_path,
            "prompts_file": prompts_file,
            "eval_prompts_file": eval_prompts_file,
            "low_rank_k": low_rank_k,
            "jacobian_probe_count": jacobian_probe_count,
            "jacobian_max_params": jacobian_max_params,
            "batch_size": batch_size,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved analytical correction evaluation to %s", output)


if __name__ == "__main__":
    main()
