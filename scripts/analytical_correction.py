from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_spec.correction import DistributionOffsetCorrection, JacobianCorrection, LowRankCorrection
from lora_spec.utils import add_common_args, get_config_value, resolve_config, setup_logging, write_json_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate analytical correction baselines.")
    add_common_args(parser)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--low-rank-k", type=int, default=8)
    parser.add_argument("--jacobian-probe-count", type=int, default=8)
    parser.add_argument("--jacobian-max-params", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="results/correction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "analytical_correction")
    config_data = resolve_config(args.config, args.override)
    base_model_value = get_config_value(config_data, args, "base_model")
    adapted_model_value = get_config_value(config_data, args, "adapted_model")
    prompts_file_value = get_config_value(config_data, args, "prompts_file")
    if not base_model_value or not adapted_model_value or not prompts_file_value:
        raise ValueError("base_model, adapted_model, and prompts_file must be provided")
    base_model_name = str(base_model_value)
    adapted_model_name = str(adapted_model_value)
    prompts_file = str(prompts_file_value)
    low_rank_k = int(get_config_value(config_data, args, "low_rank_k"))
    jacobian_probe_count = int(get_config_value(config_data, args, "jacobian_probe_count"))
    jacobian_max_params = int(get_config_value(config_data, args, "jacobian_max_params"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    prompts = [
        line.strip()
        for line in Path(prompts_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    adapted_model = AutoModelForCausalLM.from_pretrained(adapted_model_name)

    offset = DistributionOffsetCorrection().calibrate(base_model, adapted_model, prompts, tokenizer=tokenizer)
    low_rank = LowRankCorrection(rank=low_rank_k).calibrate(base_model, adapted_model, prompts, tokenizer=tokenizer)
    jacobian = JacobianCorrection(
        probe_count=jacobian_probe_count,
        max_params=jacobian_max_params,
        seed=args.seed,
    ).calibrate(base_model, adapted_model, prompts, tokenizer=tokenizer)

    sample_logits = torch.zeros(1, tokenizer.vocab_size)
    payload = {
        "distribution_offset_norm": float(offset.apply(sample_logits).norm().item()),
        "low_rank_norm": float(low_rank.apply(sample_logits).norm().item()),
        "jacobian_norm": float(jacobian.apply(sample_logits).norm().item()),
        "low_rank_k": low_rank_k,
        "jacobian_probe_count": jacobian_probe_count,
        "jacobian_max_params": jacobian_max_params,
    }
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="analytical_correction",
        config={
            "base_model": base_model_name,
            "adapted_model": adapted_model_name,
            "prompts_file": prompts_file,
            "num_prompts": len(prompts),
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved analytical correction summary to %s", output)


if __name__ == "__main__":
    main()
