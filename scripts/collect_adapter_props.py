from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_spec.adapter_props import compute_adapter_properties, compute_distribution_divergence
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LoRA adapter properties.")
    add_common_args(parser)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--adapted-adapter-path", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/adapter_props")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "collect_adapter_props")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)
    adapter_path_value = get_config_value(config_data, args, "adapter_path")
    if not adapter_path_value:
        raise ValueError("adapter_path must be provided")
    adapter_path = str(adapter_path_value)
    base_model = get_config_value(config_data, args, "base_model")
    adapted_model = get_config_value(config_data, args, "adapted_model")
    adapted_adapter_path = get_config_value(config_data, args, "adapted_adapter_path") or adapter_path
    prompts_file = get_config_value(config_data, args, "prompts_file")
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    properties = compute_adapter_properties(adapter_path, base_model=base_model)
    payload: dict[str, object] = {"properties": properties.__dict__}
    if base_model and prompts_file:
        prompts = [
            line.strip()
            for line in Path(prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tokenizer = AutoTokenizer.from_pretrained(str(base_model), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if adapted_model:
            adapted_reference: str | object = adapted_model
        else:
            base_instance = AutoModelForCausalLM.from_pretrained(str(base_model))
            adapted_reference = PeftModel.from_pretrained(base_instance, str(adapted_adapter_path)).eval()
        divergence = compute_distribution_divergence(
            base_model,
            adapted_reference,
            prompts,
            tokenizer=tokenizer,
            adapted_tokenizer=tokenizer,
        )
        payload["divergence"] = divergence.__dict__
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="adapter_props",
        config={
            "adapter_path": adapter_path,
            "base_model": base_model,
            "adapted_model": adapted_model,
            "adapted_adapter_path": adapted_adapter_path,
            "prompts_file": prompts_file,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved adapter properties to %s", output)


if __name__ == "__main__":
    main()
