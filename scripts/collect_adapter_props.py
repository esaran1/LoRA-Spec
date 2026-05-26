from __future__ import annotations

import argparse
from pathlib import Path

from lora_spec.adapter_props import compute_adapter_properties, compute_distribution_divergence
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LoRA adapter properties.")
    add_common_args(parser)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/adapter_props")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "collect_adapter_props")
    config_data = resolve_config(args.config, args.override)
    adapter_path_value = get_config_value(config_data, args, "adapter_path")
    if not adapter_path_value:
        raise ValueError("adapter_path must be provided")
    adapter_path = str(adapter_path_value)
    base_model = get_config_value(config_data, args, "base_model")
    adapted_model = get_config_value(config_data, args, "adapted_model")
    prompts_file = get_config_value(config_data, args, "prompts_file")
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    properties = compute_adapter_properties(adapter_path, base_model=base_model)
    payload: dict[str, object] = {"properties": properties.__dict__}
    if base_model and adapted_model and prompts_file:
        prompts = [
            line.strip()
            for line in Path(prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        divergence = compute_distribution_divergence(base_model, adapted_model, prompts)
        payload["divergence"] = divergence.__dict__
    output = write_json_result(
        payload=payload,
        output_dir=output_dir,
        stem="adapter_props",
        config={
            "adapter_path": adapter_path,
            "base_model": base_model,
            "adapted_model": adapted_model,
            "prompts_file": prompts_file,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved adapter properties to %s", output)


if __name__ == "__main__":
    main()
