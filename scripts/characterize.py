from __future__ import annotations

import argparse
from pathlib import Path

from lora_spec.config import AdapterConfig, ExperimentConfig, ModelPairConfig
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    load_yaml,
    resolve_config,
    setup_logging,
    write_json_result,
)
from validate_hypothesis import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep multiple adapters/model pairs.")
    add_common_args(parser)
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--adapters-config", type=str, default="configs/adapters.yaml")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="results/characterize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "characterize")
    config_data = resolve_config(args.config, args.override)
    models_config = str(get_config_value(config_data, args, "models_config"))
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    dataset = str(get_config_value(config_data, args, "dataset"))
    num_prompts = int(get_config_value(config_data, args, "num_prompts"))
    speculation_length = int(get_config_value(config_data, args, "speculation_length"))
    max_tokens = int(get_config_value(config_data, args, "max_tokens"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    models = load_yaml(models_config).get("model_pairs", {})
    adapters = load_yaml(adapters_config).get("adapters", {})
    if not models or not adapters:
        raise ValueError("models.yaml and adapters.yaml must contain entries")
    selected_model = config_data.get("selected_model")
    selected_adapter = config_data.get("selected_adapter")

    records: list[dict[str, object]] = []
    for model_name, model_values in models.items():
        if selected_model and model_name != selected_model:
            continue
        for adapter_name, adapter_values in adapters.items():
            if selected_adapter and adapter_name != selected_adapter:
                continue
            logger.info("Running characterization for %s x %s", model_name, adapter_name)
            experiment = ExperimentConfig(
                model_pair=ModelPairConfig(**model_values),
                adapter=AdapterConfig(**adapter_values),
                num_prompts=num_prompts,
                dataset=dataset,
                seed=args.seed,
                speculation_length=speculation_length,
                max_tokens=max_tokens,
            )
            result = run_validation(experiment, adapter_path=adapter_values["hf_path"], logger=logger)
            records.append(
                {
                    "model_pair_name": model_name,
                    "adapter_name": adapter_name,
                    "result": result["summary"],
                }
            )
    output = write_json_result(
        payload={"runs": records},
        output_dir=output_dir,
        stem="characterize",
        config={
            "models_config": models_config,
            "adapters_config": adapters_config,
            "dataset": dataset,
            "num_prompts": num_prompts,
            "speculation_length": speculation_length,
            "selected_model": selected_model,
            "selected_adapter": selected_adapter,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved characterization sweep to %s", output)


if __name__ == "__main__":
    main()
