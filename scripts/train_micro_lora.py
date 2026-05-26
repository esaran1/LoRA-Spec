from __future__ import annotations

import argparse
from pathlib import Path

from lora_spec.distillation import DistillationConfig, train_micro_lora_adapter
from lora_spec.serving import load_prompts
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a micro-LoRA draft adapter by KL distillation.")
    add_common_args(parser)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--target-adapter-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--draft-lora-rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="checkpoints/micro_lora")
    parser.add_argument("--results-dir", type=str, default="results/distillation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "train_micro_lora")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)
    draft_model = get_config_value(config_data, args, "draft_model")
    target_model = get_config_value(config_data, args, "target_model")
    if not draft_model or not target_model:
        raise ValueError("Both draft_model and target_model must be provided")
    target_adapter_path = get_config_value(config_data, args, "target_adapter_path")
    dataset = str(get_config_value(config_data, args, "dataset"))
    num_prompts = int(get_config_value(config_data, args, "num_prompts"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    results_dir = str(get_config_value(config_data, args, "results_dir"))

    prompts = load_prompts(dataset, num_prompts=num_prompts, seed=args.seed)
    config = DistillationConfig(
        draft_lora_rank=int(get_config_value(config_data, args, "draft_lora_rank")),
        learning_rate=float(get_config_value(config_data, args, "learning_rate")),
        batch_size=int(get_config_value(config_data, args, "batch_size")),
        epochs=int(get_config_value(config_data, args, "epochs")),
        max_length=int(get_config_value(config_data, args, "max_length")),
        seed=args.seed,
    )
    checkpoint = train_micro_lora_adapter(
        draft_model=draft_model,
        target_model=target_model,
        prompts=prompts,
        output_dir=output_dir,
        config=config,
        adapter_path=target_adapter_path,
    )
    result_path = write_json_result(
        payload={
            "checkpoint_dir": str(Path(checkpoint).resolve()),
            "num_prompts": len(prompts),
            "dataset": dataset,
            "target_adapter_path": target_adapter_path,
            "distillation_config": config.__dict__,
        },
        output_dir=results_dir,
        stem="micro_lora_train",
        config={
            "draft_model": draft_model,
            "target_model": target_model,
            "target_adapter_path": target_adapter_path,
            "dataset": dataset,
            "num_prompts": num_prompts,
            "output_dir": output_dir,
            "distillation_config": config.__dict__,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved distilled micro-LoRA checkpoint to %s", checkpoint)
    logger.info("Saved distillation metadata to %s", result_path)


if __name__ == "__main__":
    main()
