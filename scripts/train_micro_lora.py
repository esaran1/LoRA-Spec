from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from lora_spec.artifacts import materialize_artifact, resolve_artifact_revision
from lora_spec.distillation import DistillationConfig, train_micro_lora_adapter
from lora_spec.prompts import (
    FrozenPromptRecord,
    prompt_file_provenance,
    resolve_registered_prompt_split,
)
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a micro-LoRA draft adapter by KL distillation."
    )
    add_common_args(parser)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--draft-revision", type=str, default=None)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--target-revision", type=str, default=None)
    parser.add_argument("--target-adapter-path", type=str, default=None)
    parser.add_argument("--target-adapter-revision", type=str, default=None)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/calibration.jsonl",
    )
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument("--num-validation-prompts", type=int, default=4)
    parser.add_argument("--draft-lora-rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="checkpoints/micro_lora")
    parser.add_argument("--results-dir", type=str, default="results/distillation")
    return parser.parse_args()


def _partition_calibration_prompts(
    prompts_file: str,
    train_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[str], list[str], dict[str, object], dict[str, object]]:
    if train_count < 1 or validation_count < 1:
        raise ValueError("Training and validation prompt counts must be positive")
    registered = resolve_registered_prompt_split(prompts_file, expected_split="calibration")
    if train_count + validation_count > len(registered.records):
        raise ValueError(
            "Requested training and validation prompts exceed the frozen calibration split"
        )
    by_domain: dict[str, list[FrozenPromptRecord]] = defaultdict(list)
    for record in registered.records:
        by_domain[record.domain].append(record)
    rng = random.Random(seed)
    for records in by_domain.values():
        rng.shuffle(records)

    validation_records: list[FrozenPromptRecord] = []
    domains = sorted(by_domain)
    while len(validation_records) < validation_count:
        made_progress = False
        for domain in domains:
            if by_domain[domain] and len(validation_records) < validation_count:
                validation_records.append(by_domain[domain].pop())
                made_progress = True
        if not made_progress:
            break
    remaining = [record for domain in domains for record in by_domain[domain]]
    rng.shuffle(remaining)
    training_records = remaining[:train_count]
    if len(training_records) != train_count or len(validation_records) != validation_count:
        raise RuntimeError("Could not construct the requested disjoint calibration partition")

    base_provenance = prompt_file_provenance(prompts_file, expected_split="calibration")

    def provenance(records: list[FrozenPromptRecord], role: str) -> dict[str, object]:
        identifiers = [str(record.id) for record in records]
        digest = hashlib.sha256(
            json.dumps(identifiers, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            **base_provenance,
            "partition_role": role,
            "partition_seed": seed,
            "selected_prompt_ids": identifiers,
            "selected_prompt_ids_sha256": digest,
        }

    return (
        [str(record.text) for record in training_records],
        [str(record.text) for record in validation_records],
        provenance(training_records, "distillation_train"),
        provenance(validation_records, "distillation_validation"),
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "train_micro_lora")
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)
    draft_model = get_config_value(config_data, args, "draft_model")
    target_model = get_config_value(config_data, args, "target_model")
    if not draft_model or not target_model:
        raise ValueError("Both draft_model and target_model must be provided")
    target_adapter_path = get_config_value(config_data, args, "target_adapter_path")
    prompts_file = str(get_config_value(config_data, args, "prompts_file"))
    num_prompts = int(get_config_value(config_data, args, "num_prompts"))
    num_validation_prompts = int(get_config_value(config_data, args, "num_validation_prompts"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    results_dir = str(get_config_value(config_data, args, "results_dir"))

    draft_artifact = resolve_artifact_revision(
        str(draft_model),
        revision=get_config_value(config_data, args, "draft_revision"),
    )
    target_artifact = resolve_artifact_revision(
        str(target_model),
        revision=get_config_value(config_data, args, "target_revision"),
    )
    adapter_artifact = (
        resolve_artifact_revision(
            str(target_adapter_path),
            revision=get_config_value(config_data, args, "target_adapter_revision"),
        )
        if target_adapter_path
        else None
    )
    (
        prompts,
        validation_prompts,
        prompts_provenance,
        validation_prompts_provenance,
    ) = _partition_calibration_prompts(
        prompts_file,
        train_count=num_prompts,
        validation_count=num_validation_prompts,
        seed=seed,
    )
    config = DistillationConfig(
        draft_lora_rank=int(get_config_value(config_data, args, "draft_lora_rank")),
        learning_rate=float(get_config_value(config_data, args, "learning_rate")),
        batch_size=int(get_config_value(config_data, args, "batch_size")),
        epochs=int(get_config_value(config_data, args, "epochs")),
        max_length=int(get_config_value(config_data, args, "max_length")),
        seed=seed,
        torch_dtype=str(get_config_value(config_data, args, "torch_dtype")),
        gradient_accumulation_steps=int(
            get_config_value(config_data, args, "gradient_accumulation_steps")
        ),
        max_grad_norm=float(get_config_value(config_data, args, "max_grad_norm")),
        continuation_tokens=int(get_config_value(config_data, args, "continuation_tokens")),
    )
    checkpoint = train_micro_lora_adapter(
        draft_model=materialize_artifact(draft_artifact),
        target_model=materialize_artifact(target_artifact),
        prompts=prompts,
        validation_prompts=validation_prompts,
        output_dir=output_dir,
        config=config,
        adapter_path=materialize_artifact(adapter_artifact) if adapter_artifact else None,
    )
    contexts_metadata = json.loads(
        (Path(checkpoint) / "distillation_contexts.json").read_text(encoding="utf-8")
    )
    training_metrics = json.loads(
        (Path(checkpoint) / "training_metrics.json").read_text(encoding="utf-8")
    )
    result_path = write_json_result(
        payload={
            "checkpoint_dir": str(Path(checkpoint).resolve()),
            "num_prompts": len(prompts),
            "prompts_file": prompts_file,
            "prompts_provenance": prompts_provenance,
            "validation_prompts_provenance": validation_prompts_provenance,
            "target_adapter_path": target_adapter_path,
            "distillation_config": config.__dict__,
            "distillation_contexts": contexts_metadata,
            "training_metrics": training_metrics,
        },
        output_dir=results_dir,
        stem="micro_lora_train",
        config={
            "draft_model": draft_model,
            "target_model": target_model,
            "target_adapter_path": target_adapter_path,
            "prompts_file": prompts_file,
            "prompts_provenance": prompts_provenance,
            "validation_prompts_provenance": validation_prompts_provenance,
            "num_prompts": num_prompts,
            "num_validation_prompts": num_validation_prompts,
            "output_dir": output_dir,
            "distillation_config": config.__dict__,
            "distillation_contexts": contexts_metadata,
            "artifact_provenance": {
                "draft_model": draft_artifact.to_dict(),
                "target_model": target_artifact.to_dict(),
                "target_adapter": adapter_artifact.to_dict() if adapter_artifact else None,
            },
            "seed": seed,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved distilled micro-LoRA checkpoint to %s", checkpoint)
    logger.info("Saved distillation metadata to %s", result_path)


if __name__ == "__main__":
    main()
