from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from lora_spec.artifacts import materialize_artifact, resolve_artifact_revision
from lora_spec.config import AdapterConfig, ExperimentConfig, ModelPairConfig
from lora_spec.prompts import prompt_file_provenance, resolve_registered_prompt_split
from lora_spec.utils import (
    add_common_args,
    canonical_json,
    compute_config_hash,
    ensure_dir,
    get_config_value,
    load_yaml,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)
from validate_hypothesis import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA-Spec characterization sweeps.")
    add_common_args(parser)
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--adapters-config", type=str, default="configs/adapters.yaml")
    parser.add_argument("--dataset", type=str, default="lora-spec-pilot-v1/evaluation")
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/evaluation.jsonl",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--measurement-repetitions", type=int, default=3)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--runs-dir", type=str, default="results/characterize/runs")
    parser.add_argument("--output-dir", type=str, default="results/characterize")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_sweep_inputs(models_path: str, adapters_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    models_payload = load_yaml(models_path)
    adapters_payload = load_yaml(adapters_path)
    model_pairs = models_payload.get("model_pairs", {})
    adapters = adapters_payload.get("adapters", {})
    if not isinstance(model_pairs, dict) or not model_pairs:
        raise ValueError(f"{models_path} must contain a non-empty model_pairs mapping")
    if not isinstance(adapters, dict) or not adapters:
        raise ValueError(f"{adapters_path} must contain a non-empty adapters mapping")
    return model_pairs, adapters_payload


def _matches_filters(
    model_name: str,
    adapter_name: str,
    adapter_values: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    if filters.get("selected_model") and model_name != filters["selected_model"]:
        return False
    if filters.get("selected_adapter") and adapter_name != filters["selected_adapter"]:
        return False
    if filters.get("selected_rank") is not None and int(adapter_values["rank"]) != int(filters["selected_rank"]):
        return False
    if filters.get("selected_domain") and str(adapter_values["domain"]) != str(filters["selected_domain"]):
        return False
    if filters.get("selected_epochs") is not None:
        if adapter_values.get("epochs") is None:
            return False
        if int(adapter_values["epochs"]) != int(filters["selected_epochs"]):
            return False
    return True


def _build_manifest_entries(
    model_pairs: dict[str, Any],
    adapters_payload: dict[str, Any],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    adapters = adapters_payload["adapters"]
    experiment_entries = adapters_payload.get("experiments")
    manifest: list[dict[str, Any]] = []

    if isinstance(experiment_entries, list) and experiment_entries:
        for entry in experiment_entries:
            if not isinstance(entry, dict):
                raise ValueError("Each adapters.yaml experiments entry must be a mapping")
            model_name = str(entry["model_pair"])
            adapter_name = str(entry["adapter"])
            if model_name not in model_pairs:
                raise KeyError(f"Unknown model_pair in manifest: {model_name}")
            if adapter_name not in adapters:
                raise KeyError(f"Unknown adapter in manifest: {adapter_name}")
            adapter_values = dict(adapters[adapter_name])
            if not _matches_filters(model_name, adapter_name, adapter_values, filters):
                continue
            manifest.append(
                {
                    "model_pair_name": model_name,
                    "adapter_name": adapter_name,
                    "notes": entry.get("notes", ""),
                    "tags": entry.get("tags", []),
                }
            )
        return manifest

    for model_name in model_pairs:
        for adapter_name, adapter_values in adapters.items():
            if not _matches_filters(model_name, adapter_name, adapter_values, filters):
                continue
            manifest.append(
                {
                    "model_pair_name": model_name,
                    "adapter_name": adapter_name,
                    "notes": "",
                    "tags": [],
                }
            )
    return manifest


def _run_config_fingerprint(
    experiment: ExperimentConfig,
    model_pair_name: str,
    adapter_name: str,
    prompt_metadata: dict[str, Any],
    artifact_provenance: dict[str, Any],
) -> str:
    return compute_config_hash(
        {
            "model_pair_name": model_pair_name,
            "adapter_name": adapter_name,
            "experiment": experiment.model_dump(mode="json"),
            "prompt_metadata": prompt_metadata,
            "artifact_provenance": artifact_provenance,
        }
    )


def _load_existing_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "characterize")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)

    models_config = str(get_config_value(config_data, args, "models_config"))
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    dataset = str(get_config_value(config_data, args, "dataset"))
    prompts_file = str(get_config_value(config_data, args, "prompts_file"))
    num_prompts = int(get_config_value(config_data, args, "num_prompts"))
    measurement_repetitions = int(
        get_config_value(config_data, args, "measurement_repetitions"),
    )
    speculation_length = int(get_config_value(config_data, args, "speculation_length"))
    max_tokens = int(get_config_value(config_data, args, "max_tokens"))
    gpu_memory_utilization = float(get_config_value(config_data, args, "gpu_memory_utilization"))
    runs_dir = ensure_dir(str(get_config_value(config_data, args, "runs_dir")))
    output_dir = str(get_config_value(config_data, args, "output_dir"))
    resume = bool(get_config_value(config_data, args, "resume", args.resume))

    filters = {
        "selected_model": config_data.get("selected_model"),
        "selected_adapter": config_data.get("selected_adapter"),
        "selected_rank": config_data.get("selected_rank"),
        "selected_domain": config_data.get("selected_domain"),
        "selected_epochs": config_data.get("selected_epochs"),
    }

    model_pairs, adapters_payload = _load_sweep_inputs(models_config, adapters_config)
    manifest = _build_manifest_entries(model_pairs, adapters_payload, filters)
    if not manifest:
        raise ValueError("No characterization runs matched the requested manifest and filters")

    registered_prompts = resolve_registered_prompt_split(
        prompts_file,
        expected_split="evaluation",
    )
    if num_prompts > len(registered_prompts.records):
        raise ValueError(
            f"Requested {num_prompts} prompts, but frozen evaluation split has "
            f"{len(registered_prompts.records)}",
        )
    selected_indices = list(range(len(registered_prompts.records)))
    random.Random(seed).shuffle(selected_indices)
    selected_records = [
        registered_prompts.records[index] for index in selected_indices[:num_prompts]
    ]
    prompts = [record.text for record in selected_records]
    prompt_metadata = {
        **prompt_file_provenance(prompts_file, expected_split="evaluation"),
        "selected_prompt_ids": [record.id for record in selected_records],
        "selection_seed": seed,
    }

    run_records: list[dict[str, Any]] = []
    for entry in manifest:
        model_pair_name = entry["model_pair_name"]
        adapter_name = entry["adapter_name"]
        model_values = dict(model_pairs[model_pair_name])
        adapter_values = dict(adapters_payload["adapters"][adapter_name])
        compatible_target = adapter_values.get("target_model")
        configured_target = model_values.get("target_model")
        if compatible_target and compatible_target != configured_target:
            raise ValueError(
                f"Adapter {adapter_name} targets {compatible_target}, not {configured_target} "
                f"from model pair {model_pair_name}"
            )
        logger.info("Preparing characterization for %s x %s", model_pair_name, adapter_name)

        target_artifact = resolve_artifact_revision(
            str(model_values["target_model"]),
            revision=model_values.get("target_revision"),
        )
        draft_artifact = resolve_artifact_revision(
            str(model_values["draft_model"]),
            revision=model_values.get("draft_revision"),
        )
        adapter_artifact = resolve_artifact_revision(
            str(adapter_values["hf_path"]),
            revision=adapter_values.get("revision"),
        )
        model_values["target_revision"] = target_artifact.resolved_revision
        model_values["draft_revision"] = draft_artifact.resolved_revision
        adapter_values["revision"] = adapter_artifact.resolved_revision

        experiment = ExperimentConfig(
            model_pair=ModelPairConfig(**model_values),
            adapter=AdapterConfig(**adapter_values),
            num_prompts=num_prompts,
            dataset=dataset,
            prompts_file=prompts_file,
            seed=seed,
            measurement_repetitions=measurement_repetitions,
            speculation_length=speculation_length,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        artifact_provenance = {
            "target_model": target_artifact.to_dict(),
            "draft_model": draft_artifact.to_dict(),
            "adapter": adapter_artifact.to_dict(),
        }
        run_fingerprint = _run_config_fingerprint(
            experiment,
            model_pair_name,
            adapter_name,
            prompt_metadata,
            artifact_provenance,
        )
        run_path = runs_dir / f"{model_pair_name}__{adapter_name}__{run_fingerprint}.json"

        if resume and run_path.exists():
            logger.info("Resuming from existing run artifact %s", run_path.name)
            payload = _load_existing_run(run_path)
        else:
            result = run_validation(
                experiment,
                adapter_path=materialize_artifact(adapter_artifact),
                prompts=prompts,
                prompt_metadata=prompt_metadata,
                artifact_provenance=artifact_provenance,
                target_load_path=materialize_artifact(target_artifact),
                draft_load_path=materialize_artifact(draft_artifact),
                logger=logger,
            )
            payload = {
                "model_pair_name": model_pair_name,
                "adapter_name": adapter_name,
                "notes": entry.get("notes", ""),
                "tags": entry.get("tags", []),
                "result": result["summary"],
                "experiment": result["experiment"],
                "prompt_metadata": prompt_metadata,
                "artifact_provenance": artifact_provenance,
            }
            run_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            logger.info("Wrote run artifact to %s", run_path)

        run_records.append(payload)

    aggregate = {
        "runs": run_records,
        "manifest": manifest,
        "summary": {
            "num_runs": len(run_records),
            "dataset": dataset,
            "prompts_file": prompts_file,
            "prompt_metadata": prompt_metadata,
            "num_prompts": num_prompts,
            "measurement_repetitions": measurement_repetitions,
        },
    }
    output = write_json_result(
        payload=aggregate,
        output_dir=output_dir,
        stem="characterize",
        config={
            "models_config": models_config,
            "adapters_config": adapters_config,
            "dataset": dataset,
            "prompts_file": prompts_file,
            "prompt_metadata": prompt_metadata,
            "num_prompts": num_prompts,
            "measurement_repetitions": measurement_repetitions,
            "speculation_length": speculation_length,
            "max_tokens": max_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "resume": resume,
            "filters": filters,
            "manifest_hash": compute_config_hash(canonical_json(manifest)),
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved characterization aggregate to %s", output)


if __name__ == "__main__":
    main()
