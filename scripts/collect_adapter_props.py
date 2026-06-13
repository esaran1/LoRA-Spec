from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_spec.adapter_props import compute_adapter_properties, compute_distribution_divergence
from lora_spec.artifacts import materialize_artifact, resolve_artifact_revision
from lora_spec.prompts import load_frozen_prompt_texts, prompt_file_provenance
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
    parser.add_argument("--adapter-revision", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--base-revision", type=str, default=None)
    parser.add_argument("--adapted-model", type=str, default=None)
    parser.add_argument("--adapted-revision", type=str, default=None)
    parser.add_argument("--adapted-adapter-path", type=str, default=None)
    parser.add_argument("--adapted-adapter-revision", type=str, default=None)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/calibration.jsonl",
    )
    parser.add_argument("--magnitude-scale", type=float, default=1.0)
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
    magnitude_scale = float(get_config_value(config_data, args, "magnitude_scale"))
    output_dir = str(get_config_value(config_data, args, "output_dir"))

    adapter_artifact = resolve_artifact_revision(
        adapter_path,
        revision=get_config_value(config_data, args, "adapter_revision"),
    )
    adapter_load_path = materialize_artifact(adapter_artifact)
    base_artifact = (
        resolve_artifact_revision(
            str(base_model),
            revision=get_config_value(config_data, args, "base_revision"),
        )
        if base_model
        else None
    )
    adapted_artifact = (
        resolve_artifact_revision(
            str(adapted_model),
            revision=get_config_value(config_data, args, "adapted_revision"),
        )
        if adapted_model
        else None
    )
    base_load_path = materialize_artifact(base_artifact) if base_artifact else None
    adapted_load_path = materialize_artifact(adapted_artifact) if adapted_artifact else None
    adapted_adapter_artifact = (
        resolve_artifact_revision(
            str(adapted_adapter_path),
            revision=get_config_value(config_data, args, "adapted_adapter_revision"),
        )
        if adapted_adapter_path
        else None
    )
    adapted_adapter_load_path = (
        materialize_artifact(adapted_adapter_artifact) if adapted_adapter_artifact else None
    )

    properties = compute_adapter_properties(adapter_load_path, base_model=base_load_path)
    scaled_properties = {
        **properties.__dict__,
        "frobenius_norm_sum": properties.frobenius_norm_sum * magnitude_scale,
        "spectral_norm_sum": properties.spectral_norm_sum * magnitude_scale,
        "max_spectral_norm": properties.max_spectral_norm * magnitude_scale,
        "magnitude_scale": magnitude_scale,
    }
    payload: dict[str, object] = {"properties": scaled_properties}
    prompts_provenance: dict[str, object] | None = None
    if base_model and prompts_file:
        prompts = load_frozen_prompt_texts(prompts_file, expected_split="calibration")
        prompts_provenance = prompt_file_provenance(
            prompts_file,
            expected_split="calibration",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(base_load_path), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        adapted_tokenizer = tokenizer
        if adapted_model:
            adapted_reference: str | object = str(adapted_load_path)
            adapted_tokenizer = AutoTokenizer.from_pretrained(
                str(adapted_load_path),
                use_fast=True,
            )
            if adapted_tokenizer.pad_token is None:
                adapted_tokenizer.pad_token = adapted_tokenizer.eos_token
        else:
            base_instance = AutoModelForCausalLM.from_pretrained(str(base_load_path))
            adapted_reference = PeftModel.from_pretrained(
                base_instance,
                str(adapted_adapter_load_path),
            ).eval()
        divergence = compute_distribution_divergence(
            str(base_load_path),
            adapted_reference,
            prompts,
            tokenizer=tokenizer,
            adapted_tokenizer=adapted_tokenizer,
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
            "prompts_provenance": prompts_provenance,
            "artifact_provenance": {
                "adapter": adapter_artifact.to_dict(),
                "base_model": base_artifact.to_dict() if base_artifact else None,
                "adapted_model": adapted_artifact.to_dict() if adapted_artifact else None,
                "adapted_adapter": (
                    adapted_adapter_artifact.to_dict() if adapted_adapter_artifact else None
                ),
            },
            "magnitude_scale": magnitude_scale,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved adapter properties to %s", output)


if __name__ == "__main__":
    main()
