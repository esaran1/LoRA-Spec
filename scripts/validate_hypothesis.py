from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from lora_spec.config import AdapterConfig, ExperimentConfig, ModelPairConfig, ResultRecord
from lora_spec.serving import (
    build_sampling_params,
    initialize_vllm,
    load_prompts,
    run_speculative_generation,
)
from lora_spec.utils import (
    add_common_args,
    compute_config_hash,
    get_config_value,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def build_experiment_config(args: argparse.Namespace, config_data: dict[str, Any]) -> ExperimentConfig:
    target_model = get_config_value(config_data, args, "target_model")
    draft_model = get_config_value(config_data, args, "draft_model")
    adapter_path = get_config_value(config_data, args, "adapter_path")
    if not target_model or not draft_model:
        raise ValueError("Both target_model and draft_model must be provided")
    model_pair = ModelPairConfig(
        target_model=target_model,
        draft_model=draft_model,
        tensor_parallel_degree=int(
            config_data.get("tensor_parallel_degree", args.tensor_parallel_degree),
        ),
    )
    adapter = None
    if adapter_path:
        adapter = AdapterConfig(
            rank=int(config_data.get("adapter_rank", args.adapter_rank or 1)),
            domain=str(config_data.get("adapter_domain", args.adapter_domain or "unknown")),
            epochs=int(config_data.get("adapter_epochs", args.adapter_epochs or 1)),
            hf_path=str(adapter_path),
        )
    return ExperimentConfig(
        model_pair=model_pair,
        adapter=adapter,
        num_prompts=int(config_data.get("num_prompts", args.num_prompts)),
        dataset=str(config_data.get("dataset", args.dataset)),
        seed=int(config_data.get("seed", args.seed)),
        speculation_length=int(config_data.get("speculation_length", args.speculation_length)),
        max_tokens=int(config_data.get("max_tokens", args.max_tokens)),
        gpu_memory_utilization=float(
            config_data.get("gpu_memory_utilization", args.gpu_memory_utilization),
        ),
        trust_remote_code=bool(config_data.get("trust_remote_code", args.trust_remote_code)),
    )


def make_result_record(
    metrics: Any,
    config_hash: str,
    metadata: dict[str, Any],
) -> ResultRecord:
    return ResultRecord(
        config_hash=config_hash,
        acceptance_rate_overall=metrics.acceptance.overall_acceptance_rate,
        acceptance_rate_per_position=metrics.acceptance.per_position_acceptance_rate,
        throughput_tps=metrics.timing.throughput_tps,
        ttft_ms=metrics.timing.ttft_ms,
        metadata=metadata,
    )


def hypothesis_recommendation(baseline_rate: float, adapted_rate: float, baseline_tps: float, adapted_tps: float) -> str:
    acceptance_delta = adapted_rate - baseline_rate
    throughput_delta = (adapted_tps - baseline_tps) / baseline_tps if baseline_tps > 0 else 0.0
    if acceptance_delta <= -0.05 or throughput_delta <= -0.10:
        return "go"
    return "no-go"


def run_validation(experiment: ExperimentConfig, adapter_path: str | None, logger: logging.Logger) -> dict[str, Any]:
    set_seed(experiment.seed)
    prompts = load_prompts(
        dataset_name=experiment.dataset,
        num_prompts=experiment.num_prompts,
        seed=experiment.seed,
    )
    llm = initialize_vllm(
        target_model=experiment.model_pair.target_model,
        draft_model=experiment.model_pair.draft_model,
        tensor_parallel_size=experiment.model_pair.tensor_parallel_degree,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
        speculation_length=experiment.speculation_length,
        enable_lora=adapter_path is not None,
        trust_remote_code=experiment.trust_remote_code,
    )
    sampling_params = build_sampling_params(max_tokens=experiment.max_tokens)
    config_hash = compute_config_hash(experiment)

    logger.info("Running baseline speculative decoding without LoRA")
    baseline = run_speculative_generation(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        speculation_length=experiment.speculation_length,
    )

    if adapter_path is None:
        raise ValueError("adapter_path must be provided for hypothesis validation")

    logger.info("Running adapted speculative decoding with LoRA adapter %s", adapter_path)
    adapted = run_speculative_generation(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        speculation_length=experiment.speculation_length,
        adapter_path=adapter_path,
    )

    acceptance_delta = (
        adapted.metrics.acceptance.overall_acceptance_rate
        - baseline.metrics.acceptance.overall_acceptance_rate
    )
    throughput_delta = adapted.metrics.timing.throughput_tps - baseline.metrics.timing.throughput_tps
    recommendation = hypothesis_recommendation(
        baseline.metrics.acceptance.overall_acceptance_rate,
        adapted.metrics.acceptance.overall_acceptance_rate,
        baseline.metrics.timing.throughput_tps,
        adapted.metrics.timing.throughput_tps,
    )
    logger.info(
        "Baseline acceptance %.4f | Adapted acceptance %.4f | Delta %.4f",
        baseline.metrics.acceptance.overall_acceptance_rate,
        adapted.metrics.acceptance.overall_acceptance_rate,
        acceptance_delta,
    )
    logger.info(
        "Baseline throughput %.2f tok/s | Adapted throughput %.2f tok/s | Delta %.2f tok/s",
        baseline.metrics.timing.throughput_tps,
        adapted.metrics.timing.throughput_tps,
        throughput_delta,
    )
    logger.info("Phase 1 recommendation: %s", recommendation)

    summary = {
        "baseline": make_result_record(
            baseline.metrics,
            config_hash=config_hash,
            metadata={
                "condition": "baseline",
                "model_pair": experiment.model_pair.model_dump(),
                "dataset": experiment.dataset,
            },
        ).model_dump(mode="json"),
        "adapted": make_result_record(
            adapted.metrics,
            config_hash=config_hash,
            metadata={
                "condition": "adapted",
                "adapter_path": adapter_path,
                "dataset": experiment.dataset,
            },
        ).model_dump(mode="json"),
        "comparison": {
            "acceptance_delta": acceptance_delta,
            "throughput_delta_tps": throughput_delta,
            "recommendation": recommendation,
        },
    }
    return {
        "summary": summary,
        "prompts": prompts,
        "texts": {
            "baseline": baseline.texts,
            "adapted": adapted.texts,
        },
        "experiment": experiment.model_dump(mode="json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the LoRA-Spec core hypothesis.")
    add_common_args(parser)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, required=False)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-domain", type=str, default=None)
    parser.add_argument("--adapter-epochs", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--tensor-parallel-degree", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/phase1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, logger_name="validate_hypothesis")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)
    experiment = build_experiment_config(args, config_data)
    adapter_path = get_config_value(config_data, args, "adapter_path")
    payload = run_validation(experiment, adapter_path=adapter_path, logger=logger)
    output_path = write_json_result(
        payload={**payload["summary"], "prompts": payload["prompts"], "texts": payload["texts"]},
        output_dir=str(get_config_value(config_data, args, "output_dir")),
        stem="phase1_validation",
        config=payload["experiment"],
        extra_metadata={"adapter_path": adapter_path},
        cwd=Path.cwd(),
    )
    logger.info("Saved Phase 1 results to %s", output_path)


if __name__ == "__main__":
    main()
