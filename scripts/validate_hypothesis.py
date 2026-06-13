from __future__ import annotations

import argparse
import logging
import math
import random
import statistics
from pathlib import Path
from typing import Any

from lora_spec.adapter_props import read_adapter_metadata
from lora_spec.artifacts import materialize_artifact, resolve_artifact_revision
from lora_spec.config import AdapterConfig, ExperimentConfig, ModelPairConfig, ResultRecord
from lora_spec.metrics import AcceptanceMetrics, SpeculativeDecodingMetrics, TimingMetrics
from lora_spec.prompts import prompt_file_provenance, resolve_registered_prompt_split
from lora_spec.serving import (
    build_sampling_params,
    initialize_vllm,
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
        target_revision=get_config_value(config_data, args, "target_revision"),
        draft_revision=get_config_value(config_data, args, "draft_revision"),
        tensor_parallel_degree=int(
            config_data.get("tensor_parallel_degree", args.tensor_parallel_degree),
        ),
    )
    adapter = None
    if adapter_path:
        adapter_metadata = read_adapter_metadata(
            str(adapter_path),
            revision=get_config_value(config_data, args, "adapter_revision"),
        )
        configured_rank = config_data.get("adapter_rank", args.adapter_rank)
        metadata_rank = adapter_metadata.get("r")
        if configured_rank is None and metadata_rank is None:
            raise ValueError("Adapter rank is absent from both CLI/config and adapter_config.json")
        adapter_rank = int(configured_rank if configured_rank is not None else metadata_rank)
        if metadata_rank is not None and adapter_rank != int(metadata_rank):
            raise ValueError(
                f"Configured adapter rank {adapter_rank} does not match checkpoint metadata rank {metadata_rank}"
            )
        metadata_base = adapter_metadata.get("base_model_name_or_path")
        if isinstance(metadata_base, str) and metadata_base and not metadata_base.startswith("/"):
            if metadata_base != str(target_model):
                raise ValueError(f"Adapter targets {metadata_base}, but target_model is {target_model}")
        adapter = AdapterConfig(
            rank=adapter_rank,
            domain=str(config_data.get("adapter_domain", args.adapter_domain or "unknown")),
            epochs=config_data.get("adapter_epochs", args.adapter_epochs),
            hf_path=str(adapter_path),
            revision=get_config_value(config_data, args, "adapter_revision"),
            target_model=str(target_model),
        )
    return ExperimentConfig(
        model_pair=model_pair,
        adapter=adapter,
        num_prompts=int(config_data.get("num_prompts", args.num_prompts)),
        dataset=str(config_data.get("dataset", args.dataset)),
        prompts_file=str(get_config_value(config_data, args, "prompts_file")),
        seed=int(config_data.get("seed", args.seed)),
        measurement_repetitions=int(
            config_data.get("measurement_repetitions", args.measurement_repetitions),
        ),
        speculation_length=int(config_data.get("speculation_length", args.speculation_length)),
        max_tokens=int(config_data.get("max_tokens", args.max_tokens)),
        warmup_prompts=int(config_data.get("warmup_prompts", args.warmup_prompts)),
        warmup_tokens=int(config_data.get("warmup_tokens", args.warmup_tokens)),
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
    metadata = {
        **metadata,
        "acceptance_instrumentation_backend": metrics.instrumentation_backend,
        "accepted_drafted_tokens": metrics.acceptance.accepted_drafted_tokens,
        "total_drafted_tokens": metrics.acceptance.total_drafted_tokens,
        "bonus_tokens": metrics.acceptance.bonus_tokens,
    }
    return ResultRecord(
        config_hash=config_hash,
        acceptance_rate_overall=metrics.acceptance.overall_acceptance_rate,
        acceptance_rate_per_position=metrics.acceptance.per_position_acceptance_rate,
        throughput_tps=metrics.timing.throughput_tps,
        ttft_ms=metrics.timing.ttft_ms,
        metadata=metadata,
    )


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty measurement list")
    mean = statistics.mean(values)
    if len(values) == 1:
        return float(mean), float(mean), float(mean)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    t_critical_by_degrees_of_freedom = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    degrees_of_freedom = len(values) - 1
    eligible = [key for key in t_critical_by_degrees_of_freedom if key <= degrees_of_freedom]
    critical_value = (
        t_critical_by_degrees_of_freedom[max(eligible)] if eligible else 12.706
    )
    half_width = critical_value * standard_error
    return float(mean), float(mean - half_width), float(mean + half_width)


def _aggregate_metrics(runs: list[Any]) -> SpeculativeDecodingMetrics:
    if not runs:
        raise ValueError("At least one measured run is required")
    metrics = [run.metrics for run in runs]
    maximum_positions = max(len(item.acceptance.per_position_attempts) for item in metrics)
    attempts = [0] * maximum_positions
    accepted = [0] * maximum_positions
    depth_numerators = [0.0] * maximum_positions
    depth_denominators = [0] * maximum_positions
    for item in metrics:
        for index, count in enumerate(item.acceptance.per_position_attempts):
            attempts[index] += count
            accepted[index] += item.acceptance.per_position_accepted[index]
        for index, rate in enumerate(item.acceptance.acceptance_by_depth):
            depth_numerators[index] += rate * item.acceptance.speculative_steps
            depth_denominators[index] += item.acceptance.speculative_steps
    total_drafted = sum(item.acceptance.total_drafted_tokens for item in metrics)
    total_accepted = sum(item.acceptance.accepted_drafted_tokens for item in metrics)
    ttft_values = [item.timing.ttft_ms for item in metrics if math.isfinite(item.timing.ttft_ms)]
    backends = sorted({item.instrumentation_backend for item in metrics})
    return SpeculativeDecodingMetrics(
        acceptance=AcceptanceMetrics(
            overall_acceptance_rate=total_accepted / max(total_drafted, 1),
            per_position_acceptance_rate=[
                accepted_count / attempt_count if attempt_count else 0.0
                for accepted_count, attempt_count in zip(accepted, attempts)
            ],
            per_position_attempts=attempts,
            per_position_accepted=accepted,
            acceptance_by_depth=[
                numerator / denominator if denominator else 0.0
                for numerator, denominator in zip(depth_numerators, depth_denominators)
            ],
            accepted_drafted_tokens=total_accepted,
            total_drafted_tokens=total_drafted,
            bonus_tokens=sum(item.acceptance.bonus_tokens for item in metrics),
            speculative_steps=sum(item.acceptance.speculative_steps for item in metrics),
        ),
        timing=TimingMetrics(
            throughput_tps=statistics.mean(item.timing.throughput_tps for item in metrics),
            ttft_ms=statistics.mean(ttft_values) if ttft_values else float("nan"),
            total_generated_tokens=sum(item.timing.total_generated_tokens for item in metrics),
            wall_time_s=sum(item.timing.wall_time_s for item in metrics),
        ),
        instrumentation_backend=",".join(backends),
        raw_decisions=[decision for item in metrics for decision in item.raw_decisions],
    )


def hypothesis_recommendation(
    acceptance_delta_ci95: tuple[float, float, float],
    throughput_relative_delta_ci95: tuple[float, float, float],
) -> str:
    if acceptance_delta_ci95[2] <= -0.05 or throughput_relative_delta_ci95[2] <= -0.10:
        return "go"
    return "no-go"


def run_validation(
    experiment: ExperimentConfig,
    adapter_path: str,
    prompts: list[str],
    prompt_metadata: dict[str, Any],
    artifact_provenance: dict[str, Any],
    target_load_path: str,
    draft_load_path: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    set_seed(experiment.seed)
    llm = initialize_vllm(
        target_model=target_load_path,
        draft_model=draft_load_path,
        tensor_parallel_size=experiment.model_pair.tensor_parallel_degree,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
        speculation_length=experiment.speculation_length,
        enable_lora=adapter_path is not None,
        trust_remote_code=experiment.trust_remote_code,
    )
    sampling_params = build_sampling_params(max_tokens=experiment.max_tokens)
    warmup_sampling_params = build_sampling_params(max_tokens=experiment.warmup_tokens)
    run_config = {
        "experiment": experiment.model_dump(mode="json"),
        "prompt_metadata": prompt_metadata,
        "artifact_provenance": artifact_provenance,
    }
    config_hash = compute_config_hash(run_config)

    warmup_prompts = prompts[: min(experiment.warmup_prompts, len(prompts))]
    logger.info("Warming baseline and adapted conditions on %d prompts", len(warmup_prompts))
    run_speculative_generation(
        llm=llm,
        prompts=warmup_prompts,
        sampling_params=warmup_sampling_params,
        speculation_length=experiment.speculation_length,
    )
    run_speculative_generation(
        llm=llm,
        prompts=warmup_prompts,
        sampling_params=warmup_sampling_params,
        speculation_length=experiment.speculation_length,
        adapter_path=adapter_path,
    )

    measured_runs: dict[str, list[Any]] = {"baseline": [], "adapted": []}
    condition_orders: list[list[str]] = []
    for repetition in range(experiment.measurement_repetitions):
        condition_order = ["baseline", "adapted"]
        random.Random(experiment.seed + repetition).shuffle(condition_order)
        condition_orders.append(condition_order)
        for condition in condition_order:
            condition_adapter = adapter_path if condition == "adapted" else None
            logger.info(
                "Running repetition %d/%d, %s condition",
                repetition + 1,
                experiment.measurement_repetitions,
                condition,
            )
            measured_runs[condition].append(
                run_speculative_generation(
                    llm=llm,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    speculation_length=experiment.speculation_length,
                    adapter_path=condition_adapter,
                ),
            )
    baseline_metrics = _aggregate_metrics(measured_runs["baseline"])
    adapted_metrics = _aggregate_metrics(measured_runs["adapted"])

    acceptance_delta = (
        adapted_metrics.acceptance.overall_acceptance_rate
        - baseline_metrics.acceptance.overall_acceptance_rate
    )
    throughput_delta = adapted_metrics.timing.throughput_tps - baseline_metrics.timing.throughput_tps
    acceptance_deltas = [
        adapted_run.metrics.acceptance.overall_acceptance_rate
        - baseline_run.metrics.acceptance.overall_acceptance_rate
        for baseline_run, adapted_run in zip(measured_runs["baseline"], measured_runs["adapted"])
    ]
    throughput_relative_deltas = [
        (adapted_run.metrics.timing.throughput_tps - baseline_run.metrics.timing.throughput_tps)
        / max(baseline_run.metrics.timing.throughput_tps, 1e-12)
        for baseline_run, adapted_run in zip(measured_runs["baseline"], measured_runs["adapted"])
    ]
    acceptance_delta_ci95 = _mean_ci95(acceptance_deltas)
    throughput_relative_delta_ci95 = _mean_ci95(throughput_relative_deltas)
    recommendation = hypothesis_recommendation(
        acceptance_delta_ci95,
        throughput_relative_delta_ci95,
    )
    logger.info(
        "Baseline acceptance %.4f | Adapted acceptance %.4f | Delta %.4f",
        baseline_metrics.acceptance.overall_acceptance_rate,
        adapted_metrics.acceptance.overall_acceptance_rate,
        acceptance_delta,
    )
    logger.info(
        "Baseline throughput %.2f tok/s | Adapted throughput %.2f tok/s | Delta %.2f tok/s",
        baseline_metrics.timing.throughput_tps,
        adapted_metrics.timing.throughput_tps,
        throughput_delta,
    )
    logger.info("Phase 1 recommendation: %s", recommendation)

    summary = {
        "baseline": make_result_record(
            baseline_metrics,
            config_hash=config_hash,
            metadata={
                "condition": "baseline",
                "model_pair": experiment.model_pair.model_dump(),
                "dataset": experiment.dataset,
                "measurement_repetitions": experiment.measurement_repetitions,
            },
        ).model_dump(mode="json"),
        "adapted": make_result_record(
            adapted_metrics,
            config_hash=config_hash,
            metadata={
                "condition": "adapted",
                "adapter_path": adapter_path,
                "dataset": experiment.dataset,
                "measurement_repetitions": experiment.measurement_repetitions,
            },
        ).model_dump(mode="json"),
        "comparison": {
            "acceptance_delta": acceptance_delta,
            "throughput_delta_tps": throughput_delta,
            "recommendation": recommendation,
            "recommendation_rule": {
                "acceptance_delta_upper_ci_threshold": -0.05,
                "throughput_relative_delta_upper_ci_threshold": -0.10,
                "logic": "go if either paired 95% CI upper bound is at or below its threshold",
            },
            "measured_condition_orders": condition_orders,
            "acceptance_delta_ci95": {
                "mean": acceptance_delta_ci95[0],
                "lower": acceptance_delta_ci95[1],
                "upper": acceptance_delta_ci95[2],
            },
            "throughput_relative_delta_ci95": {
                "mean": throughput_relative_delta_ci95[0],
                "lower": throughput_relative_delta_ci95[1],
                "upper": throughput_relative_delta_ci95[2],
            },
            "per_repetition": [
                {
                    "repetition": index,
                    "condition_order": condition_orders[index],
                    "baseline_acceptance": measured_runs["baseline"][
                        index
                    ].metrics.acceptance.overall_acceptance_rate,
                    "adapted_acceptance": measured_runs["adapted"][
                        index
                    ].metrics.acceptance.overall_acceptance_rate,
                    "acceptance_delta": acceptance_deltas[index],
                    "baseline_throughput_tps": measured_runs["baseline"][
                        index
                    ].metrics.timing.throughput_tps,
                    "adapted_throughput_tps": measured_runs["adapted"][
                        index
                    ].metrics.timing.throughput_tps,
                    "throughput_relative_delta": throughput_relative_deltas[index],
                    "baseline_ttft_ms": measured_runs["baseline"][
                        index
                    ].metrics.timing.ttft_ms,
                    "adapted_ttft_ms": measured_runs["adapted"][index].metrics.timing.ttft_ms,
                }
                for index in range(experiment.measurement_repetitions)
            ],
            "warmup_prompts": len(warmup_prompts),
            "warmup_tokens": experiment.warmup_tokens,
        },
    }
    return {
        "summary": summary,
        "prompts": prompts,
        "prompt_metadata": prompt_metadata,
        "experiment": run_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the LoRA-Spec core hypothesis.")
    add_common_args(parser)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--target-revision", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--draft-revision", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, required=False)
    parser.add_argument("--adapter-revision", type=str, default=None)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-domain", type=str, default=None)
    parser.add_argument("--adapter-epochs", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="lora-spec-pilot-v1/evaluation")
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/evaluation.jsonl",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--measurement-repetitions", type=int, default=3)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--tensor-parallel-degree", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-prompts", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/phase1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, logger_name="validate_hypothesis")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)
    target_model = get_config_value(config_data, args, "target_model")
    draft_model = get_config_value(config_data, args, "draft_model")
    adapter_path = get_config_value(config_data, args, "adapter_path")
    if not target_model or not draft_model or not adapter_path:
        raise ValueError("target_model, draft_model, and adapter_path must be provided")
    target_artifact = resolve_artifact_revision(
        str(target_model),
        revision=get_config_value(config_data, args, "target_revision"),
    )
    draft_artifact = resolve_artifact_revision(
        str(draft_model),
        revision=get_config_value(config_data, args, "draft_revision"),
    )
    adapter_artifact = resolve_artifact_revision(
        str(adapter_path),
        revision=get_config_value(config_data, args, "adapter_revision"),
    )
    resolved_config = dict(config_data)
    resolved_config.update(
        {
            "target_revision": target_artifact.resolved_revision,
            "draft_revision": draft_artifact.resolved_revision,
            "adapter_revision": adapter_artifact.resolved_revision,
        },
    )
    experiment = build_experiment_config(args, resolved_config)

    prompts_file = str(get_config_value(resolved_config, args, "prompts_file"))
    registered = resolve_registered_prompt_split(prompts_file, expected_split="evaluation")
    if experiment.num_prompts > len(registered.records):
        raise ValueError(
            f"Requested {experiment.num_prompts} prompts, but frozen evaluation split has "
            f"{len(registered.records)}",
        )
    selected_indices = list(range(len(registered.records)))
    random.Random(experiment.seed).shuffle(selected_indices)
    selected_records = [registered.records[index] for index in selected_indices[: experiment.num_prompts]]
    prompts = [record.text for record in selected_records]
    prompt_metadata = {
        **prompt_file_provenance(prompts_file, expected_split="evaluation"),
        "selected_prompt_ids": [record.id for record in selected_records],
        "selection_seed": experiment.seed,
    }
    artifact_provenance = {
        "target_model": target_artifact.to_dict(),
        "draft_model": draft_artifact.to_dict(),
        "adapter": adapter_artifact.to_dict(),
    }
    payload = run_validation(
        experiment,
        adapter_path=materialize_artifact(adapter_artifact),
        prompts=prompts,
        prompt_metadata=prompt_metadata,
        artifact_provenance=artifact_provenance,
        target_load_path=materialize_artifact(target_artifact),
        draft_load_path=materialize_artifact(draft_artifact),
        logger=logger,
    )
    output_path = write_json_result(
        payload={
            **payload["summary"],
            "prompts": payload["prompts"],
            "prompt_metadata": payload["prompt_metadata"],
            "artifact_provenance": artifact_provenance,
        },
        output_dir=str(get_config_value(config_data, args, "output_dir")),
        stem="phase1_validation",
        config=payload["experiment"],
        extra_metadata={"adapter_path": adapter_path},
        cwd=Path.cwd(),
    )
    logger.info("Saved Phase 1 results to %s", output_path)


if __name__ == "__main__":
    main()
