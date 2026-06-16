from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lora_spec.artifacts import materialize_artifact, resolve_artifact_revision
from lora_spec.prompts import select_frozen_prompts
from lora_spec.serving import (
    TrafficPatternConfig,
    build_sampling_params,
    build_traffic_requests,
    create_local_request_executor,
    create_openai_server_request_executor,
    initialize_vllm,
    load_and_verify_server_provenance,
    run_concurrent_benchmark,
    warmup_subset,
)
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    load_yaml,
    mean_ci95,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-tenant serving benchmarks for LoRA-Spec."
    )
    add_common_args(parser)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--target-revision", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--draft-revision", type=str, default=None)
    parser.add_argument("--adapter-path", action="append", default=[])
    parser.add_argument("--adapter-revision", action="append", default=[])
    parser.add_argument("--adapter-model-name", action="append", default=[])
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/prompts/pilot_v1/evaluation.jsonl",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--measurement-repetitions", type=int, default=3)
    parser.add_argument("--tensor-parallel-degree", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--traffic-pattern", type=str, default="uniform")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-tenant", type=int, default=8)
    parser.add_argument("--hot-tenant-fraction", type=float, default=0.2)
    parser.add_argument("--hot-request-fraction", type=float, default=0.8)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument("--burst-gap-s", type=float, default=0.25)
    parser.add_argument("--request-spacing-s", type=float, default=0.02)
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--server-base-model-name", type=str, default=None)
    parser.add_argument("--server-api-key", type=str, default="EMPTY")
    parser.add_argument("--server-provenance-json", type=str, default=None)
    parser.add_argument("--warmup-requests-per-tenant", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--serving-config", type=str, default="configs/serving.yaml")
    parser.add_argument("--output-dir", type=str, default="results/serving_bench")
    return parser.parse_args()


def _serving_section(config_path: str) -> dict[str, Any]:
    payload = load_yaml(config_path)
    serving_config = payload.get("serving", {})
    if not isinstance(serving_config, dict):
        raise ValueError(f"{config_path} must contain a top-level 'serving' mapping")
    return serving_config


def _pattern_defaults(serving_config: dict[str, Any], pattern: str) -> dict[str, Any]:
    pattern_config = serving_config.get("traffic_patterns", {}).get(pattern, {})
    if not isinstance(pattern_config, dict):
        raise ValueError(f"serving.traffic_patterns.{pattern} must be a mapping")
    return pattern_config


def _resolve_adapter_paths(config_data: dict[str, Any], args: argparse.Namespace) -> list[str]:
    value = get_config_value(config_data, args, "adapter_path", [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    raise ValueError("adapter_path must be a string or list of strings")


def _resolve_adapter_revisions(
    config_data: dict[str, Any],
    args: argparse.Namespace,
    adapter_count: int,
) -> list[str | None]:
    value = get_config_value(config_data, args, "adapter_revision", [])
    if value is None:
        revisions: list[str | None] = []
    elif isinstance(value, str):
        revisions = [value]
    elif isinstance(value, list):
        revisions = [str(item) if item is not None else None for item in value]
    else:
        raise ValueError("adapter_revision must be a string or list of strings")
    if len(revisions) > adapter_count:
        raise ValueError("More adapter revisions were provided than adapter paths")
    return revisions + [None] * (adapter_count - len(revisions))


def _resolve_adapter_model_names(
    config_data: dict[str, Any],
    args: argparse.Namespace,
    adapter_count: int,
) -> list[str]:
    value = get_config_value(config_data, args, "adapter_model_name", [])
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, list):
        names = [str(item) for item in value if item]
    else:
        raise ValueError("adapter_model_name must be a string or list of strings")
    if names and len(names) != adapter_count:
        raise ValueError("Provide exactly one adapter_model_name per adapter_path")
    if len(names) != len(set(names)):
        raise ValueError("adapter_model_name values must be unique")
    return names


def _summarize_benchmark_runs(runs: list[Any]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one benchmark run is required")
    return {
        "pattern": runs[0].pattern,
        "concurrency": runs[0].concurrency,
        "total_requests_per_repetition": runs[0].total_requests,
        "throughput_rps": sum(run.throughput_rps for run in runs) / len(runs),
        "throughput_tps": sum(run.throughput_tps for run in runs) / len(runs),
        "p50_latency_ms": sum(run.p50_latency_ms for run in runs) / len(runs),
        "p95_latency_ms": sum(run.p95_latency_ms for run in runs) / len(runs),
        "completed_requests": sum(run.completed_requests for run in runs),
        "errored_requests": sum(run.errored_requests for run in runs),
        "measurement_repetitions": len(runs),
        "latency_percentile_method": "Hyndman-Fan type 7 (linear interpolation)",
        "per_repetition": [asdict(run) for run in runs],
    }


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "benchmark_serving")
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)
    serving_config = _serving_section(str(get_config_value(config_data, args, "serving_config")))
    pattern_name = str(get_config_value(config_data, args, "traffic_pattern"))
    pattern_defaults = _pattern_defaults(serving_config, pattern_name)

    target_model = get_config_value(config_data, args, "target_model")
    draft_model = get_config_value(config_data, args, "draft_model")
    if not target_model or not draft_model:
        raise ValueError("target_model and draft_model must be provided")
    adapter_paths = _resolve_adapter_paths(config_data, args)
    adapter_revisions = _resolve_adapter_revisions(config_data, args, len(adapter_paths))
    adapter_model_names = _resolve_adapter_model_names(
        config_data,
        args,
        len(adapter_paths),
    )

    num_prompts = int(get_config_value(config_data, args, "num_prompts"))
    measurement_repetitions = int(get_config_value(config_data, args, "measurement_repetitions"))
    if measurement_repetitions < 2:
        raise ValueError("measurement_repetitions must be at least 2")
    prompts_file = str(get_config_value(config_data, args, "prompts_file"))
    prompts, prompts_provenance = select_frozen_prompts(
        prompts_file,
        expected_split="evaluation",
        num_prompts=num_prompts,
        seed=seed,
    )
    target_artifact = resolve_artifact_revision(
        str(target_model),
        revision=get_config_value(config_data, args, "target_revision"),
    )
    draft_artifact = resolve_artifact_revision(
        str(draft_model),
        revision=get_config_value(config_data, args, "draft_revision"),
    )
    adapter_artifacts = [
        resolve_artifact_revision(path, revision=revision)
        for path, revision in zip(adapter_paths, adapter_revisions)
    ]
    server_url = get_config_value(config_data, args, "server_url")
    if server_url and adapter_paths and not adapter_model_names:
        raise ValueError(
            "HTTP vLLM benchmarks require --adapter-model-name for every adapter. "
            "Start the server with matching --lora-modules name=path entries."
        )
    materialized_adapters = (
        [] if server_url else [materialize_artifact(item) for item in adapter_artifacts]
    )
    expected_server_artifacts = {
        "target_model": target_artifact.to_dict(),
        "draft_model": draft_artifact.to_dict(),
        "adapters": [item.to_dict() for item in adapter_artifacts],
    }
    server_provenance = None
    server_provenance_sha256 = None
    if server_url:
        server_provenance_path = get_config_value(config_data, args, "server_provenance_json")
        if not server_provenance_path:
            raise ValueError("HTTP benchmarks require --server-provenance-json")
        server_provenance, server_provenance_sha256 = load_and_verify_server_provenance(
            str(server_provenance_path),
            expected_server_artifacts,
            adapter_model_names,
        )

    traffic_config = TrafficPatternConfig(
        pattern=pattern_name,
        concurrency=int(
            get_config_value(config_data, args, "concurrency", serving_config.get("concurrency", 4))
        ),
        requests_per_tenant=int(
            get_config_value(
                config_data,
                args,
                "requests_per_tenant",
                serving_config.get("requests_per_tenant", 8),
            )
        ),
        hot_tenant_fraction=float(
            get_config_value(
                config_data,
                args,
                "hot_tenant_fraction",
                pattern_defaults.get("hot_tenant_fraction", 0.2),
            )
        ),
        hot_request_fraction=float(
            get_config_value(
                config_data,
                args,
                "hot_request_fraction",
                pattern_defaults.get("hot_request_fraction", 0.8),
            )
        ),
        burst_size=int(
            get_config_value(
                config_data,
                args,
                "burst_size",
                pattern_defaults.get("burst_size", 4),
            )
        ),
        burst_gap_s=float(
            get_config_value(
                config_data,
                args,
                "burst_gap_s",
                pattern_defaults.get("burst_gap_s", 0.25),
            )
        ),
        request_spacing_s=float(
            get_config_value(
                config_data,
                args,
                "request_spacing_s",
                pattern_defaults.get(
                    "request_spacing_s", serving_config.get("request_spacing_s", 0.02)
                ),
            )
        ),
        seed=seed,
    )

    tenant_count = max(len(adapter_paths), 1)
    baseline_requests = build_traffic_requests(prompts, [None] * tenant_count, traffic_config)
    adapted_requests = build_traffic_requests(
        prompts,
        materialized_adapters if not server_url else [None] * tenant_count,
        traffic_config,
        adapter_model_names=(adapter_model_names or [None] * tenant_count),
    )

    if server_url:
        logger.info("Running HTTP benchmark against %s", server_url)
        baseline_executor = create_openai_server_request_executor(
            server_url=str(server_url),
            model=str(
                get_config_value(config_data, args, "server_base_model_name") or target_model
            ),
            max_tokens=int(
                get_config_value(
                    config_data, args, "max_tokens", serving_config.get("max_tokens", 128)
                )
            ),
            temperature=float(
                get_config_value(
                    config_data, args, "temperature", serving_config.get("temperature", 0.0)
                ),
            ),
            top_p=float(
                get_config_value(config_data, args, "top_p", serving_config.get("top_p", 1.0))
            ),
            api_key=str(get_config_value(config_data, args, "server_api_key", "EMPTY")),
        )
        adapted_executor = baseline_executor
    else:
        logger.info("Initializing in-process vLLM benchmark engine")
        if traffic_config.concurrency > 1:
            logger.warning(
                "In-process vLLM benchmarking serializes requests inside one engine instance; "
                "forcing concurrency=1 for non-misleading local measurements. Use --server-url for real concurrent load.",
            )
            traffic_config.concurrency = 1
        sampling_params = build_sampling_params(
            max_tokens=int(
                get_config_value(
                    config_data, args, "max_tokens", serving_config.get("max_tokens", 128)
                )
            ),
            temperature=float(
                get_config_value(
                    config_data, args, "temperature", serving_config.get("temperature", 0.0)
                ),
            ),
            top_p=float(
                get_config_value(config_data, args, "top_p", serving_config.get("top_p", 1.0))
            ),
        )
        llm = initialize_vllm(
            target_model=materialize_artifact(target_artifact),
            draft_model=materialize_artifact(draft_artifact),
            tensor_parallel_size=int(
                get_config_value(config_data, args, "tensor_parallel_degree"),
            ),
            max_model_len=int(
                get_config_value(
                    config_data, args, "max_model_len", serving_config.get("max_model_len", 4096)
                ),
            ),
            gpu_memory_utilization=float(
                get_config_value(
                    config_data,
                    args,
                    "gpu_memory_utilization",
                    serving_config.get("gpu_memory_utilization", 0.85),
                )
            ),
            speculation_length=int(
                get_config_value(
                    config_data,
                    args,
                    "speculation_length",
                    serving_config.get("speculation_length", 4),
                ),
            ),
            enable_lora=bool(adapter_paths),
            max_lora_rank=int(get_config_value(config_data, args, "max_lora_rank")),
            trust_remote_code=bool(
                get_config_value(
                    config_data,
                    args,
                    "trust_remote_code",
                    serving_config.get("trust_remote_code", False),
                ),
            ),
        )
        baseline_executor = create_local_request_executor(llm=llm, sampling_params=sampling_params)
        adapted_executor = baseline_executor

    measured_runs: dict[str, list[Any]] = {"baseline": [], "adapted": []}
    warmup_requests_per_tenant = int(
        get_config_value(config_data, args, "warmup_requests_per_tenant")
    )
    for condition, requests, executor in (
        ("baseline", baseline_requests, baseline_executor),
        ("adapted", adapted_requests, adapted_executor),
    ):
        warmup_requests = warmup_subset(requests, warmup_requests_per_tenant)
        if warmup_requests:
            logger.info("Warming %s condition with %d requests", condition, len(warmup_requests))
            run_concurrent_benchmark(
                warmup_requests,
                executor,
                concurrency=min(traffic_config.concurrency, len(warmup_requests)),
                fail_on_error=True,
            )
    condition_orders: list[list[str]] = []
    for repetition in range(measurement_repetitions):
        order = ["baseline", "adapted"]
        random.Random(seed + repetition).shuffle(order)
        condition_orders.append(order)
        for condition in order:
            logger.info(
                "Running repetition %d/%d, %s serving condition",
                repetition + 1,
                measurement_repetitions,
                condition,
            )
            measured_runs[condition].append(
                run_concurrent_benchmark(
                    requests=(baseline_requests if condition == "baseline" else adapted_requests),
                    executor=(baseline_executor if condition == "baseline" else adapted_executor),
                    concurrency=traffic_config.concurrency,
                    fail_on_error=True,
                )
            )

    baseline_summary = _summarize_benchmark_runs(measured_runs["baseline"])
    adapted_summary = _summarize_benchmark_runs(measured_runs["adapted"])
    throughput_deltas = [
        adapted.throughput_tps - baseline.throughput_tps
        for baseline, adapted in zip(measured_runs["baseline"], measured_runs["adapted"])
    ]
    latency_deltas = [
        adapted.p95_latency_ms - baseline.p95_latency_ms
        for baseline, adapted in zip(measured_runs["baseline"], measured_runs["adapted"])
    ]
    throughput_delta_ci95 = mean_ci95(throughput_deltas)
    latency_delta_ci95 = mean_ci95(latency_deltas)
    throughput_delta = throughput_delta_ci95[0]
    latency_delta = latency_delta_ci95[0]

    logger.info(
        "Baseline TPS %.2f | Adapted TPS %.2f | Delta %.2f",
        baseline_summary["throughput_tps"],
        adapted_summary["throughput_tps"],
        throughput_delta,
    )
    logger.info(
        "Baseline p95 %.2f ms | Adapted p95 %.2f ms | Delta %.2f ms",
        baseline_summary["p95_latency_ms"],
        adapted_summary["p95_latency_ms"],
        latency_delta,
    )

    output = write_json_result(
        payload={
            "baseline": baseline_summary,
            "adapted": adapted_summary,
            "comparison": {
                "throughput_delta_tps": throughput_delta,
                "p95_latency_delta_ms": latency_delta,
                "throughput_delta_ci95": {
                    "mean": throughput_delta_ci95[0],
                    "lower": throughput_delta_ci95[1],
                    "upper": throughput_delta_ci95[2],
                },
                "p95_latency_delta_ci95": {
                    "mean": latency_delta_ci95[0],
                    "lower": latency_delta_ci95[1],
                    "upper": latency_delta_ci95[2],
                },
                "condition_orders": condition_orders,
            },
        },
        output_dir=str(get_config_value(config_data, args, "output_dir")),
        stem="serving_benchmark",
        config={
            "target_model": str(target_model),
            "draft_model": str(draft_model),
            "adapter_paths": adapter_paths,
            "adapter_model_names": adapter_model_names,
            "prompts_file": prompts_file,
            "prompts_provenance": prompts_provenance,
            "num_prompts": num_prompts,
            "measurement_repetitions": measurement_repetitions,
            "execution_mode": "http_concurrent" if server_url else "in_process_serial",
            "latency_percentile_method": "Hyndman-Fan type 7 (linear interpolation)",
            "server_url": server_url,
            "server_base_model_name": get_config_value(
                config_data,
                args,
                "server_base_model_name",
            ),
            "external_server_artifacts_verified": bool(server_provenance) if server_url else True,
            "server_provenance": server_provenance,
            "server_provenance_sha256": server_provenance_sha256,
            "artifact_provenance": expected_server_artifacts,
            "warmup_requests_per_tenant": warmup_requests_per_tenant,
            "seed": seed,
            "traffic_config": asdict(traffic_config),
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved serving benchmark to %s", output)


if __name__ == "__main__":
    main()
