from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lora_spec.serving import (
    TrafficPatternConfig,
    build_sampling_params,
    build_traffic_requests,
    create_local_request_executor,
    create_openai_server_request_executor,
    initialize_vllm,
    load_prompts,
    run_concurrent_benchmark,
)
from lora_spec.utils import add_common_args, get_config_value, load_yaml, resolve_config, set_seed, setup_logging, write_json_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-tenant serving benchmarks for LoRA-Spec.")
    add_common_args(parser)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--adapter-path", action="append", default=[])
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--tensor-parallel-degree", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
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
    parser.add_argument("--server-api-key", type=str, default="EMPTY")
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


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "benchmark_serving")
    set_seed(args.seed)
    config_data = resolve_config(args.config, args.override)
    serving_config = _serving_section(str(get_config_value(config_data, args, "serving_config")))
    pattern_name = str(get_config_value(config_data, args, "traffic_pattern"))
    pattern_defaults = _pattern_defaults(serving_config, pattern_name)

    target_model = get_config_value(config_data, args, "target_model")
    draft_model = get_config_value(config_data, args, "draft_model")
    if not target_model or not draft_model:
        raise ValueError("target_model and draft_model must be provided")
    adapter_paths = _resolve_adapter_paths(config_data, args)

    num_prompts = int(get_config_value(config_data, args, "num_prompts"))
    dataset = str(get_config_value(config_data, args, "dataset"))
    prompts = load_prompts(dataset, num_prompts=num_prompts, seed=args.seed)

    traffic_config = TrafficPatternConfig(
        pattern=pattern_name,
        concurrency=int(get_config_value(config_data, args, "concurrency", serving_config.get("concurrency", 4))),
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
                pattern_defaults.get("request_spacing_s", serving_config.get("request_spacing_s", 0.02)),
            )
        ),
        seed=args.seed,
    )

    baseline_requests = build_traffic_requests(prompts, [None], traffic_config)
    adapted_requests = build_traffic_requests(prompts, adapter_paths or [None], traffic_config)

    server_url = get_config_value(config_data, args, "server_url")
    if server_url:
        logger.info("Running HTTP benchmark against %s", server_url)
        baseline_executor = create_openai_server_request_executor(
            server_url=str(server_url),
            model=str(target_model),
            max_tokens=int(get_config_value(config_data, args, "max_tokens", serving_config.get("max_tokens", 128))),
            temperature=float(
                get_config_value(config_data, args, "temperature", serving_config.get("temperature", 0.0)),
            ),
            top_p=float(get_config_value(config_data, args, "top_p", serving_config.get("top_p", 1.0))),
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
            max_tokens=int(get_config_value(config_data, args, "max_tokens", serving_config.get("max_tokens", 128))),
            temperature=float(
                get_config_value(config_data, args, "temperature", serving_config.get("temperature", 0.0)),
            ),
            top_p=float(get_config_value(config_data, args, "top_p", serving_config.get("top_p", 1.0))),
        )
        llm = initialize_vllm(
            target_model=str(target_model),
            draft_model=str(draft_model),
            tensor_parallel_size=int(
                get_config_value(config_data, args, "tensor_parallel_degree"),
            ),
            max_model_len=int(
                get_config_value(config_data, args, "max_model_len", serving_config.get("max_model_len", 4096)),
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
                get_config_value(config_data, args, "speculation_length", serving_config.get("speculation_length", 4)),
            ),
            enable_lora=bool(adapter_paths),
            trust_remote_code=bool(
                get_config_value(config_data, args, "trust_remote_code", serving_config.get("trust_remote_code", False)),
            ),
        )
        baseline_executor = create_local_request_executor(llm=llm, sampling_params=sampling_params)
        adapted_executor = baseline_executor

    logger.info(
        "Running baseline benchmark | pattern=%s | concurrency=%d | requests=%d",
        traffic_config.pattern,
        traffic_config.concurrency,
        len(baseline_requests),
    )
    baseline_result = run_concurrent_benchmark(
        requests=baseline_requests,
        executor=baseline_executor,
        concurrency=traffic_config.concurrency,
    )

    logger.info(
        "Running adapted benchmark | tenants=%d | requests=%d",
        max(len(adapter_paths), 1),
        len(adapted_requests),
    )
    adapted_result = run_concurrent_benchmark(
        requests=adapted_requests,
        executor=adapted_executor,
        concurrency=traffic_config.concurrency,
    )

    throughput_delta = adapted_result.throughput_tps - baseline_result.throughput_tps
    latency_delta = adapted_result.p95_latency_ms - baseline_result.p95_latency_ms

    logger.info(
        "Baseline TPS %.2f | Adapted TPS %.2f | Delta %.2f",
        baseline_result.throughput_tps,
        adapted_result.throughput_tps,
        throughput_delta,
    )
    logger.info(
        "Baseline p95 %.2f ms | Adapted p95 %.2f ms | Delta %.2f ms",
        baseline_result.p95_latency_ms,
        adapted_result.p95_latency_ms,
        latency_delta,
    )

    output = write_json_result(
        payload={
            "baseline": asdict(baseline_result),
            "adapted": asdict(adapted_result),
            "comparison": {
                "throughput_delta_tps": throughput_delta,
                "p95_latency_delta_ms": latency_delta,
            },
        },
        output_dir=str(get_config_value(config_data, args, "output_dir")),
        stem="serving_benchmark",
        config={
            "target_model": str(target_model),
            "draft_model": str(draft_model),
            "adapter_paths": adapter_paths,
            "dataset": dataset,
            "num_prompts": num_prompts,
            "server_url": server_url,
            "traffic_config": asdict(traffic_config),
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved serving benchmark to %s", output)


if __name__ == "__main__":
    main()
