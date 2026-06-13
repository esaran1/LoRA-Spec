from __future__ import annotations

import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from datasets import load_dataset

from .metrics import SpeculativeDecodingMetrics, collect_speculative_metrics


@dataclass
class ServingRunResult:
    prompts: list[str]
    texts: list[str]
    metrics: SpeculativeDecodingMetrics


@dataclass
class TrafficRequest:
    request_id: str
    tenant_id: str
    prompt: str
    adapter_path: str | None = None
    arrival_offset_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrafficPatternConfig:
    pattern: str
    concurrency: int
    requests_per_tenant: int
    hot_tenant_fraction: float = 0.2
    hot_request_fraction: float = 0.8
    burst_size: int = 4
    burst_gap_s: float = 0.25
    request_spacing_s: float = 0.02
    seed: int = 7


@dataclass
class RequestBenchmarkResult:
    request_id: str
    tenant_id: str
    adapter_path: str | None
    latency_ms: float
    output_tokens: int
    text: str
    error: str | None = None


@dataclass
class TrafficBenchmarkResult:
    pattern: str
    concurrency: int
    total_requests: int
    completed_requests: int
    errored_requests: int
    throughput_rps: float
    throughput_tps: float
    p50_latency_ms: float
    p95_latency_ms: float
    wall_time_s: float
    per_tenant_request_counts: dict[str, int]
    request_results: list[RequestBenchmarkResult]


def _infer_text_field(sample: dict[str, Any]) -> str:
    for field_name in ("text", "prompt", "instruction", "question", "content"):
        if field_name in sample and isinstance(sample[field_name], str) and sample[field_name].strip():
            return field_name
    if {"instruction", "input"} <= set(sample):
        return "instruction"
    raise ValueError("Could not infer a text field from the dataset sample")


def load_prompts(
    dataset_name: str,
    num_prompts: int,
    seed: int,
    split: str = "train",
    text_field: str | None = None,
) -> list[str]:
    dataset = load_dataset(dataset_name, split=split)
    if len(dataset) < num_prompts:
        raise ValueError(f"Dataset split {dataset_name}:{split} contains fewer than {num_prompts} rows")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = dataset.select(indices[:num_prompts])
    sample = selected[0]
    field = text_field or _infer_text_field(sample)
    prompts: list[str] = []
    for row in selected:
        if field == "instruction" and isinstance(row.get("input"), str) and row.get("input"):
            prompt = f"{row['instruction']}\n\n{row['input']}"
        else:
            prompt = row[field]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompts.append(prompt.strip())
    if len(prompts) < num_prompts:
        raise ValueError("Not enough valid prompts were extracted from the dataset")
    return prompts[:num_prompts]


def initialize_vllm(
    target_model: str,
    draft_model: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    speculation_length: int = 4,
    enable_lora: bool = False,
    trust_remote_code: bool = False,
):
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - requires vLLM runtime
        raise ImportError("vLLM must be installed to initialize serving") from exc

    return LLM(
        model=target_model,
        speculative_model=draft_model,
        num_speculative_tokens=speculation_length,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=enable_lora,
        trust_remote_code=trust_remote_code,
    )


def build_sampling_params(
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    try:
        from vllm import SamplingParams
    except ImportError as exc:  # pragma: no cover - requires vLLM runtime
        raise ImportError("vLLM must be installed to build sampling params") from exc
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def run_speculative_generation(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
    speculation_length: int,
    adapter_path: str | None = None,
) -> ServingRunResult:
    lora_request = None
    if adapter_path is not None:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:  # pragma: no cover - requires vLLM runtime
            raise ImportError("vLLM LoRA support is required when adapter_path is provided") from exc
        lora_request = LoRARequest("lora-spec-adapter", 1, adapter_path)

    def _generate() -> list[Any]:
        return llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

    outputs, metrics = collect_speculative_metrics(_generate, speculation_length=speculation_length)
    texts = [output.outputs[0].text if output.outputs else "" for output in outputs]
    return ServingRunResult(
        prompts=prompts,
        texts=texts,
        metrics=metrics,
    )


def _cycle_values(values: Sequence[str], total: int) -> list[str]:
    if not values:
        raise ValueError("At least one value is required")
    return [values[index % len(values)] for index in range(total)]


def build_traffic_requests(
    prompts: Sequence[str],
    adapter_paths: Sequence[str | None],
    config: TrafficPatternConfig,
) -> list[TrafficRequest]:
    if not prompts:
        raise ValueError("prompts must not be empty")
    tenant_adapters = list(adapter_paths) or [None]
    tenant_ids = [f"tenant_{index:02d}" for index in range(len(tenant_adapters))]
    total_requests = config.requests_per_tenant * len(tenant_ids)
    prompt_sequence = _cycle_values(list(prompts), total_requests)
    rng = random.Random(config.seed)

    if config.pattern == "uniform":
        tenant_sequence = _cycle_values(tenant_ids, total_requests)
        arrival_offsets = [index * config.request_spacing_s for index in range(total_requests)]
    elif config.pattern == "skewed_80_20":
        hot_tenant_count = max(1, int(round(len(tenant_ids) * config.hot_tenant_fraction)))
        hot_tenant_ids = tenant_ids[:hot_tenant_count]
        cold_tenant_ids = tenant_ids[hot_tenant_count:] or hot_tenant_ids
        hot_request_count = int(round(total_requests * config.hot_request_fraction))
        cold_request_count = total_requests - hot_request_count
        tenant_sequence = _cycle_values(hot_tenant_ids, hot_request_count) + _cycle_values(
            cold_tenant_ids,
            cold_request_count,
        )
        rng.shuffle(tenant_sequence)
        arrival_offsets = [index * config.request_spacing_s for index in range(total_requests)]
    elif config.pattern == "bursty":
        tenant_sequence = _cycle_values(tenant_ids, total_requests)
        arrival_offsets = []
        for index in range(total_requests):
            burst_index = index // max(config.burst_size, 1)
            within_burst = (index % max(config.burst_size, 1)) * min(config.request_spacing_s, 0.005)
            arrival_offsets.append((burst_index * config.burst_gap_s) + within_burst)
    else:
        raise ValueError(f"Unsupported traffic pattern: {config.pattern}")

    requests: list[TrafficRequest] = []
    adapter_by_tenant = dict(zip(tenant_ids, tenant_adapters))
    for index in range(total_requests):
        tenant_id = tenant_sequence[index]
        requests.append(
            TrafficRequest(
                request_id=f"{config.pattern}-{index:05d}",
                tenant_id=tenant_id,
                prompt=prompt_sequence[index],
                adapter_path=adapter_by_tenant[tenant_id],
                arrival_offset_s=arrival_offsets[index],
                metadata={"pattern": config.pattern},
            )
        )
    return requests


def create_local_request_executor(
    llm: Any,
    sampling_params: Any,
) -> Callable[[TrafficRequest], tuple[str, int]]:
    llm_lock = threading.Lock()

    def execute(request: TrafficRequest) -> tuple[str, int]:
        lora_request = None
        if request.adapter_path is not None:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:  # pragma: no cover - requires vLLM runtime
                raise ImportError("vLLM LoRA support is required when adapter_path is provided") from exc
            lora_request = LoRARequest(
                f"tenant-{request.tenant_id}",
                abs(hash(request.tenant_id)) % 1_000_000 + 1,
                request.adapter_path,
            )
        with llm_lock:
            outputs = llm.generate(
                [request.prompt],
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
        output = outputs[0]
        output_item = output.outputs[0] if output.outputs else None
        text = output_item.text if output_item is not None else ""
        token_count = len(output_item.token_ids) if output_item is not None else 0
        return text, token_count

    return execute


def create_openai_server_request_executor(
    server_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_key: str = "EMPTY",
    timeout_s: float = 600.0,
) -> Callable[[TrafficRequest], tuple[str, int]]:
    endpoint = server_url.rstrip("/") + "/v1/completions"

    def execute(request: TrafficRequest) -> tuple[str, int]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if request.adapter_path is not None:
            payload["extra_body"] = {
                "lora_request": {
                    "lora_name": f"tenant-{request.tenant_id}",
                    "lora_int_id": abs(hash(request.tenant_id)) % 1_000_000 + 1,
                    "lora_path": request.adapter_path,
                }
            }
        http_request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network runtime
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM server request failed: {exc.code} {message}") from exc

        choices = response_payload.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        usage = response_payload.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        return text, completion_tokens

    return execute


def run_concurrent_benchmark(
    requests: Sequence[TrafficRequest],
    executor: Callable[[TrafficRequest], tuple[str, int]],
    concurrency: int,
) -> TrafficBenchmarkResult:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    benchmark_start = time.perf_counter()

    def run_request(request: TrafficRequest) -> RequestBenchmarkResult:
        target_time = benchmark_start + request.arrival_offset_s
        wait_time = target_time - time.perf_counter()
        if wait_time > 0:
            time.sleep(wait_time)
        request_start = time.perf_counter()
        try:
            text, token_count = executor(request)
            latency_ms = (time.perf_counter() - request_start) * 1000.0
            return RequestBenchmarkResult(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                adapter_path=request.adapter_path,
                latency_ms=latency_ms,
                output_tokens=token_count,
                text=text,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            latency_ms = (time.perf_counter() - request_start) * 1000.0
            return RequestBenchmarkResult(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                adapter_path=request.adapter_path,
                latency_ms=latency_ms,
                output_tokens=0,
                text="",
                error=str(exc),
            )

    request_results: list[RequestBenchmarkResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_request, request) for request in requests]
        for future in as_completed(futures):
            request_results.append(future.result())

    wall_time_s = time.perf_counter() - benchmark_start
    request_results.sort(key=lambda item: item.request_id)
    completed = [result for result in request_results if result.error is None]
    latencies = [result.latency_ms for result in completed]
    total_tokens = sum(result.output_tokens for result in completed)
    per_tenant_counts: dict[str, int] = {}
    for result in request_results:
        per_tenant_counts[result.tenant_id] = per_tenant_counts.get(result.tenant_id, 0) + 1

    if latencies:
        p50_latency = float(statistics.median(latencies))
        if len(latencies) == 1:
            p95_latency = latencies[0]
        else:
            p95_index = max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))
            p95_latency = sorted(latencies)[p95_index]
    else:
        p50_latency = 0.0
        p95_latency = 0.0

    throughput_rps = float(len(completed) / wall_time_s) if wall_time_s > 0 else 0.0
    throughput_tps = float(total_tokens / wall_time_s) if wall_time_s > 0 else 0.0

    return TrafficBenchmarkResult(
        pattern=requests[0].metadata.get("pattern", "unknown") if requests else "unknown",
        concurrency=concurrency,
        total_requests=len(requests),
        completed_requests=len(completed),
        errored_requests=len(requests) - len(completed),
        throughput_rps=throughput_rps,
        throughput_tps=throughput_tps,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        wall_time_s=wall_time_s,
        per_tenant_request_counts=per_tenant_counts,
        request_results=request_results,
    )
