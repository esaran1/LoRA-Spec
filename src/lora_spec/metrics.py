from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable

import torch


@dataclass
class AcceptanceMetrics:
    overall_acceptance_rate: float
    per_position_acceptance_rate: list[float]
    accepted_drafted_tokens: int
    total_drafted_tokens: int
    bonus_tokens: int
    speculative_steps: int


@dataclass
class TimingMetrics:
    throughput_tps: float
    ttft_ms: float
    total_generated_tokens: int
    wall_time_s: float


@dataclass
class SpeculativeDecodingMetrics:
    acceptance: AcceptanceMetrics
    timing: TimingMetrics
    raw_decisions: list[list[bool]] = field(default_factory=list)


@dataclass
class SpeculativeDecodingProxyMetrics:
    acceptance: AcceptanceMetrics
    emitted_tokens: int
    target_model_calls: int
    draft_model_calls: int
    tokens_per_target_call: float
    draft_tokens_per_call: float
    target_call_reduction_vs_autoregressive: float


class AcceptanceAccumulator:
    def __init__(self, speculation_length: int | None = None) -> None:
        self.speculation_length = speculation_length
        self.accepted_per_position: list[int] = []
        self.attempted_per_position: list[int] = []
        self.accepted_drafted_tokens = 0
        self.total_drafted_tokens = 0
        self.bonus_tokens = 0
        self.speculative_steps = 0
        self.raw_decisions: list[list[bool]] = []

    def record(self, decisions: list[bool]) -> None:
        if not decisions:
            return
        drafted_length = len(decisions)
        has_bonus = False
        if self.speculation_length is not None and drafted_length == self.speculation_length + 1:
            has_bonus = True
            drafted_length -= 1
        elif self.speculation_length is not None and drafted_length > self.speculation_length:
            has_bonus = True
            drafted_length = self.speculation_length
        elif drafted_length > 1 and all(decisions[:-1]) and decisions[-1]:
            has_bonus = True
            drafted_length -= 1

        drafted_decisions = decisions[:drafted_length]
        bonus_decision = decisions[drafted_length] if has_bonus and drafted_length < len(decisions) else False
        self.raw_decisions.append(list(decisions))
        self.speculative_steps += 1
        self.total_drafted_tokens += drafted_length
        self.accepted_drafted_tokens += sum(bool(value) for value in drafted_decisions)

        while len(self.accepted_per_position) < drafted_length:
            self.accepted_per_position.append(0)
            self.attempted_per_position.append(0)
        for index, accepted in enumerate(drafted_decisions):
            self.attempted_per_position[index] += 1
            self.accepted_per_position[index] += int(accepted)
        if bonus_decision:
            self.bonus_tokens += 1

    def summarize(self) -> AcceptanceMetrics:
        denominator = self.total_drafted_tokens + self.bonus_tokens
        numerator = self.accepted_drafted_tokens + self.bonus_tokens
        overall = float(numerator / denominator) if denominator > 0 else 0.0
        per_position = [
            float(accepted / attempted) if attempted > 0 else 0.0
            for accepted, attempted in zip(self.accepted_per_position, self.attempted_per_position)
        ]
        return AcceptanceMetrics(
            overall_acceptance_rate=overall,
            per_position_acceptance_rate=per_position,
            accepted_drafted_tokens=self.accepted_drafted_tokens,
            total_drafted_tokens=self.total_drafted_tokens,
            bonus_tokens=self.bonus_tokens,
            speculative_steps=self.speculative_steps,
        )


def _extract_boolean_tensor(payload: Any) -> torch.Tensor | None:
    if isinstance(payload, torch.Tensor):
        if payload.dtype == torch.bool and payload.numel() > 0:
            return payload
        if payload.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            unique_values = payload.unique()
            if unique_values.numel() <= 2 and set(unique_values.detach().cpu().tolist()).issubset({0, 1}):
                return payload.bool()
        return None
    if isinstance(payload, dict):
        for key in ("accepted", "accept_mask", "accepted_mask", "bonus_mask"):
            if key in payload:
                extracted = _extract_boolean_tensor(payload[key])
                if extracted is not None:
                    return extracted
        for value in payload.values():
            extracted = _extract_boolean_tensor(value)
            if extracted is not None:
                return extracted
        return None
    if isinstance(payload, (list, tuple)):
        for value in payload:
            extracted = _extract_boolean_tensor(value)
            if extracted is not None:
                return extracted
    return None


def _record_tensor(accumulator: AcceptanceAccumulator, tensor: torch.Tensor) -> None:
    data = tensor.detach().cpu()
    if data.ndim == 0:
        accumulator.record([bool(data.item())])
        return
    if data.ndim == 1:
        accumulator.record([bool(value) for value in data.tolist()])
        return
    flattened = data.reshape(-1, data.shape[-1])
    for row in flattened:
        accumulator.record([bool(value) for value in row.tolist()])


def _wrap_callable(
    owner: Any,
    attribute_name: str,
    accumulator: AcceptanceAccumulator,
    restored: list[tuple[Any, str, Any]],
) -> None:
    original = getattr(owner, attribute_name, None)
    if original is None or not callable(original):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        decisions = _extract_boolean_tensor(result)
        if decisions is not None:
            _record_tensor(accumulator, decisions)
        return result

    setattr(owner, attribute_name, wrapped)
    restored.append((owner, attribute_name, original))


@contextlib.contextmanager
def patch_vllm_rejection_sampler(
    speculation_length: int | None = None,
) -> AcceptanceAccumulator:
    try:
        import vllm.spec_decode.rejection_sampler as rejection_sampler
    except ImportError as exc:  # pragma: no cover - requires vLLM runtime
        raise ImportError("vLLM must be installed to patch speculative decoding internals") from exc

    accumulator = AcceptanceAccumulator(speculation_length=speculation_length)
    restored: list[tuple[Any, str, Any]] = []
    candidate_names = [
        "forward",
        "__call__",
        "sample",
        "modified_rejection_sampling",
        "_batch_modified_rejection_sampling",
        "batch_modified_rejection_sampling",
    ]

    for owner in _candidate_owners(rejection_sampler):
        for candidate_name in candidate_names:
            _wrap_callable(owner, candidate_name, accumulator, restored)
    try:
        yield accumulator
    finally:
        for owner, attribute_name, original in reversed(restored):
            setattr(owner, attribute_name, original)


def _candidate_owners(module: ModuleType) -> list[Any]:
    owners: list[Any] = [module]
    for attribute_name in dir(module):
        value = getattr(module, attribute_name)
        if isinstance(value, type):
            owners.append(value)
    return owners


def infer_timing_metrics(
    outputs: list[Any],
    wall_time_s: float,
) -> TimingMetrics:
    total_tokens = 0
    ttft_values_ms: list[float] = []
    for output in outputs:
        output_items = getattr(output, "outputs", [])
        if output_items:
            total_tokens += len(getattr(output_items[0], "token_ids", []))
        metrics = getattr(output, "metrics", None)
        first_token_time = getattr(metrics, "first_token_time", None)
        arrival_time = getattr(metrics, "arrival_time", None)
        first_scheduled_time = getattr(metrics, "first_scheduled_time", None)
        if first_token_time is not None:
            baseline = arrival_time if arrival_time is not None else first_scheduled_time
            if baseline is not None:
                ttft_values_ms.append(float((first_token_time - baseline) * 1000.0))
    throughput = float(total_tokens / wall_time_s) if wall_time_s > 0 else 0.0
    ttft = float(sum(ttft_values_ms) / len(ttft_values_ms)) if ttft_values_ms else float("nan")
    return TimingMetrics(
        throughput_tps=throughput,
        ttft_ms=ttft,
        total_generated_tokens=total_tokens,
        wall_time_s=wall_time_s,
    )


def collect_speculative_metrics(
    generate_fn: Callable[[], list[Any]],
    speculation_length: int | None = None,
) -> tuple[list[Any], SpeculativeDecodingMetrics]:
    with patch_vllm_rejection_sampler(speculation_length=speculation_length) as accumulator:
        start_time = time.perf_counter()
        outputs = generate_fn()
        wall_time_s = time.perf_counter() - start_time
    acceptance = accumulator.summarize()
    timing = infer_timing_metrics(outputs, wall_time_s)
    return outputs, SpeculativeDecodingMetrics(
        acceptance=acceptance,
        timing=timing,
        raw_decisions=accumulator.raw_decisions,
    )


def _next_token_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("Model outputs must expose a logits tensor")
    return logits[:, -1, :].float()


def _greedy_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


@torch.inference_mode()
def simulate_speculative_decoding(
    draft_model: Any,
    target_model: Any,
    prompt_input_ids: list[torch.Tensor],
    speculation_length: int,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    correction: Any | None = None,
) -> SpeculativeDecodingProxyMetrics:
    accumulator = AcceptanceAccumulator(speculation_length=speculation_length)
    emitted_tokens = 0
    target_model_calls = 0
    draft_model_calls = 0

    for prompt_ids in prompt_input_ids:
        context = prompt_ids.view(1, -1).clone()
        generated_for_prompt = 0

        while generated_for_prompt < max_new_tokens:
            current_spec_length = min(speculation_length, max_new_tokens - generated_for_prompt)
            draft_context = context
            proposed_tokens: list[int] = []
            for _ in range(current_spec_length):
                draft_logits = _next_token_logits(draft_model, draft_context)
                draft_model_calls += 1
                if correction is not None:
                    draft_logits = correction.apply(draft_logits)
                next_token = int(_greedy_token(draft_logits)[0].item())
                proposed_tokens.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], device=draft_context.device, dtype=draft_context.dtype)
                draft_context = torch.cat([draft_context, next_token_tensor], dim=1)

            if not proposed_tokens:
                break

            proposed_tensor = torch.tensor(
                [proposed_tokens],
                device=context.device,
                dtype=context.dtype,
            )
            verification_input = torch.cat([context, proposed_tensor], dim=1)
            target_logits = target_model(input_ids=verification_input).logits[0].float()
            target_model_calls += 1

            decisions: list[bool] = []
            accepted_tokens: list[int] = []
            prefix_length = context.shape[1]
            mismatch = False

            for position, proposed_token in enumerate(proposed_tokens):
                target_next = int(torch.argmax(target_logits[prefix_length - 1 + position]).item())
                accepted = target_next == proposed_token
                decisions.append(accepted)
                if accepted:
                    accepted_tokens.append(proposed_token)
                else:
                    accepted_tokens.append(target_next)
                    mismatch = True
                    break

            bonus_allowed = (not mismatch) and (generated_for_prompt + len(accepted_tokens) < max_new_tokens)
            if bonus_allowed:
                bonus_token = int(torch.argmax(target_logits[prefix_length - 1 + len(proposed_tokens)]).item())
                decisions.append(True)
                accepted_tokens.append(bonus_token)

            accumulator.record(decisions)
            if not accepted_tokens:
                break

            appended = torch.tensor(
                [accepted_tokens],
                device=context.device,
                dtype=context.dtype,
            )
            context = torch.cat([context, appended], dim=1)
            emitted_now = len(accepted_tokens)
            emitted_tokens += emitted_now
            generated_for_prompt += emitted_now

            if eos_token_id is not None and accepted_tokens[-1] == eos_token_id:
                break

    acceptance = accumulator.summarize()
    tokens_per_target_call = float(emitted_tokens / target_model_calls) if target_model_calls > 0 else 0.0
    draft_tokens_per_call = float(emitted_tokens / draft_model_calls) if draft_model_calls > 0 else 0.0
    autoregressive_target_calls = max(emitted_tokens, 1)
    target_call_reduction = 1.0 - (target_model_calls / autoregressive_target_calls)
    return SpeculativeDecodingProxyMetrics(
        acceptance=acceptance,
        emitted_tokens=emitted_tokens,
        target_model_calls=target_model_calls,
        draft_model_calls=draft_model_calls,
        tokens_per_target_call=tokens_per_target_call,
        draft_tokens_per_call=draft_tokens_per_call,
        target_call_reduction_vs_autoregressive=float(target_call_reduction),
    )
