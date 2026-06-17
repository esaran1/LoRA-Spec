from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class AcceptanceMetrics:
    overall_acceptance_rate: float
    per_position_acceptance_rate: list[float]
    per_position_attempts: list[int]
    per_position_accepted: list[int]
    acceptance_by_depth: list[float]
    depth_attempts: list[int]
    depth_accepted: list[int]
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
    instrumentation_backend: str
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
        self.accepted_prefix_per_depth: list[int] = []
        self.attempted_prefix_per_depth: list[int] = []
        self.accepted_drafted_tokens = 0
        self.total_drafted_tokens = 0
        self.bonus_tokens = 0
        self.speculative_steps = 0
        self.raw_decisions: list[list[bool]] = []

    def record(self, decisions: list[bool], bonus_tokens: int = 0) -> None:
        if not decisions:
            return
        if bonus_tokens < 0:
            raise ValueError("bonus_tokens must be non-negative")
        drafted_decisions = [bool(value) for value in decisions]
        drafted_length = len(drafted_decisions)
        if self.speculation_length is not None and drafted_length > self.speculation_length:
            raise ValueError(
                f"Captured {drafted_length} draft decisions, exceeding configured "
                f"speculation_length={self.speculation_length}"
            )
        self.speculative_steps += 1
        self.total_drafted_tokens += drafted_length

        while len(self.accepted_per_position) < drafted_length:
            self.accepted_per_position.append(0)
            self.attempted_per_position.append(0)
            self.accepted_prefix_per_depth.append(0)
            self.attempted_prefix_per_depth.append(0)
        prefix_survives = True
        effective_decisions: list[bool] = []
        for index, accepted in enumerate(drafted_decisions):
            reached_position = prefix_survives
            if reached_position:
                self.attempted_per_position[index] += 1
                self.accepted_per_position[index] += int(accepted)
            prefix_survives = reached_position and accepted
            effective_decisions.append(prefix_survives)
            self.attempted_prefix_per_depth[index] += 1
            self.accepted_prefix_per_depth[index] += int(prefix_survives)
        self.accepted_drafted_tokens += sum(effective_decisions)
        self.raw_decisions.append(drafted_decisions)
        self.bonus_tokens += bonus_tokens

    def add_bonus_tokens(self, count: int) -> None:
        if count < 0:
            raise ValueError("bonus token count must be non-negative")
        self.bonus_tokens += count

    def summarize(self) -> AcceptanceMetrics:
        denominator = self.total_drafted_tokens
        numerator = self.accepted_drafted_tokens
        overall = float(numerator / denominator) if denominator > 0 else 0.0
        per_position = [
            float(accepted / attempted) if attempted > 0 else 0.0
            for accepted, attempted in zip(self.accepted_per_position, self.attempted_per_position)
        ]
        acceptance_by_depth = [
            float(accepted / attempted) if attempted > 0 else 0.0
            for accepted, attempted in zip(
                self.accepted_prefix_per_depth,
                self.attempted_prefix_per_depth,
            )
        ]
        return AcceptanceMetrics(
            overall_acceptance_rate=overall,
            per_position_acceptance_rate=per_position,
            per_position_attempts=list(self.attempted_per_position),
            per_position_accepted=list(self.accepted_per_position),
            acceptance_by_depth=acceptance_by_depth,
            depth_attempts=list(self.attempted_prefix_per_depth),
            depth_accepted=list(self.accepted_prefix_per_depth),
            accepted_drafted_tokens=self.accepted_drafted_tokens,
            total_drafted_tokens=self.total_drafted_tokens,
            bonus_tokens=self.bonus_tokens,
            speculative_steps=self.speculative_steps,
        )


def _record_tensor(
    accumulator: AcceptanceAccumulator,
    tensor: torch.Tensor,
) -> None:
    data = tensor.detach().cpu()
    if data.ndim == 0:
        decision = bool(data.item())
        accumulator.record([decision])
        return
    if data.ndim == 1:
        decisions = [bool(value) for value in data.tolist()]
        accumulator.record(decisions)
        return
    flattened = data.reshape(-1, data.shape[-1])
    for row in flattened:
        decisions = [bool(value) for value in row.tolist()]
        accumulator.record(decisions)


def _wrap_acceptance_method(
    rejection_sampler_class: type[Any],
    accumulator: AcceptanceAccumulator,
    restored: list[tuple[Any, str, Any]],
) -> Callable[[], None] | None:
    accepted_attribute = "_get_accepted"
    original_accepted = getattr(rejection_sampler_class, accepted_attribute, None)
    original_forward = getattr(rejection_sampler_class, "forward", None)
    if original_forward is None or not callable(original_forward):
        raise RuntimeError("Installed vLLM RejectionSampler does not expose forward")

    if original_accepted is None or not callable(original_accepted):
        original_parse = getattr(rejection_sampler_class, "parse_output", None)
        parse_descriptor = rejection_sampler_class.__dict__.get("parse_output")
        if original_parse is None or not callable(original_parse) or parse_descriptor is None:
            raise RuntimeError("Installed vLLM V1 RejectionSampler does not expose parse_output")
        pending_draft_counts: list[list[int]] = []

        def wrapped_v1_forward(*args: Any, **kwargs: Any) -> Any:
            result = original_forward(*args, **kwargs)
            metadata = kwargs.get("metadata")
            if metadata is None and len(args) >= 2:
                metadata = args[1]
            token_ids = getattr(result, "sampled_token_ids", result)
            draft_counts = getattr(metadata, "num_draft_tokens", None)
            if not isinstance(token_ids, torch.Tensor) or draft_counts is None:
                raise RuntimeError("Unsupported vLLM V1 rejection-sampler output contract")
            if token_ids.ndim != 2 or len(draft_counts) != token_ids.shape[0]:
                raise RuntimeError("vLLM V1 sampler metadata is not batch-aligned")
            normalized_counts = [int(count) for count in draft_counts]
            if any(count < 0 for count in normalized_counts):
                raise RuntimeError("vLLM V1 sampler reported a negative draft-token count")
            pending_draft_counts.append(normalized_counts)
            return result

        def wrapped_parse_output(
            output_token_ids: torch.Tensor,
            vocab_size: int,
            discard_req_indices: Any = (),
            logprobs_tensors: Any = None,
        ) -> Any:
            result = original_parse(
                output_token_ids,
                vocab_size,
                discard_req_indices,
                logprobs_tensors,
            )
            if not pending_draft_counts:
                raise RuntimeError("vLLM V1 parse_output ran without matching sampler metadata")
            draft_counts = pending_draft_counts.pop(0)
            parsed_token_ids = result[0]
            if len(parsed_token_ids) != len(draft_counts):
                raise RuntimeError("vLLM V1 parsed output is not batch-aligned")
            discarded = {int(index) for index in discard_req_indices}
            for row_index, (emitted_tokens, draft_count) in enumerate(
                zip(parsed_token_ids, draft_counts)
            ):
                if row_index in discarded or draft_count <= 0:
                    continue
                emitted_count = len(emitted_tokens)
                if emitted_count < 1:
                    raise RuntimeError(
                        "vLLM emitted no recovery token for a non-discarded speculative request"
                    )
                accepted_count = min(max(emitted_count - 1, 0), draft_count)
                decisions = [True] * accepted_count + [False] * (draft_count - accepted_count)
                accumulator.record(
                    decisions,
                    bonus_tokens=int(emitted_count > draft_count),
                )
            return result

        setattr(rejection_sampler_class, "forward", wrapped_v1_forward)
        setattr(rejection_sampler_class, "parse_output", staticmethod(wrapped_parse_output))
        restored.append((rejection_sampler_class, "forward", original_forward))
        restored.append((rejection_sampler_class, "parse_output", parse_descriptor))

        def validate_all_outputs_parsed() -> None:
            if pending_draft_counts:
                raise RuntimeError(
                    "vLLM V1 sampler outputs were not parsed in-process; acceptance metrics "
                    "cannot be attributed exactly"
                )

        return validate_all_outputs_parsed

    def wrapped_accepted(*args: Any, **kwargs: Any) -> Any:
        result = original_accepted(*args, **kwargs)
        if (
            not isinstance(result, torch.Tensor)
            or result.dtype != torch.bool
            or result.numel() == 0
        ):
            raise RuntimeError("vLLM RejectionSampler._get_accepted did not return a boolean mask")
        _record_tensor(accumulator, result)
        return result

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        result = original_forward(*args, **kwargs)
        draft_token_ids = kwargs.get("draft_token_ids")
        if draft_token_ids is None and len(args) >= 5:
            draft_token_ids = args[4]
        if not isinstance(result, torch.Tensor) or not isinstance(draft_token_ids, torch.Tensor):
            raise RuntimeError("vLLM RejectionSampler.forward returned an unsupported structure")
        bonus_width = result.shape[-1] - draft_token_ids.shape[-1]
        if bonus_width < 0:
            raise RuntimeError("vLLM sampler output is shorter than the draft sequence")
        if bonus_width > 1:
            raise RuntimeError(
                "Pinned vLLM sampler contract permits at most one bonus-token column"
            )
        if bonus_width > 0:
            bonus_tokens = int((result[..., -bonus_width:] >= 0).sum().item())
            accumulator.add_bonus_tokens(bonus_tokens)
        return result

    setattr(rejection_sampler_class, accepted_attribute, wrapped_accepted)
    setattr(rejection_sampler_class, "forward", wrapped_forward)
    restored.append((rejection_sampler_class, accepted_attribute, original_accepted))
    restored.append((rejection_sampler_class, "forward", original_forward))
    return None


def _resolve_vllm_rejection_sampler() -> tuple[type[Any], str]:
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None
    supported_versions = {"0.5.3.post1", "0.15.1"}
    if vllm_version is not None and vllm_version not in supported_versions:
        raise RuntimeError(
            "Acceptance instrumentation is validated only for "
            f"{sorted(supported_versions)}; "
            f"installed version is {vllm_version}",
        )
    candidates = (
        "vllm.v1.sample.rejection_sampler",
        "vllm.model_executor.layers.rejection_sampler",
        "vllm.spec_decode.rejection_sampler",
    )
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        rejection_sampler_class = getattr(module, "RejectionSampler", None)
        if isinstance(rejection_sampler_class, type):
            method = (
                "_get_accepted"
                if callable(getattr(rejection_sampler_class, "_get_accepted", None))
                else "parse_output.accepted_prefix"
            )
            return rejection_sampler_class, f"{module_name}.RejectionSampler.{method}"
        errors.append(f"{module_name}: RejectionSampler class not found")
    raise ImportError("No supported vLLM rejection sampler was found: " + "; ".join(errors))


@contextlib.contextmanager
def patch_vllm_rejection_sampler(
    speculation_length: int | None = None,
) -> Iterator[tuple[AcceptanceAccumulator, str]]:
    rejection_sampler_class, backend = _resolve_vllm_rejection_sampler()
    accumulator = AcceptanceAccumulator(speculation_length=speculation_length)
    restored: list[tuple[Any, str, Any]] = []
    validate_complete = _wrap_acceptance_method(rejection_sampler_class, accumulator, restored)
    try:
        yield accumulator, backend
        if validate_complete is not None:
            validate_complete()
    finally:
        for owner, attribute_name, original in reversed(restored):
            setattr(owner, attribute_name, original)


def aggregate_speculative_metrics(
    metrics: list[SpeculativeDecodingMetrics],
) -> SpeculativeDecodingMetrics:
    if not metrics:
        raise ValueError("At least one speculative-decoding metric is required")
    maximum_positions = max(len(item.acceptance.per_position_attempts) for item in metrics)
    attempts = [0] * maximum_positions
    accepted = [0] * maximum_positions
    depth_attempts = [0] * maximum_positions
    depth_accepted = [0] * maximum_positions
    for item in metrics:
        for index, count in enumerate(item.acceptance.per_position_attempts):
            attempts[index] += count
            accepted[index] += item.acceptance.per_position_accepted[index]
        for index, count in enumerate(item.acceptance.depth_attempts):
            depth_attempts[index] += count
            depth_accepted[index] += item.acceptance.depth_accepted[index]

    total_drafted = sum(item.acceptance.total_drafted_tokens for item in metrics)
    total_accepted = sum(item.acceptance.accepted_drafted_tokens for item in metrics)
    total_generated_tokens = sum(item.timing.total_generated_tokens for item in metrics)
    total_wall_time_s = sum(item.timing.wall_time_s for item in metrics)
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
                accepted_count / attempt_count if attempt_count else 0.0
                for accepted_count, attempt_count in zip(depth_accepted, depth_attempts)
            ],
            depth_attempts=depth_attempts,
            depth_accepted=depth_accepted,
            accepted_drafted_tokens=total_accepted,
            total_drafted_tokens=total_drafted,
            bonus_tokens=sum(item.acceptance.bonus_tokens for item in metrics),
            speculative_steps=sum(item.acceptance.speculative_steps for item in metrics),
        ),
        timing=TimingMetrics(
            throughput_tps=(
                float(total_generated_tokens / total_wall_time_s)
                if total_wall_time_s > 0.0
                else 0.0
            ),
            ttft_ms=sum(ttft_values) / len(ttft_values) if ttft_values else float("nan"),
            total_generated_tokens=total_generated_tokens,
            wall_time_s=total_wall_time_s,
        ),
        instrumentation_backend=",".join(backends),
        raw_decisions=[decision for item in metrics for decision in item.raw_decisions],
    )


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
    with patch_vllm_rejection_sampler(speculation_length=speculation_length) as (
        accumulator,
        backend,
    ):
        start_time = time.perf_counter()
        outputs = generate_fn()
        wall_time_s = time.perf_counter() - start_time
    acceptance = accumulator.summarize()
    if acceptance.total_drafted_tokens == 0:
        raise RuntimeError(
            "No vLLM rejection decisions were captured. The sampler likely ran in a separate worker process; "
            "this instrumentation backend is unsupported for that executor and will not emit false metrics."
        )
    timing = infer_timing_metrics(outputs, wall_time_s)
    return outputs, SpeculativeDecodingMetrics(
        acceptance=acceptance,
        timing=timing,
        instrumentation_backend=backend,
        raw_decisions=accumulator.raw_decisions,
    )


def _next_token_output(
    model: Any,
    input_ids: torch.Tensor,
    output_hidden_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        outputs = model(input_ids=input_ids, output_hidden_states=output_hidden_states)
    except TypeError:
        if output_hidden_states:
            raise
        outputs = model(input_ids=input_ids)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("Model outputs must expose a logits tensor")
    hidden_state = None
    if output_hidden_states:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            hidden_state = hidden_states[-1][:, -1, :].float()
    return logits[:, -1, :].float(), hidden_state


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
                requires_hidden = (
                    bool(getattr(correction, "requires_hidden_state", False))
                    if correction is not None
                    else False
                )
                draft_logits, hidden_state = _next_token_output(
                    draft_model,
                    draft_context,
                    output_hidden_states=requires_hidden,
                )
                draft_model_calls += 1
                if correction is not None:
                    draft_logits = correction.apply(draft_logits, hidden_state=hidden_state)
                next_token = int(_greedy_token(draft_logits)[0].item())
                proposed_tokens.append(next_token)
                next_token_tensor = torch.tensor(
                    [[next_token]], device=draft_context.device, dtype=draft_context.dtype
                )
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

            decisions.extend([False] * (len(proposed_tokens) - len(decisions)))

            bonus_allowed = (not mismatch) and (
                generated_for_prompt + len(accepted_tokens) < max_new_tokens
            )
            if bonus_allowed:
                bonus_token = int(
                    torch.argmax(target_logits[prefix_length - 1 + len(proposed_tokens)]).item()
                )
                accepted_tokens.append(bonus_token)

            accumulator.record(decisions, bonus_tokens=int(bonus_allowed))
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
    tokens_per_target_call = (
        float(emitted_tokens / target_model_calls) if target_model_calls > 0 else 0.0
    )
    draft_tokens_per_call = (
        float(emitted_tokens / draft_model_calls) if draft_model_calls > 0 else 0.0
    )
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
