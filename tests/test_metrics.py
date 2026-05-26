from __future__ import annotations

import sys
import types

import pytest
import torch

from lora_spec.metrics import AcceptanceAccumulator, collect_speculative_metrics, patch_vllm_rejection_sampler


def test_acceptance_accumulator_accounts_for_bonus_token() -> None:
    accumulator = AcceptanceAccumulator(speculation_length=3)
    accumulator.record([True, True, True, True])
    summary = accumulator.summarize()
    assert summary.total_drafted_tokens == 3
    assert summary.accepted_drafted_tokens == 3
    assert summary.bonus_tokens == 1
    assert summary.overall_acceptance_rate == 1.0
    assert summary.per_position_acceptance_rate == [1.0, 1.0, 1.0]


def test_patch_vllm_rejection_sampler_collects_boolean_masks() -> None:
    rejection_module = types.ModuleType("vllm.spec_decode.rejection_sampler")

    class RejectionSampler:
        def forward(self) -> torch.Tensor:
            return torch.tensor([[1, 0, 1]], dtype=torch.int64)

    rejection_module.RejectionSampler = RejectionSampler
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.spec_decode"] = types.ModuleType("vllm.spec_decode")
    sys.modules["vllm.spec_decode.rejection_sampler"] = rejection_module

    sampler = RejectionSampler()
    with patch_vllm_rejection_sampler(speculation_length=3) as accumulator:
        sampler.forward()
    summary = accumulator.summarize()
    assert summary.total_drafted_tokens == 3
    assert summary.accepted_drafted_tokens == 2
    assert summary.per_position_acceptance_rate == [1.0, 0.0, 1.0]


def test_collect_speculative_metrics_infers_timing() -> None:
    class DummyOutputItem:
        token_ids = [1, 2, 3]
        text = "abc"

    class DummyMetrics:
        arrival_time = 1.0
        first_token_time = 1.05

    class DummyOutput:
        outputs = [DummyOutputItem()]
        metrics = DummyMetrics()

    rejection_module = types.ModuleType("vllm.spec_decode.rejection_sampler")

    def sample() -> torch.Tensor:
        return torch.tensor([True, False, True])

    rejection_module.sample = sample
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.spec_decode"] = types.ModuleType("vllm.spec_decode")
    sys.modules["vllm.spec_decode.rejection_sampler"] = rejection_module

    def generate() -> list[DummyOutput]:
        rejection_module.sample()
        return [DummyOutput()]

    _, metrics = collect_speculative_metrics(generate, speculation_length=3)
    assert metrics.acceptance.accepted_drafted_tokens == 2
    assert metrics.timing.total_generated_tokens == 3
    assert metrics.timing.ttft_ms == pytest.approx(50.0)
