from __future__ import annotations

import sys
import types

import pytest
import torch

from lora_spec.metrics import (
    AcceptanceAccumulator,
    collect_speculative_metrics,
    patch_vllm_rejection_sampler,
    simulate_speculative_decoding,
)


def _install_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    rejection_module: types.ModuleType,
) -> None:
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor",
        types.ModuleType("vllm.model_executor"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers",
        types.ModuleType("vllm.model_executor.layers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.rejection_sampler",
        rejection_module,
    )


def test_acceptance_accumulator_accounts_for_bonus_token() -> None:
    accumulator = AcceptanceAccumulator(speculation_length=3)
    accumulator.record([True, True, True], bonus_tokens=1)
    summary = accumulator.summarize()
    assert summary.total_drafted_tokens == 3
    assert summary.accepted_drafted_tokens == 3
    assert summary.bonus_tokens == 1
    assert summary.overall_acceptance_rate == 1.0
    assert summary.per_position_acceptance_rate == [1.0, 1.0, 1.0]
    assert summary.acceptance_by_depth == [1.0, 1.0, 1.0]


def test_bonus_token_does_not_inflate_draft_acceptance() -> None:
    accumulator = AcceptanceAccumulator(speculation_length=3)
    accumulator.record([True, True, True], bonus_tokens=1)
    accumulator.record([False, False, False])
    summary = accumulator.summarize()
    assert summary.bonus_tokens == 1
    assert summary.total_drafted_tokens == 6
    assert summary.accepted_drafted_tokens == 3
    assert summary.overall_acceptance_rate == pytest.approx(0.5)


def test_patch_vllm_rejection_sampler_collects_boolean_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection_module = types.ModuleType("vllm.model_executor.layers.rejection_sampler")

    class RejectionSampler:
        def _get_accepted(self) -> torch.Tensor:
            return torch.tensor([[True, False, True]])

        def forward(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
            decisions = self._get_accepted()
            bonus = torch.tensor([[-1]], dtype=draft_token_ids.dtype)
            return torch.cat([torch.where(decisions, draft_token_ids, -1), bonus], dim=-1)

    rejection_module.RejectionSampler = RejectionSampler
    _install_fake_vllm(monkeypatch, rejection_module)

    sampler = RejectionSampler()
    with patch_vllm_rejection_sampler(speculation_length=3) as (accumulator, backend):
        sampler._get_accepted()
    assert backend.endswith("RejectionSampler._get_accepted")
    summary = accumulator.summarize()
    assert summary.total_drafted_tokens == 3
    assert summary.accepted_drafted_tokens == 1
    assert summary.per_position_acceptance_rate == [1.0, 0.0, 0.0]
    assert summary.per_position_attempts == [1, 1, 0]
    assert summary.acceptance_by_depth == [1.0, 0.0, 0.0]


def test_collect_speculative_metrics_infers_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyOutputItem:
        token_ids = [1, 2, 3]
        text = "abc"

    class DummyMetrics:
        arrival_time = 1.0
        first_token_time = 1.05

    class DummyOutput:
        outputs = [DummyOutputItem()]
        metrics = DummyMetrics()

    rejection_module = types.ModuleType("vllm.model_executor.layers.rejection_sampler")

    class RejectionSampler:
        def _get_accepted(self) -> torch.Tensor:
            return torch.tensor([True, False, True])

        def forward(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
            decisions = self._get_accepted().view(1, -1)
            bonus = torch.tensor([[-1]], dtype=draft_token_ids.dtype)
            return torch.cat([torch.where(decisions, draft_token_ids, -1), bonus], dim=-1)

    rejection_module.RejectionSampler = RejectionSampler
    _install_fake_vllm(monkeypatch, rejection_module)

    def generate() -> list[DummyOutput]:
        RejectionSampler()._get_accepted()
        return [DummyOutput()]

    _, metrics = collect_speculative_metrics(generate, speculation_length=3)
    assert metrics.acceptance.accepted_drafted_tokens == 1
    assert metrics.timing.total_generated_tokens == 3
    assert metrics.timing.ttft_ms == pytest.approx(50.0)
    assert metrics.instrumentation_backend.endswith("RejectionSampler._get_accepted")


def test_patch_vllm_rejection_sampler_observes_bonus_tokens_from_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection_module = types.ModuleType("vllm.model_executor.layers.rejection_sampler")

    class RejectionSampler:
        def _get_accepted(self) -> torch.Tensor:
            return torch.tensor([[True, True]])

        def forward(
            self,
            target_probs: torch.Tensor,
            bonus_token_ids: torch.Tensor,
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            generators: list[object],
        ) -> torch.Tensor:
            _ = target_probs, bonus_token_ids, draft_probs, generators
            accepted = self._get_accepted()
            bonus = torch.tensor([[7]], dtype=draft_token_ids.dtype)
            return torch.cat([torch.where(accepted, draft_token_ids, -1), bonus], dim=-1)

    rejection_module.RejectionSampler = RejectionSampler
    _install_fake_vllm(monkeypatch, rejection_module)

    sampler = RejectionSampler()
    with patch_vllm_rejection_sampler(speculation_length=2) as (accumulator, _):
        sampler.forward(
            torch.empty(1, 2, 8),
            torch.tensor([[7]]),
            torch.empty(1, 2, 8),
            torch.tensor([[3, 4]], dtype=torch.long),
            [None],
        )
    summary = accumulator.summarize()
    assert summary.accepted_drafted_tokens == 2
    assert summary.bonus_tokens == 1


def test_simulate_speculative_decoding_tracks_acceptance_and_bonus_tokens() -> None:
    class DummyOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class TransitionModel:
        def __init__(self, transitions: dict[int, int], vocab_size: int = 8) -> None:
            self.transitions = transitions
            self.vocab_size = vocab_size

        def __call__(self, input_ids: torch.Tensor) -> DummyOutput:
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
                fill_value=-1000.0,
                dtype=torch.float32,
                device=input_ids.device,
            )
            for batch_index in range(input_ids.shape[0]):
                for position in range(input_ids.shape[1]):
                    token = int(input_ids[batch_index, position].item())
                    next_token = self.transitions.get(token, 0)
                    logits[batch_index, position, next_token] = 0.0
            return DummyOutput(logits)

    draft_model = TransitionModel({0: 1, 1: 2, 2: 3, 3: 4})
    target_model = TransitionModel({0: 1, 1: 2, 2: 5, 5: 6, 6: 7})
    metrics = simulate_speculative_decoding(
        draft_model=draft_model,
        target_model=target_model,
        prompt_input_ids=[torch.tensor([0], dtype=torch.long)],
        speculation_length=2,
        max_new_tokens=4,
        eos_token_id=None,
        correction=None,
    )

    assert metrics.acceptance.total_drafted_tokens == 3
    assert metrics.acceptance.accepted_drafted_tokens == 2
    assert metrics.acceptance.bonus_tokens == 1
    assert metrics.acceptance.per_position_acceptance_rate == [0.5, 1.0]
    assert metrics.acceptance.acceptance_by_depth == [0.5, 1.0]
    assert metrics.emitted_tokens == 4
    assert metrics.target_model_calls == 2
    assert metrics.tokens_per_target_call == pytest.approx(2.0)
