from __future__ import annotations

import sys
import types

import pytest
import torch

from lora_spec.metrics import (
    AcceptanceAccumulator,
    SpeculativeDecodingMetrics,
    TimingMetrics,
    aggregate_speculative_metrics,
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


def test_acceptance_accumulator_rejects_depth_beyond_configured_speculation() -> None:
    accumulator = AcceptanceAccumulator(speculation_length=2)
    with pytest.raises(ValueError, match="exceeding configured"):
        accumulator.record([True, True, False])


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


def test_patch_vllm_v1_sampler_recovers_exact_accepted_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection_module = types.ModuleType("vllm.v1.sample.rejection_sampler")

    class Metadata:
        num_draft_tokens = [3, 2]

    class RejectionSampler:
        def forward(self, metadata: Metadata) -> torch.Tensor:
            _ = metadata
            return torch.tensor(
                [
                    [11, 12, 13, 14],  # Three accepted drafts plus bonus.
                    [21, -1, -1, -1],  # Immediate rejection plus recovery.
                ],
                dtype=torch.int32,
            )

        @staticmethod
        def parse_output(
            output_token_ids: torch.Tensor,
            vocab_size: int,
            discard_req_indices: tuple[int, ...] = (),
            logprobs_tensors: object | None = None,
        ) -> tuple[list[list[int]], None]:
            _ = logprobs_tensors
            discarded = set(discard_req_indices)
            outputs = []
            for index, row in enumerate(output_token_ids.tolist()):
                outputs.append(
                    []
                    if index in discarded
                    else [token for token in row if token != -1 and token < vocab_size]
                )
            return outputs, None

    rejection_module.RejectionSampler = RejectionSampler
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample", types.ModuleType("vllm.v1.sample"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.sample.rejection_sampler",
        rejection_module,
    )

    with patch_vllm_rejection_sampler(speculation_length=3) as (accumulator, backend):
        output = RejectionSampler().forward(Metadata())
        RejectionSampler.parse_output(output, vocab_size=100)
    summary = accumulator.summarize()
    assert backend.endswith("RejectionSampler.parse_output.accepted_prefix")
    assert summary.total_drafted_tokens == 5
    assert summary.accepted_drafted_tokens == 3
    assert summary.bonus_tokens == 1
    assert summary.per_position_attempts == [2, 1, 1]


def test_patch_vllm_v1_uses_parsed_tokens_and_skips_discarded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection_module = types.ModuleType("vllm.v1.sample.rejection_sampler")

    class Metadata:
        num_draft_tokens = [2, 2, 2]

    class RejectionSampler:
        def forward(self, metadata: Metadata) -> torch.Tensor:
            _ = metadata
            return torch.tensor(
                [
                    [1, 2, 3],
                    [4, 100, -1],
                    [5, 6, 7],
                ],
                dtype=torch.int32,
            )

        @staticmethod
        def parse_output(
            output_token_ids: torch.Tensor,
            vocab_size: int,
            discard_req_indices: tuple[int, ...] = (),
            logprobs_tensors: object | None = None,
        ) -> tuple[list[list[int]], None]:
            _ = logprobs_tensors
            discarded = set(discard_req_indices)
            outputs = []
            for index, row in enumerate(output_token_ids.tolist()):
                outputs.append(
                    []
                    if index in discarded
                    else [token for token in row if token != -1 and token < vocab_size]
                )
            return outputs, None

    rejection_module.RejectionSampler = RejectionSampler
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample", types.ModuleType("vllm.v1.sample"))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.rejection_sampler", rejection_module)

    with patch_vllm_rejection_sampler(speculation_length=2) as (accumulator, _):
        output = RejectionSampler().forward(Metadata())
        RejectionSampler.parse_output(output, vocab_size=100, discard_req_indices=(2,))

    summary = accumulator.summarize()
    assert summary.total_drafted_tokens == 4
    assert summary.accepted_drafted_tokens == 2
    assert summary.bonus_tokens == 1
    assert summary.per_position_attempts == [2, 1]


def test_patch_vllm_v1_fails_if_sampler_output_is_not_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection_module = types.ModuleType("vllm.v1.sample.rejection_sampler")

    class Metadata:
        num_draft_tokens = [1]

    class RejectionSampler:
        def forward(self, metadata: Metadata) -> torch.Tensor:
            _ = metadata
            return torch.tensor([[1, 2]], dtype=torch.int32)

        @staticmethod
        def parse_output(
            output_token_ids: torch.Tensor,
            vocab_size: int,
            discard_req_indices: tuple[int, ...] = (),
            logprobs_tensors: object | None = None,
        ) -> tuple[list[list[int]], None]:
            _ = discard_req_indices, logprobs_tensors
            return [
                [token for token in row if token != -1 and token < vocab_size]
                for row in output_token_ids.tolist()
            ], None

    rejection_module.RejectionSampler = RejectionSampler
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample", types.ModuleType("vllm.v1.sample"))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.rejection_sampler", rejection_module)

    with pytest.raises(RuntimeError, match="were not parsed in-process"):
        with patch_vllm_rejection_sampler(speculation_length=1):
            RejectionSampler().forward(Metadata())


def test_aggregate_speculative_metrics_weights_each_depth_by_eligible_steps() -> None:
    long = AcceptanceAccumulator()
    long.record([True, True, True])
    short = AcceptanceAccumulator()
    short.record([True])
    short.record([False])

    def measured(accumulator: AcceptanceAccumulator) -> SpeculativeDecodingMetrics:
        return SpeculativeDecodingMetrics(
            acceptance=accumulator.summarize(),
            timing=TimingMetrics(1.0, 1.0, 1, 1.0),
            instrumentation_backend="test",
        )

    aggregate = aggregate_speculative_metrics([measured(long), measured(short)])

    assert aggregate.acceptance.depth_attempts == [3, 1, 1]
    assert aggregate.acceptance.depth_accepted == [2, 1, 1]
    assert aggregate.acceptance.acceptance_by_depth == pytest.approx([2 / 3, 1.0, 1.0])


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
