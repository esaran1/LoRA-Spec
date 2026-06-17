from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from transformers import PreTrainedModel, PretrainedConfig

from lora_spec.correction import ContextDependentCorrection, LowRankCorrection, MeanShiftCorrection
from lora_spec.theory import ContinuationContextSet, LogitShiftDataset


class TinyConfig(PretrainedConfig):
    model_type = "tiny"

    def __init__(self, vocab_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size


class TinyTokenizer:
    def __init__(self, vocab_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None
        self.unk_token_id = None
        self.mask_token_id = None
        self.sep_token_id = None
        self.cls_token_id = None

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return {str(index): index for index in range(self.vocab_size)}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def __call__(
        self,
        texts: list[str] | str,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ):
        _ = return_tensors, padding, truncation, add_special_tokens
        if isinstance(texts, str):
            return {"input_ids": [int(piece) % self.vocab_size for piece in texts.split()]}
        tokenized = []
        for text in texts:
            values = [int(piece) % self.vocab_size for piece in text.split()]
            tokenized.append(values)
        max_len = max(len(row) for row in tokenized)
        input_ids = []
        attention_mask = []
        for row in tokenized:
            pad = [0] * (max_len - len(row))
            input_ids.append(row + pad)
            attention_mask.append([1] * len(row) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TinyOutput:
    def __init__(
        self, logits: torch.Tensor, hidden_states: tuple[torch.Tensor, ...] | None = None
    ) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class TinyLM(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, shift_matrix: torch.Tensor | None = None) -> None:
        super().__init__(TinyConfig())
        self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.vocab_size)
        self.lm_head = torch.nn.Linear(self.config.vocab_size, self.config.vocab_size, bias=False)
        self.shift_matrix = torch.nn.Parameter(
            torch.zeros(self.config.vocab_size, self.config.vocab_size)
            if shift_matrix is None
            else shift_matrix.clone().float(),
        )
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(self.config.vocab_size))
            self.lm_head.weight.copy_(torch.eye(self.config.vocab_size))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden) + hidden @ self.shift_matrix
        hidden_states = (hidden,) if output_hidden_states else None
        return TinyOutput(logits=logits, hidden_states=hidden_states)


def _all_token_contexts() -> ContinuationContextSet:
    return ContinuationContextSet(
        input_ids=(torch.tensor([0, 1, 2, 3, 0], dtype=torch.long),),
        prompt_lengths=(1,),
        continuation_lengths=(4,),
        trajectory_model="synthetic_base_target",
        generation_policy="fixed_test_contexts",
    )


def _shift_dataset(shift_matrix: torch.Tensor) -> LogitShiftDataset:
    base_logits = torch.zeros_like(shift_matrix)
    return LogitShiftDataset(
        shift_matrix=shift_matrix.float(),
        base_logits_matrix=base_logits,
        adapted_logits_matrix=shift_matrix.float(),
        hidden_state_matrix=None,
        prompt_indices=[0] * shift_matrix.shape[0],
        token_positions=list(range(shift_matrix.shape[0])),
        vocabulary_size=int(shift_matrix.shape[1]),
        continuation_contexts=_all_token_contexts(),
    )


def test_mean_shift_correction_recovers_constant_shift() -> None:
    tokenizer = TinyTokenizer()
    delta = torch.tensor(
        [
            [0.4, -0.2, 0.1, -0.3],
            [0.4, -0.2, 0.1, -0.3],
            [0.4, -0.2, 0.1, -0.3],
            [0.4, -0.2, 0.1, -0.3],
        ],
    )
    base = TinyLM()
    adapted = TinyLM(shift_matrix=delta)
    correction = MeanShiftCorrection().calibrate(
        base,
        adapted,
        ["0"],
        tokenizer=tokenizer,
        continuation_contexts=_all_token_contexts(),
    )
    adjusted = correction.apply(torch.zeros(1, tokenizer.vocab_size)).squeeze(0)
    assert adjusted.shape[0] == tokenizer.vocab_size
    assert torch.allclose(adjusted, delta.mean(dim=0), atol=1e-5)


def test_low_rank_correction_recovers_structured_shift() -> None:
    tokenizer = TinyTokenizer()
    left = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    right = torch.tensor([[0.5, -0.25, 0.125, -0.125]])
    shift_matrix = left @ right
    base = TinyLM()
    adapted = TinyLM(shift_matrix=shift_matrix)
    correction = LowRankCorrection(rank=1).calibrate(
        base,
        adapted,
        ["0"],
        tokenizer=tokenizer,
        continuation_contexts=_all_token_contexts(),
    )
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    adjusted = correction.apply(logits)
    expected = logits + shift_matrix[0].view(1, -1)
    assert adjusted.shape == expected.shape
    assert torch.linalg.norm(adjusted - expected) < 0.2
    report = correction.approximation_error()
    assert report.selected_rank == 1
    assert report.spectral_tail_relative_frobenius <= 1.0
    assert report.centered_shift_reconstruction_relative_frobenius < 1e-4
    assert report.coefficient_regression_relative_frobenius < 0.2
    assert report.centered_operator_relative_frobenius == pytest.approx(
        report.predicted_centered_operator_relative_frobenius,
        abs=1e-5,
    )
    assert report.operator_calibration_relative_frobenius < 0.2
    assert report.end_to_end_calibration_relative_frobenius < 0.2


def test_context_dependent_correction_uses_hidden_state() -> None:
    tokenizer = TinyTokenizer()
    shift_matrix = torch.tensor(
        [
            [0.4, -0.4, 0.0, 0.0],
            [0.0, 0.0, 0.3, -0.3],
            [0.4, -0.4, 0.0, 0.0],
            [0.0, 0.0, 0.3, -0.3],
        ],
    )
    base = TinyLM()
    adapted = TinyLM(shift_matrix=shift_matrix)
    correction = ContextDependentCorrection(
        rank=2, hidden_dim=8, epochs=100, lr=1e-2, seed=0
    ).calibrate(
        base,
        adapted,
        ["0"],
        tokenizer=tokenizer,
        continuation_contexts=_all_token_contexts(),
    )
    hidden_state = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    adjusted = correction.apply(torch.zeros(1, tokenizer.vocab_size), hidden_state=hidden_state)
    expected = shift_matrix[0].view(1, -1)
    assert torch.linalg.norm(adjusted - expected) < 0.25


def test_context_dependent_recalibration_clears_cached_tensors() -> None:
    feature_matrix = torch.eye(4)
    alternating_shift = torch.tensor(
        [
            [0.4, -0.4, 0.0, 0.0],
            [-0.4, 0.4, 0.0, 0.0],
            [0.4, -0.4, 0.0, 0.0],
            [-0.4, 0.4, 0.0, 0.0],
        ],
    )
    mean_offset = torch.tensor([0.0, 0.0, 1.5, -1.5])
    correction = ContextDependentCorrection(
        rank=1,
        hidden_dim=4,
        epochs=0,
        lr=1e-2,
        seed=11,
    )
    correction.calibrate_from_dataset(_shift_dataset(alternating_shift), feature_matrix)
    first = correction.apply(
        torch.zeros(1, 4),
        hidden_state=feature_matrix[:1],
    )

    correction.calibrate_from_dataset(
        _shift_dataset(alternating_shift + mean_offset),
        feature_matrix,
    )
    second = correction.apply(
        torch.zeros(1, 4),
        hidden_state=feature_matrix[:1],
    )

    observed_offset = second - first
    assert torch.allclose(observed_offset, mean_offset.view(1, -1), atol=1e-6)


def test_context_dependent_uses_population_feature_std_with_variance_floor() -> None:
    shift = torch.tensor(
        [
            [0.4, -0.4, 0.0, 0.0],
            [-0.4, 0.4, 0.0, 0.0],
            [0.4, -0.4, 0.0, 0.0],
            [-0.4, 0.4, 0.0, 0.0],
        ],
    )
    features = torch.tensor(
        [
            [0.0, 3.0],
            [2.0, 3.0],
            [4.0, 3.0],
            [6.0, 3.0],
        ],
    )
    correction = ContextDependentCorrection(
        rank=1,
        hidden_dim=4,
        epochs=0,
        lr=1e-2,
        seed=13,
    ).calibrate_from_dataset(_shift_dataset(shift), features)

    assert correction.feature_std is not None
    assert torch.allclose(correction.feature_std[0, 0], features.std(dim=0, unbiased=False)[0])
    assert correction.feature_std[0, 1] == pytest.approx(1e-6)
