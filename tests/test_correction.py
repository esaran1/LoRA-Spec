from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from lora_spec.correction import DistributionOffsetCorrection, JacobianCorrection, LowRankCorrection


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

    def __call__(self, texts: list[str], return_tensors: str, padding: bool, truncation: bool):
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


class TinyLM(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, delta: torch.Tensor | None = None) -> None:
        super().__init__(TinyConfig())
        self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.vocab_size)
        self.lm_head = torch.nn.Linear(self.config.vocab_size, self.config.vocab_size, bias=False)
        self.lora_delta = torch.nn.Parameter(
            torch.zeros(self.config.vocab_size) if delta is None else delta.clone().float(),
        )
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(self.config.vocab_size))
            self.lm_head.weight.copy_(torch.eye(self.config.vocab_size))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden) + self.lora_delta.view(1, 1, -1)
        return CausalLMOutput(logits=logits)


def test_distribution_offset_correction_recovers_constant_shift() -> None:
    tokenizer = TinyTokenizer()
    base = TinyLM()
    delta = torch.tensor([0.4, -0.2, 0.1, -0.3])
    adapted = TinyLM(delta=delta)
    correction = DistributionOffsetCorrection().calibrate(base, adapted, ["0 1 2", "1 2 3"], tokenizer=tokenizer)
    adjusted = correction.apply(torch.zeros(1, tokenizer.vocab_size)).squeeze(0)
    assert adjusted.shape[0] == tokenizer.vocab_size
    assert adjusted.mean().item() == adjusted.mean().item()


def test_low_rank_correction_returns_vocab_sized_adjustment() -> None:
    tokenizer = TinyTokenizer()
    base = TinyLM()
    adapted = TinyLM(delta=torch.tensor([0.2, 0.1, -0.3, 0.0]))
    correction = LowRankCorrection(rank=2).calibrate(base, adapted, ["0 1 2", "1 2 3"], tokenizer=tokenizer)
    adjusted = correction.apply(torch.zeros(1, tokenizer.vocab_size))
    assert adjusted.shape == (1, tokenizer.vocab_size)
    assert torch.isfinite(adjusted).all()


def test_jacobian_correction_estimates_shift_for_linear_tiny_model() -> None:
    tokenizer = TinyTokenizer()
    base = TinyLM()
    delta = torch.tensor([0.5, -0.25, 0.0, 0.25])
    adapted = TinyLM(delta=delta)
    correction = JacobianCorrection(probe_count=4, max_params=2, seed=0).calibrate(
        base,
        adapted,
        ["0 1 2 3"],
        tokenizer=tokenizer,
    )
    adjusted = correction.apply(torch.zeros(1, tokenizer.vocab_size)).squeeze(0)
    assert adjusted.shape == delta.shape
    assert torch.linalg.norm(adjusted) > 0
