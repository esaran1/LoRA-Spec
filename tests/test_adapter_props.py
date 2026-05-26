from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import save_file
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from lora_spec.adapter_props import compute_adapter_properties, compute_distribution_divergence, load_lora_matrices


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
            tokenized.append([int(piece) % self.vocab_size for piece in text.split()])
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
        self.delta = torch.nn.Parameter(
            torch.zeros(self.config.vocab_size) if delta is None else delta.clone().float(),
        )
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(self.config.vocab_size))
            self.lm_head.weight.copy_(torch.eye(self.config.vocab_size))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden) + self.delta.view(1, 1, -1)
        return CausalLMOutput(logits=logits)


def test_load_lora_matrices_and_compute_properties(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_file(
        {
            "layer0.lora_A.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "layer0.lora_B.weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "layer1.lora_A.weight": torch.tensor([[1.0, 1.0]]),
            "layer1.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    matrices = load_lora_matrices(adapter_dir)
    assert set(matrices) == {"layer0", "layer1"}

    base_model = torch.nn.Linear(2, 2, bias=False)
    properties = compute_adapter_properties(adapter_dir, base_model=base_model)
    assert properties.adapted_parameter_count == 10
    assert properties.frobenius_norm_sum == pytest.approx(9.0710678118, rel=1e-5)
    assert properties.max_spectral_norm == pytest.approx(5.0, rel=1e-5)
    assert properties.layer_frobenius_norms["layer0"] == pytest.approx(2.8284271247, rel=1e-5)
    assert properties.adapted_parameter_fraction == pytest.approx(2.5)


def test_compute_distribution_divergence_reports_true_per_prompt_values() -> None:
    tokenizer = TinyTokenizer()
    base = TinyLM()
    adapted = TinyLM(delta=torch.tensor([0.3, -0.2, 0.1, -0.1]))
    prompts = ["0 1 2 3", "1 2", "2 3 0"]

    divergence = compute_distribution_divergence(
        base,
        adapted,
        prompts,
        tokenizer=tokenizer,
        adapted_tokenizer=tokenizer,
        batch_size=2,
    )

    assert len(divergence.per_prompt_kl) == len(prompts)
    assert len(divergence.per_prompt_js) == len(prompts)
    assert divergence.num_positions == 6
    assert divergence.kl_divergence > 0.0
    assert divergence.js_divergence > 0.0
