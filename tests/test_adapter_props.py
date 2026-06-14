from __future__ import annotations

from enum import Enum
from pathlib import Path

import json
import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import save_file
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from lora_spec.adapter_props import (
    compute_adapter_properties,
    compute_distribution_divergence,
    load_lora_matrices,
    validate_plain_lora_config,
)
from lora_spec.theory import ContinuationContextSet


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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs
    ):
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
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "use_rslora": False}),
        encoding="utf-8",
    )
    matrices = load_lora_matrices(adapter_dir)
    assert set(matrices) == {"layer0", "layer1"}

    base_model = torch.nn.Linear(2, 2, bias=False)
    properties = compute_adapter_properties(adapter_dir, base_model=base_model)
    assert properties.adapted_parameter_count == 12
    assert properties.frobenius_norm_sum == pytest.approx(19.7989898732, rel=1e-5)
    assert properties.max_spectral_norm == pytest.approx(14.1421356237, rel=1e-5)
    assert properties.layer_frobenius_norms["layer0"] == pytest.approx(5.6568542495, rel=1e-5)
    assert properties.layer_scalings == {"layer0": 2.0, "layer1": 2.0}
    assert properties.adapted_parameter_fraction == pytest.approx(3.0)


def test_compute_distribution_divergence_reports_true_per_prompt_values() -> None:
    tokenizer = TinyTokenizer()
    base = TinyLM()
    adapted = TinyLM(delta=torch.tensor([0.3, -0.2, 0.1, -0.1]))
    prompts = ["0 1 2 3", "1 2", "2 3 0"]
    contexts = ContinuationContextSet(
        input_ids=(
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 2, 3]),
            torch.tensor([2, 3, 0]),
        ),
        prompt_lengths=(1, 1, 1),
        continuation_lengths=(2, 2, 2),
        trajectory_model="synthetic_base_target",
        generation_policy="fixed_test_contexts",
    )

    divergence = compute_distribution_divergence(
        base,
        adapted,
        prompts,
        tokenizer=tokenizer,
        adapted_tokenizer=tokenizer,
        batch_size=2,
        continuation_contexts=contexts,
    )

    assert len(divergence.per_prompt_kl) == len(prompts)
    assert len(divergence.per_prompt_js) == len(prompts)
    assert divergence.num_positions == 6
    assert divergence.kl_divergence > 0.0
    assert divergence.js_divergence > 0.0


@pytest.mark.parametrize(
    "config, marker",
    [
        ({"peft_type": "LORA", "use_dora": True}, "use_dora"),
        ({"peft_type": "LORA", "bias": "all"}, "bias=all"),
        ({"peft_type": "LORA", "modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"peft_type": "LOHA"}, "peft_type=LOHA"),
    ],
)
def test_plain_lora_validation_fails_closed(config: dict[str, object], marker: str) -> None:
    with pytest.raises(ValueError, match=marker):
        validate_plain_lora_config(config)


def test_plain_lora_validation_accepts_peft_enum_string_form() -> None:
    class PeftType(Enum):
        LORA = "LORA"

    validate_plain_lora_config({"peft_type": PeftType.LORA})
