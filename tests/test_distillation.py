from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from lora_spec import distillation
from lora_spec.distillation import DistillationConfig, train_micro_lora_adapter
from lora_spec.theory import ContinuationContextSet


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def save_pretrained(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModel(torch.nn.Module):
    def __init__(self, logits: tuple[float, float]) -> None:
        super().__init__()
        self.logit_bias = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> object:
        _ = attention_mask
        logits = self.logit_bias.view(1, 1, 2).expand(input_ids.shape[0], input_ids.shape[1], 2)
        return type("Output", (), {"logits": logits})()

    def save_pretrained(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.logit_bias.detach(), directory / "adapter.bin")


def test_distillation_keeps_epoch_zero_when_training_worsens_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    teacher = FakeModel((1.0, -1.0))
    student = FakeModel((0.0, 0.0))
    tokenizer = FakeTokenizer()
    contexts = ContinuationContextSet(
        input_ids=(torch.tensor([0, 1], dtype=torch.long),),
        prompt_lengths=(1,),
        continuation_lengths=(1,),
        trajectory_model="adapted_target",
        generation_policy="fixed",
    )

    monkeypatch.setattr(distillation, "tokenizers_are_equivalent", lambda *args: True)
    monkeypatch.setattr(
        distillation, "build_continuation_contexts", lambda *args, **kwargs: contexts
    )
    monkeypatch.setattr(
        distillation,
        "_load_model",
        lambda model, device, torch_dtype: teacher if model == "target" else student,
    )
    monkeypatch.setattr(distillation, "_infer_target_modules", lambda model: ["logit_bias"])
    monkeypatch.setattr(distillation, "LoraConfig", lambda **kwargs: object())
    monkeypatch.setattr(distillation, "get_peft_model", lambda model, config: model)
    validation_values = iter([(0.1, 1), (0.2, 1)])
    monkeypatch.setattr(distillation, "_evaluate_context_kl", lambda *args: next(validation_values))

    checkpoint = train_micro_lora_adapter(
        draft_model="draft",
        target_model="target",
        prompts=["train"],
        validation_prompts=["validation"],
        output_dir=tmp_path,
        config=DistillationConfig(
            draft_lora_rank=1,
            learning_rate=0.1,
            batch_size=1,
            epochs=1,
            max_length=4,
            continuation_tokens=1,
        ),
        draft_tokenizer=tokenizer,  # type: ignore[arg-type]
        target_tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    metrics = json.loads((checkpoint / "training_metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_epoch"] == 0
    assert metrics["initial_validation_mean_token_kl"] == pytest.approx(0.1)
    assert metrics["best_validation_mean_token_kl"] == pytest.approx(0.1)
    assert metrics["improved_over_initial"] is False
