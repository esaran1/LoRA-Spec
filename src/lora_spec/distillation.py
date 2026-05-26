from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class DistillationConfig:
    draft_lora_rank: int
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int = 512
    save_every_epoch: bool = True
    seed: int = 7


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_or_name: str | PreTrainedModel,
    device: str | torch.device,
) -> PreTrainedModel:
    if isinstance(model_or_name, PreTrainedModel):
        return model_or_name.to(device)
    return AutoModelForCausalLM.from_pretrained(model_or_name).to(device)


def _infer_target_modules(model: PreTrainedModel) -> list[str]:
    preferred = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    discovered = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in preferred
        }
    )
    if discovered:
        return discovered
    return sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and module.weight.ndim == 2
        }
    )[:8]


def _collate_prompts(
    prompts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits[:, :-1, :], dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    mask = attention_mask[:, 1:].float()
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)


def train_micro_lora_adapter(
    draft_model: str | PreTrainedModel,
    target_model: str | PreTrainedModel,
    prompts: list[str] | Dataset[str],
    output_dir: str | Path,
    config: DistillationConfig,
    draft_tokenizer: PreTrainedTokenizerBase | None = None,
    target_tokenizer: PreTrainedTokenizerBase | None = None,
    adapter_path: str | None = None,
) -> Path:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(prompts, Dataset):
        dataset = prompts
    else:
        dataset = PromptDataset(prompts)

    if isinstance(draft_model, str):
        draft_tokenizer = draft_tokenizer or _load_tokenizer(draft_model)
    if isinstance(target_model, str):
        target_tokenizer = target_tokenizer or _load_tokenizer(target_model)
    if draft_tokenizer is None or target_tokenizer is None:
        raise ValueError("Tokenizers must be available for both draft and target models")
    if draft_tokenizer.vocab_size != target_tokenizer.vocab_size:
        raise ValueError("Draft and target tokenizers must share the same vocabulary")

    teacher_model = _load_model(target_model, device=device).eval()
    if adapter_path is not None:
        teacher_model = PeftModel.from_pretrained(teacher_model, adapter_path).to(device).eval()

    student_base = _load_model(draft_model, device=device)
    lora_config = LoraConfig(
        r=config.draft_lora_rank,
        lora_alpha=max(config.draft_lora_rank * 2, 8),
        target_modules=_infer_target_modules(student_base),
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    student_model = get_peft_model(student_base, lora_config).train()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate)
    save_root = Path(output_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        for prompt_batch in dataloader:
            batch = list(prompt_batch) if isinstance(prompt_batch, Iterable) else [prompt_batch]
            encoded_student = _collate_prompts(batch, draft_tokenizer, max_length=config.max_length)
            encoded_student = {key: tensor.to(device) for key, tensor in encoded_student.items()}
            encoded_teacher = _collate_prompts(batch, target_tokenizer, max_length=config.max_length)
            encoded_teacher = {key: tensor.to(device) for key, tensor in encoded_teacher.items()}

            with torch.no_grad():
                teacher_logits = teacher_model(**encoded_teacher).logits.float()
            student_logits = student_model(**encoded_student).logits.float()
            loss = _full_vocab_kl(student_logits, teacher_logits, encoded_student["attention_mask"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if config.save_every_epoch:
            epoch_dir = save_root / f"epoch_{epoch + 1}"
            student_model.save_pretrained(epoch_dir)
            draft_tokenizer.save_pretrained(epoch_dir)

    final_dir = save_root / "final"
    student_model.save_pretrained(final_dir)
    draft_tokenizer.save_pretrained(final_dir)
    return final_dir
