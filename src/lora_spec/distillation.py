from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .artifacts import tokenizers_are_equivalent
from .theory import ContinuationContextSet, build_continuation_contexts
from .utils import resolve_torch_dtype, set_seed


@dataclass
class DistillationConfig:
    draft_lora_rank: int
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int = 512
    save_every_epoch: bool = True
    seed: int = 7
    torch_dtype: str = "auto"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    continuation_tokens: int = 16

    def __post_init__(self) -> None:
        if self.draft_lora_rank < 1:
            raise ValueError("draft_lora_rank must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.continuation_tokens < 1 or self.continuation_tokens >= self.max_length:
            raise ValueError("continuation_tokens must lie in [1, max_length)")


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
    torch_dtype: torch.dtype,
) -> PreTrainedModel:
    if isinstance(model_or_name, PreTrainedModel):
        return model_or_name.to(device)
    return AutoModelForCausalLM.from_pretrained(
        model_or_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)


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


def _collate_contexts(
    contexts: ContinuationContextSet,
    indices: list[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [contexts.input_ids[index] for index in indices]
    maximum_length = max(int(sequence.numel()) for sequence in sequences)
    input_ids = torch.full((len(indices), maximum_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for row, (index, sequence) in enumerate(zip(indices, sequences)):
        length = int(sequence.numel())
        input_ids[row, :length] = sequence
        attention_mask[row, :length] = 1
        first = contexts.prompt_lengths[index] - 1
        count = contexts.continuation_lengths[index]
        loss_mask[row, first : first + count] = 1.0
    return input_ids, attention_mask, loss_mask


def _full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    return (token_kl * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


@torch.no_grad()
def _evaluate_context_kl(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    contexts: ContinuationContextSet,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    was_training = student_model.training
    student_model.eval()
    loss_sum = 0.0
    token_count = 0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Draft tokenizer requires a pad or EOS token")
    for start in range(0, len(contexts.input_ids), batch_size):
        indices = list(range(start, min(start + batch_size, len(contexts.input_ids))))
        input_ids, attention_mask, loss_mask = _collate_contexts(
            contexts,
            indices,
            pad_token_id,
        )
        encoded = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
        mask = loss_mask.to(device)
        teacher_logits = teacher_model(**encoded).logits.float()
        student_logits = student_model(**encoded).logits.float()
        loss = _full_vocab_kl(student_logits, teacher_logits, mask)
        masked_tokens = int(mask.sum().item())
        loss_sum += float(loss.item()) * masked_tokens
        token_count += masked_tokens
    if was_training:
        student_model.train()
    return loss_sum / max(token_count, 1), token_count


def train_micro_lora_adapter(
    draft_model: str | PreTrainedModel,
    target_model: str | PreTrainedModel,
    prompts: list[str] | Dataset[str],
    validation_prompts: list[str] | Dataset[str],
    output_dir: str | Path,
    config: DistillationConfig,
    draft_tokenizer: PreTrainedTokenizerBase | None = None,
    target_tokenizer: PreTrainedTokenizerBase | None = None,
    adapter_path: str | None = None,
) -> Path:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(config.torch_dtype, device=device)
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
    validation_dataset = (
        validation_prompts
        if isinstance(validation_prompts, Dataset)
        else PromptDataset(validation_prompts)
    )
    prompt_probe = [dataset[index] for index in range(len(dataset))]
    validation_probe = [validation_dataset[index] for index in range(len(validation_dataset))]
    if not prompt_probe or not validation_probe:
        raise ValueError("Training and validation prompt splits must both be non-empty")
    if not all(isinstance(prompt, str) for prompt in prompt_probe + validation_probe):
        raise TypeError("Distillation datasets must return prompt strings")
    if not tokenizers_are_equivalent(
        draft_tokenizer,
        target_tokenizer,
        prompt_probe + validation_probe,
    ):
        raise ValueError(
            "Draft and target tokenizers must be tokenization-compatible for full-vocabulary KL distillation",
        )

    teacher_model = _load_model(target_model, device=device, torch_dtype=torch_dtype).eval()
    if adapter_path is not None:
        teacher_model = PeftModel.from_pretrained(teacher_model, adapter_path).to(device).eval()
    teacher_model.requires_grad_(False)
    contexts = build_continuation_contexts(
        teacher_model,
        target_tokenizer,
        prompt_probe,
        max_new_tokens=config.continuation_tokens,
        trajectory_model="adapted_target",
        max_prompt_length=config.max_length - config.continuation_tokens,
    )
    validation_contexts = build_continuation_contexts(
        teacher_model,
        target_tokenizer,
        validation_probe,
        max_new_tokens=config.continuation_tokens,
        trajectory_model="adapted_target",
        max_prompt_length=config.max_length - config.continuation_tokens,
    )

    student_base = _load_model(draft_model, device=device, torch_dtype=torch_dtype)
    lora_config = LoraConfig(
        r=config.draft_lora_rank,
        lora_alpha=max(config.draft_lora_rank * 2, 8),
        target_modules=_infer_target_modules(student_base),
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    student_model = get_peft_model(student_base, lora_config).train()

    data_generator = torch.Generator().manual_seed(config.seed)
    dataloader = DataLoader(
        list(range(len(dataset))),
        batch_size=config.batch_size,
        shuffle=True,
        generator=data_generator,
    )
    trainable_parameters = [
        parameter for parameter in student_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    save_root = Path(output_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    epoch_metrics: list[dict[str, float | int]] = []
    optimizer_steps = 0
    initial_validation_kl, initial_validation_tokens = _evaluate_context_kl(
        student_model,
        teacher_model,
        validation_contexts,
        draft_tokenizer,
        config.batch_size,
        device,
    )
    best_validation_kl = initial_validation_kl
    best_epoch = 0
    best_dir = save_root / "best"
    student_model.save_pretrained(best_dir)
    draft_tokenizer.save_pretrained(best_dir)
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_token_count = 0
        for step, batch_indices in enumerate(dataloader):
            indices = [int(index) for index in batch_indices]
            pad_token_id = draft_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = draft_tokenizer.eos_token_id
            if pad_token_id is None:
                raise ValueError("Draft tokenizer requires a pad or EOS token")
            input_ids, attention_mask, loss_mask = _collate_contexts(
                contexts,
                indices,
                pad_token_id,
            )
            encoded_inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }
            loss_mask = loss_mask.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(**encoded_inputs).logits.float()
            student_logits = student_model(**encoded_inputs).logits.float()
            loss = _full_vocab_kl(student_logits, teacher_logits, loss_mask)
            masked_tokens = int(loss_mask.sum().item())
            epoch_loss_sum += float(loss.detach().item()) * masked_tokens
            epoch_token_count += masked_tokens
            accumulation_window_start = (
                step // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            accumulation_window_size = min(
                config.gradient_accumulation_steps,
                len(dataloader) - accumulation_window_start,
            )
            scaled_loss = loss / accumulation_window_size

            scaled_loss.backward()
            should_step = (step + 1) % config.gradient_accumulation_steps == 0 or step + 1 == len(
                dataloader
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        validation_kl, validation_tokens = _evaluate_context_kl(
            student_model,
            teacher_model,
            validation_contexts,
            draft_tokenizer,
            config.batch_size,
            device,
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "mean_token_kl": epoch_loss_sum / max(epoch_token_count, 1),
                "validation_mean_token_kl": validation_kl,
                "validation_tokens": validation_tokens,
                "supervised_tokens": epoch_token_count,
                "optimizer_steps_cumulative": optimizer_steps,
            }
        )

        if validation_kl < best_validation_kl:
            best_validation_kl = validation_kl
            best_epoch = epoch + 1
            student_model.save_pretrained(best_dir)
            draft_tokenizer.save_pretrained(best_dir)

        if config.save_every_epoch:
            epoch_dir = save_root / f"epoch_{epoch + 1}"
            student_model.save_pretrained(epoch_dir)
            draft_tokenizer.save_pretrained(epoch_dir)

    final_dir = save_root / "final"
    student_model.save_pretrained(final_dir)
    draft_tokenizer.save_pretrained(final_dir)
    context_payload = {
        "training": contexts.to_dict(),
        "validation": validation_contexts.to_dict(),
    }
    metrics_payload = {
        "config": config.__dict__,
        "context_sha256": contexts.sha256(),
        "validation_context_sha256": validation_contexts.sha256(),
        "epochs": epoch_metrics,
        "optimizer_steps": optimizer_steps,
        "initial_validation_mean_token_kl": initial_validation_kl,
        "initial_validation_tokens": initial_validation_tokens,
        "best_epoch": best_epoch,
        "best_validation_mean_token_kl": best_validation_kl,
        "improved_over_initial": best_epoch > 0,
        "best_checkpoint": str(best_dir.resolve()),
    }
    for checkpoint_dir in (best_dir, final_dir):
        (checkpoint_dir / "distillation_contexts.json").write_text(
            json.dumps(context_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (checkpoint_dir / "training_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return best_dir
