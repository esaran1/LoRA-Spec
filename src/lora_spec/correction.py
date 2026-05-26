from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class Correction(Protocol):
    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> "Correction":
        ...

    def apply(self, draft_logits: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class CalibrationBundle:
    base_log_probs: torch.Tensor
    adapted_log_probs: torch.Tensor
    tokenizer: PreTrainedTokenizerBase


def _load_model_and_tokenizer(
    model_or_name: str | PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase | None,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if isinstance(model_or_name, PreTrainedModel):
        if tokenizer is None:
            raise ValueError("Tokenizer is required when passing an instantiated model")
        model = model_or_name.eval()
        tok = tokenizer
    else:
        tok = AutoTokenizer.from_pretrained(model_or_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_or_name).eval()
    if device is not None:
        model = model.to(device)
    return model, tok


def _batch_prompts(prompts: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for prompt in prompts:
        batch.append(prompt)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@torch.inference_mode()
def _collect_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int = 2,
) -> torch.Tensor:
    device = next(model.parameters()).device
    per_position: list[torch.Tensor] = []
    for batch_prompts in _batch_prompts(prompts, batch_size):
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        outputs = model(**encoded)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
        mask = encoded["attention_mask"][:, 1:].bool()
        for batch_index in range(log_probs.shape[0]):
            per_position.append(log_probs[batch_index][mask[batch_index]])
    if not per_position:
        raise ValueError("Calibration prompts must not be empty")
    return torch.cat(per_position, dim=0)


def _collect_calibration_bundle(
    base_model: str | PreTrainedModel,
    adapted_model: str | PreTrainedModel,
    prompts: list[str],
    tokenizer: PreTrainedTokenizerBase | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
) -> CalibrationBundle:
    base, tok = _load_model_and_tokenizer(base_model, tokenizer=tokenizer, device=device)
    adapted, adapted_tok = _load_model_and_tokenizer(
        adapted_model,
        tokenizer=tokenizer or tok,
        device=device or next(base.parameters()).device,
    )
    if tok.vocab_size != adapted_tok.vocab_size:
        raise ValueError("Correction calibration requires matching tokenizers")
    base_log_probs = _collect_log_probs(base, tok, prompts, batch_size=batch_size)
    adapted_log_probs = _collect_log_probs(adapted, adapted_tok, prompts, batch_size=batch_size)
    return CalibrationBundle(
        base_log_probs=base_log_probs,
        adapted_log_probs=adapted_log_probs,
        tokenizer=tok,
    )


class DistributionOffsetCorrection:
    def __init__(self) -> None:
        self.correction_vector: torch.Tensor | None = None

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> "DistributionOffsetCorrection":
        bundle = _collect_calibration_bundle(base_model, adapted_model, prompts, tokenizer=tokenizer)
        self.correction_vector = (
            bundle.adapted_log_probs - bundle.base_log_probs
        ).mean(dim=0)
        return self

    def apply(self, draft_logits: torch.Tensor) -> torch.Tensor:
        if self.correction_vector is None:
            raise RuntimeError("Correction must be calibrated before apply")
        return draft_logits + self.correction_vector.to(draft_logits.device, draft_logits.dtype)


class LowRankCorrection:
    def __init__(self, rank: int = 8) -> None:
        self.rank = rank
        self.projection_basis: torch.Tensor | None = None
        self.mapping: torch.Tensor | None = None
        self.mean_base: torch.Tensor | None = None
        self.mean_diff: torch.Tensor | None = None

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> "LowRankCorrection":
        bundle = _collect_calibration_bundle(base_model, adapted_model, prompts, tokenizer=tokenizer)
        base = bundle.base_log_probs
        diff = bundle.adapted_log_probs - bundle.base_log_probs
        self.mean_base = base.mean(dim=0)
        self.mean_diff = diff.mean(dim=0)
        centered_base = base - self.mean_base
        centered_diff = diff - self.mean_diff
        q = min(self.rank, centered_base.shape[0], centered_base.shape[1])
        _, _, basis = torch.pca_lowrank(centered_base, q=q)
        projected = centered_base @ basis
        self.projection_basis = basis
        self.mapping = torch.linalg.pinv(projected) @ centered_diff
        return self

    def apply(self, draft_logits: torch.Tensor) -> torch.Tensor:
        if any(
            component is None
            for component in (self.projection_basis, self.mapping, self.mean_base, self.mean_diff)
        ):
            raise RuntimeError("Correction must be calibrated before apply")
        centered = draft_logits - self.mean_base.to(draft_logits.device, draft_logits.dtype)
        low_rank = (
            centered
            @ self.projection_basis.to(draft_logits.device, draft_logits.dtype)
            @ self.mapping.to(draft_logits.device, draft_logits.dtype)
        )
        return draft_logits + low_rank + self.mean_diff.to(draft_logits.device, draft_logits.dtype)


class JacobianCorrection:
    def __init__(self, probe_count: int = 8, max_params: int = 12, seed: int = 7) -> None:
        self.probe_count = probe_count
        self.max_params = max_params
        self.seed = seed
        self.correction_vector: torch.Tensor | None = None

    def _select_parameters(
        self,
        base_model: PreTrainedModel,
        adapted_model: PreTrainedModel,
    ) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        base_params = dict(base_model.named_parameters())
        selected: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for name, parameter in adapted_model.named_parameters():
            if "lora_" in name or "lm_head" in name or "layers." in name:
                base_value = base_params.get(name)
                if base_value is None:
                    delta = parameter.detach().float().clone()
                else:
                    delta = (parameter.detach().float() - base_value.detach().float()).clone()
                if torch.count_nonzero(delta).item() == 0:
                    continue
                selected.append((parameter, delta.to(parameter.device)))
        if not selected:
            raise ValueError("No non-zero parameter deltas were found for Jacobian correction")
        return selected[-self.max_params :]

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> "JacobianCorrection":
        base, tok = _load_model_and_tokenizer(base_model, tokenizer=tokenizer)
        adapted, _ = _load_model_and_tokenizer(
            adapted_model,
            tokenizer=tokenizer or tok,
            device=next(base.parameters()).device,
        )
        selected = self._select_parameters(base, adapted)
        vocab_size = adapted.get_output_embeddings().weight.shape[0]
        generator = torch.Generator(device=next(adapted.parameters()).device)
        generator.manual_seed(self.seed)
        probes = torch.randn(
            self.probe_count,
            vocab_size,
            generator=generator,
            device=next(adapted.parameters()).device,
        )
        probes = F.normalize(probes, dim=-1)

        observed_shifts: list[torch.Tensor] = []
        for batch_prompts in _batch_prompts(prompts, batch_size=1):
            encoded = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            encoded = {name: tensor.to(next(adapted.parameters()).device) for name, tensor in encoded.items()}
            outputs = adapted(**encoded)
            logits = outputs.logits[:, :-1, :].float()
            scalars: list[torch.Tensor] = []
            for probe in probes:
                scalar = (logits * probe.view(1, 1, -1)).mean()
                gradients = torch.autograd.grad(
                    scalar,
                    [parameter for parameter, _ in selected],
                    retain_graph=True,
                    allow_unused=True,
                )
                shift = torch.zeros((), device=logits.device)
                for gradient, (_, delta) in zip(gradients, selected):
                    if gradient is None:
                        continue
                    shift = shift + torch.sum(gradient.float() * delta)
                scalars.append(shift)
            observed_shifts.append(torch.stack(scalars))

        mean_shift = torch.stack(observed_shifts).mean(dim=0)
        probe_matrix = probes.float()
        solution = torch.linalg.pinv(probe_matrix) @ mean_shift.unsqueeze(-1)
        self.correction_vector = solution.squeeze(-1).detach().cpu()
        return self

    def apply(self, draft_logits: torch.Tensor) -> torch.Tensor:
        if self.correction_vector is None:
            raise RuntimeError("Correction must be calibrated before apply")
        return draft_logits + self.correction_vector.to(draft_logits.device, draft_logits.dtype)
