from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .artifacts import tokenizers_are_equivalent

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional direct dependency in some environments
    snapshot_download = None


@dataclass
class AdapterProperties:
    frobenius_norm_sum: float
    spectral_norm_sum: float
    max_spectral_norm: float
    adapted_parameter_count: int
    adapted_parameter_fraction: float
    layer_frobenius_norms: dict[str, float]
    layer_spectral_norms: dict[str, float]
    layer_weight_norm_distribution: dict[str, float]
    layer_scalings: dict[str, float]


@dataclass
class CalibrationDivergence:
    kl_divergence: float
    js_divergence: float
    num_positions: int
    per_prompt_kl: list[float]
    per_prompt_js: list[float]


def _resolve_adapter_path(adapter_path: str | Path, revision: str | None = None) -> Path:
    path = Path(adapter_path)
    if path.exists():
        return path
    if snapshot_download is None:
        raise FileNotFoundError(f"Adapter path does not exist locally: {adapter_path}")
    downloaded = snapshot_download(
        repo_id=str(adapter_path),
        revision=revision,
        allow_patterns=["*.json", "*.bin", "*.safetensors"],
    )
    return Path(downloaded)


def _find_adapter_weights(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find adapter weights inside {path}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_safetensors(str(path))
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported adapter checkpoint format at {path}")
    return {str(k): v for k, v in loaded.items() if isinstance(v, torch.Tensor)}


def _normalize_key(key: str, marker: str) -> str:
    key = key.replace(f".{marker}.default.weight", "")
    key = key.replace(f".{marker}.weight", "")
    return key


def load_lora_matrices(
    adapter_path: str | Path,
    revision: str | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    resolved_path = _resolve_adapter_path(adapter_path, revision=revision)
    state_dict = _load_state_dict(_find_adapter_weights(resolved_path))
    matrices_a: dict[str, torch.Tensor] = {}
    matrices_b: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "lora_A" in key:
            matrices_a[_normalize_key(key, "lora_A")] = value.float().cpu()
        elif "lora_B" in key:
            matrices_b[_normalize_key(key, "lora_B")] = value.float().cpu()
    common_keys = sorted(set(matrices_a) & set(matrices_b))
    if not common_keys:
        raise ValueError(f"No LoRA A/B matrices found in adapter checkpoint at {adapter_path}")
    return {key: (matrices_b[key], matrices_a[key]) for key in common_keys}


def _load_adapter_config(adapter_path: str | Path, revision: str | None = None) -> dict[str, Any]:
    resolved_path = _resolve_adapter_path(adapter_path, revision=revision)
    config_path = resolved_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"LoRA adapter config is required to compute effective BA scaling: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter config must contain a JSON object: {config_path}")
    return payload


def _pattern_value(pattern: Any, layer_name: str, default: float) -> float:
    if not isinstance(pattern, dict):
        return default
    if layer_name in pattern:
        return float(pattern[layer_name])
    matches = [
        (str(key), value)
        for key, value in pattern.items()
        if layer_name.endswith(str(key)) or str(key).endswith(layer_name)
    ]
    if not matches:
        return default
    key, value = max(matches, key=lambda item: len(item[0]))
    _ = key
    return float(value)


def _layer_lora_scaling(adapter_config: dict[str, Any], layer_name: str, matrix_a: torch.Tensor) -> float:
    inferred_rank = int(matrix_a.shape[0])
    rank = _pattern_value(adapter_config.get("rank_pattern"), layer_name, float(adapter_config.get("r", inferred_rank)))
    alpha = _pattern_value(adapter_config.get("alpha_pattern"), layer_name, float(adapter_config.get("lora_alpha", rank)))
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank for {layer_name}: {rank}")
    denominator = rank**0.5 if bool(adapter_config.get("use_rslora", False)) else rank
    return float(alpha / denominator)


def _infer_base_parameter_count(
    base_model: str | torch.nn.Module | None,
) -> int:
    if base_model is None:
        return 0
    if isinstance(base_model, torch.nn.Module):
        return sum(parameter.numel() for parameter in base_model.parameters())
    config = AutoConfig.from_pretrained(base_model)
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
        return sum(parameter.numel() for parameter in model.parameters())
    except Exception as exc:
        raise RuntimeError(
            "Could not infer the base-model parameter count without loading weights; "
            "pass an instantiated base model instead"
        ) from exc


def _low_rank_product_singular_values(
    matrix_b: torch.Tensor,
    matrix_a: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    if matrix_b.ndim != 2 or matrix_a.ndim != 2:
        raise ValueError("LoRA matrices must be two-dimensional")
    if matrix_b.shape[1] != matrix_a.shape[0]:
        raise ValueError(
            f"Incompatible LoRA shapes: B={tuple(matrix_b.shape)}, A={tuple(matrix_a.shape)}",
        )
    _, triangular_b = torch.linalg.qr(matrix_b.float(), mode="reduced")
    _, triangular_a_transpose = torch.linalg.qr(matrix_a.float().T, mode="reduced")
    core = (triangular_b @ triangular_a_transpose.T) * scaling
    return torch.linalg.svdvals(core)


def compute_adapter_properties(
    adapter_path: str | Path,
    base_model: str | torch.nn.Module | None = None,
    revision: str | None = None,
) -> AdapterProperties:
    matrices = load_lora_matrices(adapter_path, revision=revision)
    adapter_config = _load_adapter_config(adapter_path, revision=revision)
    layer_frobenius_norms: dict[str, float] = {}
    layer_spectral_norms: dict[str, float] = {}
    layer_weight_norm_distribution: dict[str, float] = {}
    adapted_parameter_count = 0
    layer_scalings: dict[str, float] = {}

    for layer_name, (matrix_b, matrix_a) in matrices.items():
        scaling = _layer_lora_scaling(adapter_config, layer_name, matrix_a)
        singular_values = _low_rank_product_singular_values(matrix_b, matrix_a, scaling)
        adapted_parameter_count += matrix_b.numel() + matrix_a.numel()
        fro_value = float(torch.linalg.vector_norm(singular_values).item())
        spectral_value = float(singular_values.max().item())
        dense_parameter_count = matrix_b.shape[0] * matrix_a.shape[1]
        weight_norm = float(fro_value / max(dense_parameter_count**0.5, 1.0))
        layer_frobenius_norms[layer_name] = fro_value
        layer_spectral_norms[layer_name] = spectral_value
        layer_weight_norm_distribution[layer_name] = weight_norm
        layer_scalings[layer_name] = scaling

    base_parameter_count = _infer_base_parameter_count(base_model)
    fraction = (
        adapted_parameter_count / base_parameter_count
        if base_parameter_count > 0
        else float("nan")
    )
    return AdapterProperties(
        frobenius_norm_sum=float(sum(layer_frobenius_norms.values())),
        spectral_norm_sum=float(sum(layer_spectral_norms.values())),
        max_spectral_norm=float(max(layer_spectral_norms.values())),
        adapted_parameter_count=adapted_parameter_count,
        adapted_parameter_fraction=float(fraction),
        layer_frobenius_norms=layer_frobenius_norms,
        layer_spectral_norms=layer_spectral_norms,
        layer_weight_norm_distribution=layer_weight_norm_distribution,
        layer_scalings=layer_scalings,
    )


def _resolve_model(
    model_or_name: str | PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase | None = None,
    adapter_path: str | None = None,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if isinstance(model_or_name, PreTrainedModel):
        if tokenizer is None:
            raise ValueError("Tokenizer must be supplied when passing a model instance")
        model = model_or_name
        tok = tokenizer
    else:
        tok = AutoTokenizer.from_pretrained(model_or_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_or_name)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model = model.to(target_device).eval()
    return model, tok


def _prompt_batches(prompts: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for prompt in prompts:
        batch.append(prompt)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _per_prompt_token_divergence(
    base_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[list[float], list[float], int]:
    base_log_probs = F.log_softmax(base_logits[:, :-1, :], dim=-1)
    adapted_log_probs = F.log_softmax(adapted_logits[:, :-1, :], dim=-1)
    base_probs = base_log_probs.exp()
    adapted_probs = adapted_log_probs.exp()

    mask = attention_mask[:, 1:].bool()
    kl = torch.sum(adapted_probs * (adapted_log_probs - base_log_probs), dim=-1)
    midpoint = 0.5 * (adapted_probs + base_probs)
    midpoint_log = torch.log(midpoint.clamp_min(1e-12))
    js_left = torch.sum(adapted_probs * (adapted_log_probs - midpoint_log), dim=-1)
    js_right = torch.sum(base_probs * (base_log_probs - midpoint_log), dim=-1)
    js = 0.5 * (js_left + js_right)

    per_prompt_kl: list[float] = []
    per_prompt_js: list[float] = []
    total_positions = 0
    for batch_index in range(mask.shape[0]):
        prompt_mask = mask[batch_index]
        positions = int(prompt_mask.sum().item())
        if positions == 0:
            continue
        total_positions += positions
        per_prompt_kl.append(float(kl[batch_index].masked_select(prompt_mask).mean().item()))
        per_prompt_js.append(float(js[batch_index].masked_select(prompt_mask).mean().item()))
    return per_prompt_kl, per_prompt_js, total_positions


@torch.inference_mode()
def compute_distribution_divergence(
    base_model: str | PreTrainedModel,
    adapted_model: str | PreTrainedModel,
    prompts: list[str],
    tokenizer: PreTrainedTokenizerBase | None = None,
    adapted_tokenizer: PreTrainedTokenizerBase | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
) -> CalibrationDivergence:
    base, base_tokenizer = _resolve_model(base_model, tokenizer=tokenizer, device=device)
    adapted, adapted_tokenizer = _resolve_model(
        adapted_model,
        tokenizer=adapted_tokenizer or tokenizer,
        device=device or next(base.parameters()).device,
    )
    if not tokenizers_are_equivalent(base_tokenizer, adapted_tokenizer, prompts):
        raise ValueError(
            "Base and adapted tokenizers must be exactly equivalent for KL/JSD comparison"
        )

    per_prompt_kl: list[float] = []
    per_prompt_js: list[float] = []
    total_positions = 0
    weighted_kl_sum = 0.0
    weighted_js_sum = 0.0

    for batch_prompts in _prompt_batches(prompts, batch_size):
        encoded = base_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(next(base.parameters()).device) for key, value in encoded.items()}
        base_outputs = base(**encoded)
        adapted_outputs = adapted(**encoded)
        batch_prompt_kl, batch_prompt_js, positions = _per_prompt_token_divergence(
            base_outputs.logits.float(),
            adapted_outputs.logits.float(),
            encoded["attention_mask"],
        )
        total_positions += positions
        per_prompt_kl.extend(batch_prompt_kl)
        per_prompt_js.extend(batch_prompt_js)
        prompt_positions = [
            int(encoded["attention_mask"][index, 1:].sum().item())
            for index in range(encoded["attention_mask"].shape[0])
            if int(encoded["attention_mask"][index, 1:].sum().item()) > 0
        ]
        weighted_kl_sum += sum(value * prompt_count for value, prompt_count in zip(batch_prompt_kl, prompt_positions))
        weighted_js_sum += sum(value * prompt_count for value, prompt_count in zip(batch_prompt_js, prompt_positions))

    if not per_prompt_kl:
        raise ValueError("Prompt set must not be empty")
    return CalibrationDivergence(
        kl_divergence=float(weighted_kl_sum / total_positions),
        js_divergence=float(weighted_js_sum / total_positions),
        num_positions=total_positions,
        per_prompt_kl=per_prompt_kl,
        per_prompt_js=per_prompt_js,
    )


def read_adapter_metadata(
    adapter_path: str | Path,
    revision: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_adapter_path(adapter_path, revision=revision)
    metadata_path = resolved / "adapter_config.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
