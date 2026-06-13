from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import matplotlib.pyplot as plt
import torch
from torch.func import functional_call, jvp

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional for CPU-only theory utilities
    PeftModel = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LogitShiftDataset:
    shift_matrix: torch.Tensor
    base_logits_matrix: torch.Tensor
    adapted_logits_matrix: torch.Tensor
    hidden_state_matrix: torch.Tensor | None
    prompt_indices: list[int]
    token_positions: list[int]
    vocabulary_size: int


@dataclass
class SpectralAnalysisResult:
    singular_values: list[float]
    cumulative_energy: list[float]
    participation_ratio: float
    stable_rank: float
    effective_rank_95: int
    effective_rank_99: int

    def save_spectrum_plot(self, output_path: str | Path, title: str = "Logit Shift Spectrum") -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        indices = list(range(1, len(self.singular_values) + 1))
        axes[0].plot(indices, self.singular_values, marker="o", linewidth=1.5)
        axes[0].set_title(title)
        axes[0].set_xlabel("Singular value index")
        axes[0].set_ylabel("Magnitude")
        axes[1].plot(indices, self.cumulative_energy, marker="o", linewidth=1.5)
        axes[1].axhline(0.95, linestyle="--", color="black", linewidth=1)
        axes[1].axhline(0.99, linestyle="--", color="black", linewidth=1)
        axes[1].set_title("Cumulative Spectral Energy")
        axes[1].set_xlabel("Singular value index")
        axes[1].set_ylabel("Energy fraction")
        axes[1].set_ylim(0.0, 1.02)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return path


@dataclass
class NonlinearityResidualResult:
    frobenius_fraction: float
    relative_row_mean: float
    cosine_similarity_mean: float


@dataclass
class SubspaceOverlapResult:
    rank: int
    principal_angles_degrees: list[float]
    cosines: list[float]
    mean_cosine: float
    chordal_distance: float


def _load_model_and_tokenizer(
    model_or_name: str | "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    adapter_path: str | None = None,
    device: str | torch.device | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    except ImportError as exc:  # pragma: no cover - only needed for model-backed paths
        raise ImportError("transformers must be installed for model-backed theory utilities") from exc
    if isinstance(model_or_name, PreTrainedModel):
        if tokenizer is None:
            raise ValueError("Tokenizer is required when passing an instantiated model")
        model = model_or_name.eval()
        tok = tokenizer
    else:
        tok = AutoTokenizer.from_pretrained(model_or_name, use_fast=True)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_or_name).eval()
    if adapter_path:
        if PeftModel is None:
            raise ImportError("peft must be installed to load an adapter-backed model")
        model = PeftModel.from_pretrained(model, adapter_path).eval()
    if device is not None:
        model = model.to(device)
    return model, tok


def _batch_prompts(prompts: Iterable[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    batch: list[str] = []
    start_index = 0
    for prompt_index, prompt in enumerate(prompts):
        if not batch:
            start_index = prompt_index
        batch.append(prompt)
        if len(batch) == batch_size:
            yield start_index, batch
            batch = []
    if batch:
        yield start_index, batch


def _normalize_parameter_name(name: str) -> str:
    normalized = name
    for prefix in ("base_model.model.", "model.", "module."):
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


@torch.inference_mode()
def collect_logit_shift_dataset(
    base_model: str | "PreTrainedModel",
    adapted_model: str | "PreTrainedModel",
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    adapted_tokenizer: "PreTrainedTokenizerBase" | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    collect_hidden_states: bool = True,
) -> LogitShiftDataset:
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty")
    base, base_tokenizer = _load_model_and_tokenizer(base_model, tokenizer=tokenizer, device=device)
    adapted, adapted_tokenizer = _load_model_and_tokenizer(
        adapted_model,
        tokenizer=adapted_tokenizer or tokenizer or base_tokenizer,
        device=device or next(base.parameters()).device,
    )
    if base_tokenizer.vocab_size != adapted_tokenizer.vocab_size:
        raise ValueError("Base and adapted models must share the same tokenizer vocabulary")
    base_tokenizer.padding_side = "right"
    adapted_tokenizer.padding_side = "right"

    shift_rows: list[torch.Tensor] = []
    base_rows: list[torch.Tensor] = []
    adapted_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    prompt_indices: list[int] = []
    token_positions: list[int] = []
    model_device = next(base.parameters()).device

    for prompt_offset, batch_prompts in _batch_prompts(calibration_prompts, batch_size):
        encoded = base_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        base_outputs = base(**encoded, output_hidden_states=collect_hidden_states)
        adapted_outputs = adapted(**encoded)
        base_logits = base_outputs.logits[:, :-1, :].float()
        adapted_logits = adapted_outputs.logits[:, :-1, :].float()
        shift = adapted_logits - base_logits
        mask = encoded["attention_mask"][:, 1:].bool()

        base_hidden = None
        if collect_hidden_states:
            hidden_states = getattr(base_outputs, "hidden_states", None)
            if not hidden_states:
                raise ValueError("Base model did not return hidden states for calibration")
            base_hidden = hidden_states[-1][:, :-1, :].float()

        for batch_index in range(mask.shape[0]):
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).reshape(-1)
            for position in valid_positions.tolist():
                shift_rows.append(shift[batch_index, position].detach().cpu())
                base_rows.append(base_logits[batch_index, position].detach().cpu())
                adapted_rows.append(adapted_logits[batch_index, position].detach().cpu())
                if base_hidden is not None:
                    hidden_rows.append(base_hidden[batch_index, position].detach().cpu())
                prompt_indices.append(prompt_offset + batch_index)
                token_positions.append(position)

    if not shift_rows:
        raise ValueError("Calibration prompts did not yield any next-token positions")
    hidden_state_matrix = torch.stack(hidden_rows) if hidden_rows else None
    return LogitShiftDataset(
        shift_matrix=torch.stack(shift_rows),
        base_logits_matrix=torch.stack(base_rows),
        adapted_logits_matrix=torch.stack(adapted_rows),
        hidden_state_matrix=hidden_state_matrix,
        prompt_indices=prompt_indices,
        token_positions=token_positions,
        vocabulary_size=base_tokenizer.vocab_size,
    )


def compute_logit_shift_matrix(
    base_model: str | "PreTrainedModel",
    adapted_model: str | "PreTrainedModel",
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    adapted_tokenizer: "PreTrainedTokenizerBase" | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    dataset = collect_logit_shift_dataset(
        base_model=base_model,
        adapted_model=adapted_model,
        calibration_prompts=calibration_prompts,
        tokenizer=tokenizer,
        adapted_tokenizer=adapted_tokenizer,
        batch_size=batch_size,
        device=device,
        collect_hidden_states=False,
    )
    return dataset.shift_matrix


@torch.inference_mode()
def collect_hidden_state_matrix(
    model: str | "PreTrainedModel",
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Collect final-layer context features without materializing a second logit dataset."""
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty")
    loaded_model, loaded_tokenizer = _load_model_and_tokenizer(
        model,
        tokenizer=tokenizer,
        device=device,
    )
    model_device = next(loaded_model.parameters()).device
    hidden_rows: list[torch.Tensor] = []
    prompt_indices: list[int] = []
    token_positions: list[int] = []
    for prompt_offset, batch_prompts in _batch_prompts(calibration_prompts, batch_size):
        encoded = loaded_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        outputs = loaded_model(**encoded, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Model did not return hidden states for calibration")
        final_hidden = hidden_states[-1][:, :-1, :].float()
        mask = encoded["attention_mask"][:, 1:].bool()
        for batch_index in range(mask.shape[0]):
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).reshape(-1)
            for position in valid_positions.tolist():
                hidden_rows.append(final_hidden[batch_index, position].detach().cpu())
                prompt_indices.append(prompt_offset + batch_index)
                token_positions.append(position)
    if not hidden_rows:
        raise ValueError("Calibration prompts did not yield any hidden-state positions")
    return torch.stack(hidden_rows), prompt_indices, token_positions


def center_logit_shift_rows(shift_matrix: torch.Tensor) -> torch.Tensor:
    """Fix the softmax logit gauge by removing each context's vocabulary mean."""
    if shift_matrix.ndim != 2:
        raise ValueError("shift_matrix must be a 2D tensor")
    return shift_matrix - shift_matrix.mean(dim=-1, keepdim=True)


def singular_value_spectrum(matrix: torch.Tensor) -> torch.Tensor:
    """Return exact singular values using the smaller covariance matrix when beneficial."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor")
    if min(matrix.shape) == 0:
        return torch.empty(0, dtype=torch.float32)
    values = matrix.float()
    if values.shape[0] <= values.shape[1]:
        eigenvalues = torch.linalg.eigvalsh(values @ values.T).clamp_min(0.0)
        return torch.sqrt(eigenvalues.flip(0))
    return torch.linalg.svdvals(values)


def truncated_right_singular_subspace(
    matrix: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top right-singular vectors and the full exact singular spectrum."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    values = matrix.float()
    maximum_rank = min(values.shape)
    if maximum_rank == 0:
        raise ValueError("matrix must not be empty")
    selected_rank = min(rank, maximum_rank)
    if values.shape[0] <= values.shape[1]:
        eigenvalues, eigenvectors = torch.linalg.eigh(values @ values.T)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        left_vectors = eigenvectors[:, order[:selected_rank]]
        singular_values = torch.sqrt(eigenvalues)
        selected_values = singular_values[:selected_rank]
        tolerance = torch.finfo(values.dtype).eps * max(values.shape) * singular_values[0].clamp_min(1.0)
        nonzero = selected_values > tolerance
        if not torch.any(nonzero):
            return torch.empty(
                values.shape[1],
                0,
                dtype=values.dtype,
                device=values.device,
            ), singular_values
        basis = values.T @ left_vectors[:, nonzero]
        basis = basis / selected_values[nonzero].unsqueeze(0)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        return basis.contiguous(), singular_values
    _, singular_values, vh = torch.linalg.svd(values, full_matrices=False)
    return vh[:selected_rank].T.contiguous(), singular_values


def effective_rank(shift_matrix: torch.Tensor, threshold: float = 0.99) -> int:
    if shift_matrix.ndim != 2:
        raise ValueError("shift_matrix must be a 2D tensor")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    singular_values = singular_value_spectrum(shift_matrix)
    spectral_energy = singular_values.square()
    total_energy = float(spectral_energy.sum().item())
    if total_energy <= 0.0:
        return 0
    cumulative = torch.cumsum(spectral_energy, dim=0) / total_energy
    threshold_tensor = torch.tensor(float(threshold), device=cumulative.device)
    return int(torch.searchsorted(cumulative, threshold_tensor, right=False).item() + 1)


def spectral_analysis(shift_matrix: torch.Tensor) -> SpectralAnalysisResult:
    if shift_matrix.ndim != 2:
        raise ValueError("shift_matrix must be a 2D tensor")
    singular_values = singular_value_spectrum(shift_matrix)
    if singular_values.numel() == 0:
        raise ValueError("shift_matrix must not be empty")
    squared = singular_values.square()
    energy = squared / squared.sum().clamp_min(1e-12)
    cumulative = torch.cumsum(energy, dim=0)
    sigma_sum = float(squared.sum().item())
    sigma_max = float(squared.max().item())
    participation_ratio = float((sigma_sum**2) / squared.square().sum().clamp_min(1e-12).item())
    stable_rank = float(sigma_sum / max(sigma_max, 1e-12))
    return SpectralAnalysisResult(
        singular_values=singular_values.detach().cpu().tolist(),
        cumulative_energy=cumulative.detach().cpu().tolist(),
        participation_ratio=participation_ratio,
        stable_rank=stable_rank,
        effective_rank_95=effective_rank(shift_matrix, threshold=0.95),
        effective_rank_99=effective_rank(shift_matrix, threshold=0.99),
    )


def parameter_delta_from_models(
    base_model: "PreTrainedModel",
    adapted_model: "PreTrainedModel",
) -> dict[str, torch.Tensor]:
    base_params = {
        _normalize_parameter_name(name): parameter.detach()
        for name, parameter in base_model.named_parameters()
    }
    delta: dict[str, torch.Tensor] = {}
    has_live_lora = any(
        getattr(module, "lora_A", None) is not None and getattr(module, "lora_B", None) is not None
        for module in adapted_model.modules()
    )
    if not has_live_lora:
        for name, parameter in adapted_model.named_parameters():
            normalized = _normalize_parameter_name(name)
            if normalized in base_params and base_params[normalized].shape == parameter.shape:
                difference = (parameter.detach() - base_params[normalized].to(parameter.device)).float()
                if torch.count_nonzero(difference).item() > 0:
                    delta[normalized] = difference
    for module_name, module in adapted_model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None:
            continue
        adapter_keys = set(getattr(lora_a, "keys", lambda: [])()) & set(getattr(lora_b, "keys", lambda: [])())
        for adapter_key in adapter_keys:
            matrix_a = lora_a[adapter_key].weight.detach().float()
            matrix_b = lora_b[adapter_key].weight.detach().float()
            scale = float(scaling.get(adapter_key, 1.0)) if isinstance(scaling, dict) else 1.0
            target_name = _normalize_parameter_name(f"{module_name}.weight")
            if target_name not in base_params:
                continue
            effective_delta = (matrix_b @ matrix_a) * scale
            existing = delta.get(target_name)
            delta[target_name] = effective_delta if existing is None else existing + effective_delta.to(existing.device)
    if not delta:
        raise ValueError("No parameter deltas were found between base and adapted models")
    return delta


def first_order_logit_shift(
    base_model: "PreTrainedModel",
    delta_W: Mapping[str, torch.Tensor],
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    batch_size: int = 1,
) -> torch.Tensor:
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty")
    model_device = next(base_model.parameters()).device
    base_model = base_model.eval()
    base_params = {name: parameter.detach() for name, parameter in base_model.named_parameters()}
    active_names = [name for name in base_params if _normalize_parameter_name(name) in delta_W]
    if not active_names:
        raise ValueError("delta_W does not match any base-model parameter names")
    active_primals = tuple(base_params[name] for name in active_names)
    active_tangents = tuple(
        delta_W[_normalize_parameter_name(name)].detach().to(
            device=base_params[name].device,
            dtype=base_params[name].dtype,
        )
        for name in active_names
    )
    static_params = {name: parameter for name, parameter in base_params.items() if name not in active_names}
    buffers = {
        name: buffer.detach()
        for name, buffer in base_model.named_buffers()
    }
    rows: list[torch.Tensor] = []

    for _, batch_prompts in _batch_prompts(calibration_prompts, batch_size):
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        def _forward_from_active(*active_values: torch.Tensor) -> torch.Tensor:
            parameter_dict = dict(static_params)
            parameter_dict.update(zip(active_names, active_values))
            outputs = functional_call(
                base_model,
                (parameter_dict, buffers),
                (),
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded.get("attention_mask"),
                },
            )
            return outputs.logits[:, :-1, :].float()

        _, directional_shift = jvp(_forward_from_active, active_primals, active_tangents)
        mask = encoded["attention_mask"][:, 1:].bool()
        for batch_index in range(mask.shape[0]):
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).reshape(-1)
            for position in valid_positions.tolist():
                rows.append(directional_shift[batch_index, position].detach().cpu())

    if not rows:
        raise ValueError("Calibration prompts did not yield any next-token positions")
    return torch.stack(rows)


def nonlinearity_residual(
    true_shift: torch.Tensor,
    first_order_shift: torch.Tensor,
) -> NonlinearityResidualResult:
    if true_shift.shape != first_order_shift.shape:
        raise ValueError("true_shift and first_order_shift must have matching shapes")
    gauge_true = center_logit_shift_rows(true_shift.float())
    gauge_first_order = center_logit_shift_rows(first_order_shift.float())
    residual = gauge_true - gauge_first_order
    denominator = torch.linalg.matrix_norm(gauge_true, ord="fro").item()
    frobenius_fraction = float(
        torch.linalg.matrix_norm(residual, ord="fro").item() / max(denominator, 1e-12)
    )
    row_norm = torch.linalg.vector_norm(gauge_true, dim=-1).clamp_min(1e-12)
    row_residual = torch.linalg.vector_norm(residual, dim=-1) / row_norm
    cosine = torch.nn.functional.cosine_similarity(
        gauge_true,
        gauge_first_order,
        dim=-1,
        eps=1e-12,
    )
    return NonlinearityResidualResult(
        frobenius_fraction=frobenius_fraction,
        relative_row_mean=float(row_residual.mean().item()),
        cosine_similarity_mean=float(cosine.mean().item()),
    )


def dominant_subspace_basis(
    matrix: torch.Tensor,
    rank: int | None = None,
    threshold: float = 0.99,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    selected_rank = rank if rank is not None else effective_rank(matrix, threshold=threshold)
    if selected_rank == 0:
        return torch.empty(matrix.shape[1], 0, dtype=torch.float32, device=matrix.device)
    basis, _ = truncated_right_singular_subspace(matrix, rank=selected_rank)
    return basis


def subspace_overlap_from_bases(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
) -> SubspaceOverlapResult:
    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("basis_a and basis_b must be 2D")
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("basis_a and basis_b must share the same ambient dimension")
    shared_rank = min(basis_a.shape[1], basis_b.shape[1])
    if shared_rank == 0:
        return SubspaceOverlapResult(
            rank=0,
            principal_angles_degrees=[],
            cosines=[],
            mean_cosine=0.0,
            chordal_distance=0.0,
        )
    qa, _ = torch.linalg.qr(basis_a[:, :shared_rank].float(), mode="reduced")
    qb, _ = torch.linalg.qr(basis_b[:, :shared_rank].float(), mode="reduced")
    cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosines))
    chordal = torch.sqrt(torch.clamp(shared_rank - cosines.square().sum(), min=0.0))
    return SubspaceOverlapResult(
        rank=int(shared_rank),
        principal_angles_degrees=angles.detach().cpu().tolist(),
        cosines=cosines.detach().cpu().tolist(),
        mean_cosine=float(cosines.mean().item()),
        chordal_distance=float(chordal.item()),
    )


def subspace_overlap(
    shift_matrix_A: torch.Tensor,
    shift_matrix_B: torch.Tensor,
    rank: int | None = None,
    threshold: float = 0.99,
) -> SubspaceOverlapResult:
    basis_a = dominant_subspace_basis(shift_matrix_A, rank=rank, threshold=threshold)
    basis_b = dominant_subspace_basis(shift_matrix_B, rank=rank, threshold=threshold)
    return subspace_overlap_from_bases(basis_a, basis_b)
