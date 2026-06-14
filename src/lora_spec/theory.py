from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import matplotlib.pyplot as plt
import torch
from torch.func import functional_call, jvp

from .artifacts import tokenizers_are_equivalent

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
    continuation_contexts: ContinuationContextSet


@dataclass(frozen=True)
class ContinuationContextSet:
    """Token trajectories defining the exact next-token contexts under study."""

    input_ids: tuple[torch.Tensor, ...]
    prompt_lengths: tuple[int, ...]
    continuation_lengths: tuple[int, ...]
    trajectory_model: str
    generation_policy: str

    def __post_init__(self) -> None:
        count = len(self.input_ids)
        if count == 0:
            raise ValueError("Continuation contexts must not be empty")
        if len(self.prompt_lengths) != count or len(self.continuation_lengths) != count:
            raise ValueError("Context sequences and length metadata must have equal lengths")
        if not self.trajectory_model.strip() or not self.generation_policy.strip():
            raise ValueError("Trajectory model and generation policy must be non-empty")
        for index, (sequence, prompt_length, continuation_length) in enumerate(
            zip(self.input_ids, self.prompt_lengths, self.continuation_lengths)
        ):
            if sequence.ndim != 1:
                raise ValueError(f"Context sequence {index} must be one-dimensional")
            if sequence.dtype != torch.long:
                raise ValueError(f"Context sequence {index} must use torch.long token IDs")
            if prompt_length < 1 or continuation_length < 1:
                raise ValueError(f"Context sequence {index} has non-positive segment lengths")
            expected_length = prompt_length + continuation_length
            if sequence.numel() != expected_length:
                raise ValueError(
                    f"Context sequence {index} has {sequence.numel()} tokens; "
                    f"expected {expected_length}",
                )

    @property
    def num_positions(self) -> int:
        return sum(self.continuation_lengths)

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"lora-spec-continuation-context-v2\0")
        for value in (self.trajectory_model, self.generation_policy):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(len(self.input_ids).to_bytes(8, "big"))
        for prompt_length, continuation_length, sequence in zip(
            self.prompt_lengths,
            self.continuation_lengths,
            self.input_ids,
        ):
            digest.update(int(prompt_length).to_bytes(8, "big"))
            digest.update(int(continuation_length).to_bytes(8, "big"))
            values = sequence.to(dtype=torch.int64, device="cpu").contiguous().numpy()
            digest.update(int(values.size).to_bytes(8, "big"))
            digest.update(values.tobytes())
        return digest.hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256(),
            "trajectory_model": self.trajectory_model,
            "generation_policy": self.generation_policy,
            "prompt_lengths": list(self.prompt_lengths),
            "continuation_lengths": list(self.continuation_lengths),
            "input_ids": [sequence.tolist() for sequence in self.input_ids],
            "num_positions": self.num_positions,
        }


@dataclass
class SpectralAnalysisResult:
    singular_values: list[float]
    cumulative_energy: list[float]
    participation_ratio: float
    stable_rank: float
    effective_rank_95: int
    effective_rank_99: int

    def save_spectrum_plot(
        self, output_path: str | Path, title: str = "Logit Shift Spectrum"
    ) -> Path:
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


@dataclass(frozen=True)
class FactoredParameterDelta:
    terms: tuple[tuple[torch.Tensor, torch.Tensor, float], ...]

    def materialize(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.terms:
            raise ValueError("Factored parameter delta must contain at least one term")
        result: torch.Tensor | None = None
        for matrix_b, matrix_a, scale in self.terms:
            term = (
                matrix_b.to(device=device, dtype=dtype) @ matrix_a.to(device=device, dtype=dtype)
            ) * scale
            result = term if result is None else result + term
        if result is None:
            raise RuntimeError("Failed to materialize factored parameter delta")
        return result

    def dense_numel(self) -> int:
        matrix_b, matrix_a, _ = self.terms[0]
        return int(matrix_b.shape[0] * matrix_a.shape[1])


ParameterDelta = torch.Tensor | FactoredParameterDelta


@dataclass
class SubspaceOverlapResult:
    rank: int
    rank_a: int
    rank_b: int
    principal_angles_degrees: list[float]
    cosines: list[float]
    mean_cosine: float
    chordal_distance: float
    overlap_fraction_a: float
    overlap_fraction_b: float


def _load_model_and_tokenizer(
    model_or_name: str | "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    adapter_path: str | None = None,
    device: str | torch.device | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    except ImportError as exc:  # pragma: no cover - only needed for model-backed paths
        raise ImportError(
            "transformers must be installed for model-backed theory utilities"
        ) from exc
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


@torch.inference_mode()
def build_continuation_contexts(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompts: list[str],
    max_new_tokens: int = 16,
    trajectory_model: str = "base_target",
    max_prompt_length: int | None = None,
) -> ContinuationContextSet:
    """Generate deterministic trajectories and retain every proposal-time context.

    For a prompt of length ``p`` and ``g`` generated tokens, the measured logits
    are positions ``p - 1`` through ``p + g - 2``. This includes the first
    continuation prediction and excludes prompt-internal teacher-forced logits.
    """
    if not prompts:
        raise ValueError("prompts must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_prompt_length is not None and max_prompt_length < 1:
        raise ValueError("max_prompt_length must be positive")
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        if eos_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token for continuation generation")
        pad_token_id = eos_token_id

    sequences: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    continuation_lengths: list[int] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        prompt_length = int(attention_mask.sum().item())
        if prompt_length < 1:
            raise ValueError("Every prompt must yield at least one token")
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )[0]
        continuation_length = int(generated.numel() - input_ids.shape[1])
        if continuation_length < 1:
            raise RuntimeError("Trajectory generation produced no continuation tokens")
        sequences.append(generated.detach().cpu())
        prompt_lengths.append(prompt_length)
        continuation_lengths.append(continuation_length)
    return ContinuationContextSet(
        input_ids=tuple(sequences),
        prompt_lengths=tuple(prompt_lengths),
        continuation_lengths=tuple(continuation_lengths),
        trajectory_model=trajectory_model,
        generation_policy=f"greedy_max_new_tokens_{max_new_tokens}",
    )


def iter_continuation_context_batches(
    contexts: ContinuationContextSet,
    batch_size: int,
    pad_token_id: int,
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    for start in range(0, len(contexts.input_ids), batch_size):
        sequences = contexts.input_ids[start : start + batch_size]
        maximum_length = max(int(sequence.numel()) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), maximum_length),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            length = int(sequence.numel())
            input_ids[row, :length] = sequence
            attention_mask[row, :length] = 1
        yield start, input_ids, attention_mask


@torch.inference_mode()
def collect_context_model_outputs(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    contexts: ContinuationContextSet,
    batch_size: int = 2,
    collect_hidden_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, list[int], list[int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires a pad or EOS token")
    device = next(model.parameters()).device
    logit_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    prompt_indices: list[int] = []
    token_positions: list[int] = []
    for start, input_ids, attention_mask in iter_continuation_context_batches(
        contexts,
        batch_size,
        pad_token_id,
    ):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=collect_hidden_states,
        )
        logits = outputs.logits.float()
        hidden_states = getattr(outputs, "hidden_states", None)
        if collect_hidden_states and not hidden_states:
            raise ValueError("Model did not return hidden states")
        final_hidden = hidden_states[-1].float() if hidden_states else None
        for row in range(input_ids.shape[0]):
            prompt_index = start + row
            first_position = contexts.prompt_lengths[prompt_index] - 1
            count = contexts.continuation_lengths[prompt_index]
            last_position = first_position + count
            logit_rows.extend(logits[row, first_position:last_position].detach().cpu())
            if final_hidden is not None:
                hidden_rows.extend(final_hidden[row, first_position:last_position].detach().cpu())
            prompt_indices.extend([prompt_index] * count)
            token_positions.extend(range(first_position, last_position))
    if not logit_rows:
        raise ValueError("Continuation trajectories did not yield prediction contexts")
    return (
        torch.stack(logit_rows),
        torch.stack(hidden_rows) if hidden_rows else None,
        prompt_indices,
        token_positions,
    )


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
    continuation_tokens: int = 16,
    continuation_contexts: ContinuationContextSet | None = None,
) -> LogitShiftDataset:
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty")
    base, base_tokenizer = _load_model_and_tokenizer(base_model, tokenizer=tokenizer, device=device)
    adapted_tokenizer_input = (
        adapted_tokenizer
        if isinstance(adapted_model, str)
        else adapted_tokenizer or tokenizer or base_tokenizer
    )
    adapted, adapted_tokenizer = _load_model_and_tokenizer(
        adapted_model,
        tokenizer=adapted_tokenizer_input,
        device=device or next(base.parameters()).device,
    )
    if not tokenizers_are_equivalent(
        base_tokenizer,
        adapted_tokenizer,
        calibration_prompts,
    ):
        raise ValueError(
            "Base and adapted models must use exactly equivalent tokenizers for logit-shift analysis"
        )
    base_tokenizer.padding_side = "right"
    adapted_tokenizer.padding_side = "right"
    contexts = continuation_contexts or build_continuation_contexts(
        base,
        base_tokenizer,
        calibration_prompts,
        max_new_tokens=continuation_tokens,
    )
    base_logits_matrix, hidden_state_matrix, prompt_indices, token_positions = (
        collect_context_model_outputs(
            base,
            base_tokenizer,
            contexts,
            batch_size=batch_size,
            collect_hidden_states=collect_hidden_states,
        )
    )
    adapted_logits_matrix, _, adapted_prompt_indices, adapted_token_positions = (
        collect_context_model_outputs(
            adapted,
            adapted_tokenizer,
            contexts,
            batch_size=batch_size,
            collect_hidden_states=False,
        )
    )
    if prompt_indices != adapted_prompt_indices or token_positions != adapted_token_positions:
        raise RuntimeError("Base and adapted continuation contexts are not aligned")
    return LogitShiftDataset(
        shift_matrix=adapted_logits_matrix - base_logits_matrix,
        base_logits_matrix=base_logits_matrix,
        adapted_logits_matrix=adapted_logits_matrix,
        hidden_state_matrix=hidden_state_matrix,
        prompt_indices=prompt_indices,
        token_positions=token_positions,
        vocabulary_size=int(base_logits_matrix.shape[-1]),
        continuation_contexts=contexts,
    )


def compute_logit_shift_matrix(
    base_model: str | "PreTrainedModel",
    adapted_model: str | "PreTrainedModel",
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    adapted_tokenizer: "PreTrainedTokenizerBase" | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    continuation_tokens: int = 16,
    continuation_contexts: ContinuationContextSet | None = None,
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
        continuation_tokens=continuation_tokens,
        continuation_contexts=continuation_contexts,
    )
    return dataset.shift_matrix


@torch.inference_mode()
def collect_hidden_state_matrix(
    model: str | "PreTrainedModel",
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase" | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    continuation_tokens: int = 16,
    continuation_contexts: ContinuationContextSet | None = None,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Collect final-layer context features without materializing a second logit dataset."""
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty")
    loaded_model, loaded_tokenizer = _load_model_and_tokenizer(
        model,
        tokenizer=tokenizer,
        device=device,
    )
    contexts = continuation_contexts or build_continuation_contexts(
        loaded_model,
        loaded_tokenizer,
        calibration_prompts,
        max_new_tokens=continuation_tokens,
    )
    _, hidden_states, prompt_indices, token_positions = collect_context_model_outputs(
        loaded_model,
        loaded_tokenizer,
        contexts,
        batch_size=batch_size,
        collect_hidden_states=True,
    )
    if hidden_states is None:
        raise RuntimeError("Hidden-state collection returned no hidden states")
    return hidden_states, prompt_indices, token_positions


def center_logit_shift_rows(shift_matrix: torch.Tensor) -> torch.Tensor:
    """Fix the softmax logit gauge by removing each context's vocabulary mean."""
    if shift_matrix.ndim != 2:
        raise ValueError("shift_matrix must be a 2D tensor")
    return shift_matrix - shift_matrix.mean(dim=-1, keepdim=True)


def _row_gram_float64(matrix: torch.Tensor, column_chunk_size: int = 16_384) -> torch.Tensor:
    """Accumulate ``matrix @ matrix.T`` without squaring error in float32."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor")
    if column_chunk_size < 1:
        raise ValueError("column_chunk_size must be positive")
    gram = torch.zeros(
        matrix.shape[0],
        matrix.shape[0],
        dtype=torch.float64,
        device=matrix.device,
    )
    for start in range(0, matrix.shape[1], column_chunk_size):
        chunk = matrix[:, start : start + column_chunk_size].to(dtype=torch.float64)
        gram.addmm_(chunk, chunk.T)
    return gram


def singular_value_spectrum(matrix: torch.Tensor) -> torch.Tensor:
    """Return singular values with float64 accumulation for reliable spectral tails."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor")
    if min(matrix.shape) == 0:
        return torch.empty(0, dtype=torch.float32)
    values = matrix.float()
    if values.shape[0] <= values.shape[1]:
        eigenvalues = torch.linalg.eigvalsh(_row_gram_float64(values)).clamp_min(0.0)
        return torch.sqrt(eigenvalues.flip(0))
    return torch.linalg.svdvals(values.to(dtype=torch.float64))


def _effective_rank_from_singular_values(
    singular_values: torch.Tensor,
    threshold: float,
) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    spectral_energy = singular_values.square()
    total_energy = float(spectral_energy.sum().item())
    if total_energy <= 0.0:
        return 0
    cumulative = torch.cumsum(spectral_energy, dim=0) / total_energy
    threshold_tensor = torch.tensor(
        float(threshold),
        dtype=cumulative.dtype,
        device=cumulative.device,
    )
    return int(torch.searchsorted(cumulative, threshold_tensor, right=False).item() + 1)


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
        eigenvalues, eigenvectors = torch.linalg.eigh(_row_gram_float64(values))
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        left_vectors = eigenvectors[:, order[:selected_rank]]
        singular_values = torch.sqrt(eigenvalues)
        selected_values = singular_values[:selected_rank]
        # The Gram matrix and eigendecomposition are float64. Using float32
        # epsilon here makes the numerical-rank cutoff scale with vocabulary
        # size and can discard valid correction directions at large vocabularies.
        tolerance = (
            torch.finfo(torch.float64).eps * max(values.shape) * singular_values[0].clamp_min(1.0)
        )
        nonzero = selected_values > tolerance
        if not torch.any(nonzero):
            return torch.empty(
                values.shape[1],
                0,
                dtype=values.dtype,
                device=values.device,
            ), singular_values
        selected_left_vectors = left_vectors[:, nonzero]
        basis = torch.empty(
            values.shape[1],
            int(nonzero.sum().item()),
            dtype=torch.float64,
            device=values.device,
        )
        for start in range(0, values.shape[1], 16_384):
            chunk = values[:, start : start + 16_384].to(dtype=torch.float64)
            basis[start : start + chunk.shape[1]] = chunk.T @ selected_left_vectors
        basis = basis / selected_values[nonzero].unsqueeze(0)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        return basis.to(dtype=values.dtype).contiguous(), singular_values
    _, singular_values, vh = torch.linalg.svd(values.to(dtype=torch.float64), full_matrices=False)
    return vh[:selected_rank].T.to(dtype=values.dtype).contiguous(), singular_values


def effective_rank(shift_matrix: torch.Tensor, threshold: float = 0.99) -> int:
    if shift_matrix.ndim != 2:
        raise ValueError("shift_matrix must be a 2D tensor")
    singular_values = singular_value_spectrum(shift_matrix)
    return _effective_rank_from_singular_values(singular_values, threshold)


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
        effective_rank_95=_effective_rank_from_singular_values(singular_values, threshold=0.95),
        effective_rank_99=_effective_rank_from_singular_values(singular_values, threshold=0.99),
    )


def spectral_sample_size_sensitivity(
    shift_matrix: torch.Tensor,
    cluster_ids: list[int],
    sample_sizes: list[int],
    seed: int = 7,
    threshold: float = 0.99,
    repetitions: int = 5,
) -> list[dict[str, float | int | list[float] | list[int]]]:
    """Measure spectral summaries under repeated prompt-level subsampling."""
    if shift_matrix.ndim != 2:
        raise ValueError("shift_matrix must be 2D")
    if len(cluster_ids) != shift_matrix.shape[0]:
        raise ValueError("cluster_ids must contain one entry per matrix row")
    unique_clusters = sorted(set(cluster_ids))
    if not unique_clusters:
        raise ValueError("cluster_ids must not be empty")
    valid_sizes = sorted(set(sample_sizes))
    if not valid_sizes or valid_sizes[0] < 1 or valid_sizes[-1] > len(unique_clusters):
        raise ValueError("sample_sizes must lie within the available cluster count")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    cluster_tensor = torch.tensor(cluster_ids, dtype=torch.long)
    results: list[dict[str, float | int | list[float] | list[int]]] = []
    for sample_size in valid_sizes:
        effective_ranks: list[int] = []
        stable_ranks: list[float] = []
        participation_ratios: list[float] = []
        row_counts: list[int] = []
        rank_ceilings: list[int] = []
        actual_repetitions = 1 if sample_size == len(unique_clusters) else repetitions
        for repetition in range(actual_repetitions):
            generator = torch.Generator(device="cpu").manual_seed(seed + repetition)
            order = torch.randperm(len(unique_clusters), generator=generator).tolist()
            selected = torch.tensor(
                [unique_clusters[index] for index in order[:sample_size]],
                dtype=torch.long,
            )
            mask = torch.isin(cluster_tensor, selected)
            subset = shift_matrix[mask]
            analysis = spectral_analysis(subset)
            row_counts.append(int(subset.shape[0]))
            rank_ceilings.append(int(min(subset.shape)))
            effective_ranks.append(effective_rank(subset, threshold=threshold))
            stable_ranks.append(analysis.stable_rank)
            participation_ratios.append(analysis.participation_ratio)
        median_rank = int(torch.tensor(effective_ranks).median().item())
        median_ceiling = int(torch.tensor(rank_ceilings).median().item())
        results.append(
            {
                "num_clusters": sample_size,
                "num_rows": int(torch.tensor(row_counts).median().item()),
                "num_rows_range": [min(row_counts), max(row_counts)],
                "rank_ceiling": median_ceiling,
                "rank_ceiling_range": [min(rank_ceilings), max(rank_ceilings)],
                "effective_rank": median_rank,
                "effective_rank_estimates": effective_ranks,
                "effective_rank_range": [min(effective_ranks), max(effective_ranks)],
                "effective_rank_fraction_of_ceiling": float(median_rank / max(median_ceiling, 1)),
                "stable_rank": float(torch.tensor(stable_ranks).median().item()),
                "stable_rank_estimates": stable_ranks,
                "participation_ratio": float(torch.tensor(participation_ratios).median().item()),
                "participation_ratio_estimates": participation_ratios,
                "repetitions": actual_repetitions,
            }
        )
    return results


def parameter_delta_from_models(
    base_model: "PreTrainedModel",
    adapted_model: "PreTrainedModel",
) -> dict[str, ParameterDelta]:
    peft_configs = getattr(adapted_model, "peft_config", None)
    if isinstance(peft_configs, Mapping):
        for adapter_name, config in peft_configs.items():
            peft_type = str(getattr(config, "peft_type", "LORA")).upper().split(".")[-1]
            unsupported: list[str] = []
            if peft_type != "LORA":
                unsupported.append(f"peft_type={peft_type}")
            if bool(getattr(config, "use_dora", False)):
                unsupported.append("use_dora=True")
            bias = str(getattr(config, "bias", "none")).lower()
            if bias != "none":
                unsupported.append(f"bias={bias}")
            modules_to_save = getattr(config, "modules_to_save", None)
            if modules_to_save:
                unsupported.append("modules_to_save")
            if unsupported:
                raise ValueError(
                    f"First-order analysis supports plain LoRA only; adapter {adapter_name!r} "
                    f"uses {', '.join(unsupported)}"
                )
    base_params = {
        _normalize_parameter_name(name): parameter.detach()
        for name, parameter in base_model.named_parameters()
    }
    delta: dict[str, ParameterDelta] = {}
    has_live_lora = any(
        getattr(module, "lora_A", None) is not None and getattr(module, "lora_B", None) is not None
        for module in adapted_model.modules()
    )
    if not has_live_lora:
        for name, parameter in adapted_model.named_parameters():
            normalized = _normalize_parameter_name(name)
            if normalized in base_params and base_params[normalized].shape == parameter.shape:
                difference = (
                    parameter.detach() - base_params[normalized].to(parameter.device)
                ).float()
                if torch.count_nonzero(difference).item() > 0:
                    delta[normalized] = difference
    for module_name, module in adapted_model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None:
            continue
        adapter_keys = set(getattr(lora_a, "keys", lambda: [])()) & set(
            getattr(lora_b, "keys", lambda: [])()
        )
        for adapter_key in adapter_keys:
            matrix_a = lora_a[adapter_key].weight.detach().float()
            matrix_b = lora_b[adapter_key].weight.detach().float()
            scale = float(scaling.get(adapter_key, 1.0)) if isinstance(scaling, dict) else 1.0
            target_name = _normalize_parameter_name(f"{module_name}.weight")
            if target_name not in base_params:
                continue
            existing = delta.get(target_name)
            term = (matrix_b.cpu(), matrix_a.cpu(), scale)
            if existing is None:
                delta[target_name] = FactoredParameterDelta((term,))
            elif isinstance(existing, FactoredParameterDelta):
                delta[target_name] = FactoredParameterDelta(existing.terms + (term,))
            else:
                delta[target_name] = existing + FactoredParameterDelta((term,)).materialize(
                    device=existing.device,
                    dtype=existing.dtype,
                )
    if not delta:
        raise ValueError("No parameter deltas were found between base and adapted models")
    return delta


def first_order_logit_shift(
    base_model: "PreTrainedModel",
    delta_W: Mapping[str, ParameterDelta],
    calibration_prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    batch_size: int = 1,
    max_tangent_bytes: int = 512 * 1024 * 1024,
    continuation_tokens: int = 16,
    continuation_contexts: ContinuationContextSet | None = None,
) -> torch.Tensor:
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty")
    model_device = next(base_model.parameters()).device
    base_model = base_model.eval()
    base_params = {name: parameter.detach() for name, parameter in base_model.named_parameters()}
    active_names = [name for name in base_params if _normalize_parameter_name(name) in delta_W]
    if not active_names:
        raise ValueError("delta_W does not match any base-model parameter names")
    if max_tangent_bytes < 1:
        raise ValueError("max_tangent_bytes must be positive")

    def delta_bytes(name: str) -> int:
        value = delta_W[_normalize_parameter_name(name)]
        numel = value.dense_numel() if isinstance(value, FactoredParameterDelta) else value.numel()
        return int(numel * base_params[name].element_size())

    active_groups: list[list[str]] = []
    current_group: list[str] = []
    current_bytes = 0
    for name in active_names:
        required_bytes = delta_bytes(name)
        if required_bytes > max_tangent_bytes:
            raise ValueError(
                f"Dense tangent for {name} requires {required_bytes} bytes, exceeding "
                f"max_tangent_bytes={max_tangent_bytes}. Increase the explicit budget or "
                "exclude this parameter from first-order analysis."
            )
        if current_group and current_bytes + required_bytes > max_tangent_bytes:
            active_groups.append(current_group)
            current_group = []
            current_bytes = 0
        current_group.append(name)
        current_bytes += required_bytes
    if current_group:
        active_groups.append(current_group)

    buffers = {name: buffer.detach() for name, buffer in base_model.named_buffers()}
    rows: list[torch.Tensor] = []

    contexts = continuation_contexts or build_continuation_contexts(
        base_model,
        tokenizer,
        calibration_prompts,
        max_new_tokens=continuation_tokens,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires a pad or EOS token")
    for context_start, input_ids, attention_mask in iter_continuation_context_batches(
        contexts,
        batch_size,
        pad_token_id,
    ):
        encoded = {
            "input_ids": input_ids.to(model_device),
            "attention_mask": attention_mask.to(model_device),
        }

        directional_shift: torch.Tensor | None = None
        for group_names in active_groups:
            group_name_set = set(group_names)
            static_params = {
                name: parameter
                for name, parameter in base_params.items()
                if name not in group_name_set
            }
            active_primals = tuple(base_params[name] for name in group_names)
            tangent_values: list[torch.Tensor] = []
            for name in group_names:
                value = delta_W[_normalize_parameter_name(name)]
                if isinstance(value, FactoredParameterDelta):
                    tangent = value.materialize(
                        device=base_params[name].device,
                        dtype=base_params[name].dtype,
                    )
                else:
                    tangent = value.detach().to(
                        device=base_params[name].device,
                        dtype=base_params[name].dtype,
                    )
                tangent_values.append(tangent)

            def _forward_from_active(*active_values: torch.Tensor) -> torch.Tensor:
                parameter_dict = dict(static_params)
                parameter_dict.update(zip(group_names, active_values))
                outputs = functional_call(
                    base_model,
                    (parameter_dict, buffers),
                    (),
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded.get("attention_mask"),
                    },
                )
                return outputs.logits.float()

            _, group_shift = jvp(
                _forward_from_active,
                active_primals,
                tuple(tangent_values),
            )
            directional_shift = (
                group_shift if directional_shift is None else directional_shift + group_shift
            )
            del tangent_values, group_shift
        if directional_shift is None:
            raise RuntimeError("First-order JVP produced no directional shift")
        for batch_index in range(encoded["input_ids"].shape[0]):
            prompt_index = context_start + batch_index
            first_position = contexts.prompt_lengths[prompt_index] - 1
            count = contexts.continuation_lengths[prompt_index]
            rows.extend(
                directional_shift[
                    batch_index,
                    first_position : first_position + count,
                ]
                .detach()
                .cpu()
            )

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
    rank_a = int(basis_a.shape[1])
    rank_b = int(basis_b.shape[1])
    if shared_rank == 0:
        return SubspaceOverlapResult(
            rank=0,
            rank_a=rank_a,
            rank_b=rank_b,
            principal_angles_degrees=[],
            cosines=[],
            mean_cosine=0.0,
            chordal_distance=float(((rank_a + rank_b) / 2.0) ** 0.5),
            overlap_fraction_a=0.0,
            overlap_fraction_b=0.0,
        )
    qa, _ = torch.linalg.qr(basis_a.float(), mode="reduced")
    qb, _ = torch.linalg.qr(basis_b.float(), mode="reduced")
    cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosines))
    overlap_energy = cosines.square().sum()
    chordal = torch.sqrt(torch.clamp((rank_a + rank_b) / 2.0 - overlap_energy, min=0.0))
    return SubspaceOverlapResult(
        rank=int(shared_rank),
        rank_a=rank_a,
        rank_b=rank_b,
        principal_angles_degrees=angles.detach().cpu().tolist(),
        cosines=cosines.detach().cpu().tolist(),
        mean_cosine=float(cosines.mean().item()),
        chordal_distance=float(chordal.item()),
        overlap_fraction_a=float((overlap_energy / max(rank_a, 1)).item()),
        overlap_fraction_b=float((overlap_energy / max(rank_b, 1)).item()),
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
