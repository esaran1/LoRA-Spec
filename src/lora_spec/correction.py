from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .theory import (
    ContinuationContextSet,
    LogitShiftDataset,
    build_continuation_contexts,
    center_logit_shift_rows,
    collect_context_model_outputs,
    collect_hidden_state_matrix,
    collect_logit_shift_dataset,
    spectral_analysis,
    truncated_right_singular_subspace,
)


class Correction(Protocol):
    requires_hidden_state: bool

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
        feature_model: PreTrainedModel | None = None,
        feature_tokenizer: PreTrainedTokenizerBase | None = None,
        continuation_tokens: int = 16,
        continuation_contexts: ContinuationContextSet | None = None,
    ) -> "Correction": ...

    def apply(
        self,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def measure_overhead_ms(
        self,
        repeats: int = 200,
        warmup: int = 20,
        device: str | torch.device | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> float: ...


@dataclass
class ApproximationErrorReport:
    spectral_tail_relative_frobenius: float
    centered_shift_reconstruction_relative_frobenius: float
    coefficient_regression_relative_frobenius: float
    predicted_centered_operator_relative_frobenius: float
    centered_operator_relative_frobenius: float
    operator_calibration_relative_frobenius: float
    end_to_end_calibration_relative_frobenius: float
    base_feature_coefficient_relative_frobenius: float
    base_feature_operator_relative_frobenius: float
    retained_energy_fraction: float
    selected_rank: int


class _BaseCorrection:
    requires_hidden_state = False

    def __init__(self) -> None:
        self.vocab_size: int | None = None
        self._tensor_cache: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}

    def _check_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        if self.vocab_size is None:
            raise RuntimeError("Correction must be calibrated before apply")
        if draft_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Expected logits last dimension {self.vocab_size}, got {draft_logits.shape[-1]}",
            )
        return draft_logits

    def measure_overhead_ms(
        self,
        repeats: int = 200,
        warmup: int = 20,
        device: str | torch.device | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> float:
        if self.vocab_size is None:
            raise RuntimeError("Correction must be calibrated before overhead measurement")
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        logits = torch.zeros(1, self.vocab_size, dtype=torch.float32, device=target_device)
        if hidden_state is not None:
            hidden_state = hidden_state.to(target_device)
        for _ in range(max(warmup, 0)):
            self.apply(logits, hidden_state=hidden_state)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        start_time = time.perf_counter()
        for _ in range(repeats):
            self.apply(logits, hidden_state=hidden_state)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        elapsed = time.perf_counter() - start_time
        return float((elapsed * 1000.0) / max(repeats, 1))

    def _cached(
        self, name: str, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (name, str(device), dtype)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = tensor.to(device=device, dtype=dtype)
            self._tensor_cache[key] = cached
        return cached


class MeanShiftCorrection(_BaseCorrection):
    def __init__(self) -> None:
        super().__init__()
        self.mean_shift: torch.Tensor | None = None

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
        feature_model: PreTrainedModel | None = None,
        feature_tokenizer: PreTrainedTokenizerBase | None = None,
        continuation_tokens: int = 16,
        continuation_contexts: ContinuationContextSet | None = None,
    ) -> "MeanShiftCorrection":
        _ = feature_model, feature_tokenizer
        contexts = continuation_contexts
        if contexts is None and isinstance(base_model, torch.nn.Module) and tokenizer is not None:
            contexts = build_continuation_contexts(
                base_model,
                tokenizer,
                prompts,
                max_new_tokens=continuation_tokens,
            )
        dataset = collect_logit_shift_dataset(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=prompts,
            tokenizer=tokenizer,
            collect_hidden_states=False,
            continuation_tokens=continuation_tokens,
            continuation_contexts=contexts,
        )
        return self.calibrate_from_dataset(dataset)

    def calibrate_from_dataset(self, dataset: LogitShiftDataset) -> "MeanShiftCorrection":
        self.vocab_size = dataset.vocabulary_size
        self.mean_shift = center_logit_shift_rows(dataset.shift_matrix.float()).mean(dim=0)
        self._tensor_cache.clear()
        return self

    def apply(
        self,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = hidden_state
        draft_logits = self._check_logits(draft_logits)
        if self.mean_shift is None:
            raise RuntimeError("Correction must be calibrated before apply")
        return draft_logits + self._cached(
            "mean_shift",
            self.mean_shift,
            draft_logits.device,
            draft_logits.dtype,
        )


class LowRankCorrection(_BaseCorrection):
    def __init__(self, rank: int = 8, ridge: float = 1e-5) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.rank = rank
        self.ridge = ridge
        self.mean_shift: torch.Tensor | None = None
        self.mean_base_logits: torch.Tensor | None = None
        self.mean_feature_logits: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.input_to_coefficients: torch.Tensor | None = None
        self.base_input_to_coefficients: torch.Tensor | None = None
        self.singular_values: torch.Tensor | None = None
        self.selected_rank: int | None = None
        self._dataset: LogitShiftDataset | None = None
        self._feature_logits_matrix: torch.Tensor | None = None

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
        feature_model: PreTrainedModel | None = None,
        feature_tokenizer: PreTrainedTokenizerBase | None = None,
        continuation_tokens: int = 16,
        continuation_contexts: ContinuationContextSet | None = None,
    ) -> "LowRankCorrection":
        contexts = continuation_contexts
        if contexts is None and isinstance(base_model, torch.nn.Module) and tokenizer is not None:
            contexts = build_continuation_contexts(
                base_model,
                tokenizer,
                prompts,
                max_new_tokens=continuation_tokens,
            )
        dataset = collect_logit_shift_dataset(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=prompts,
            tokenizer=tokenizer,
            collect_hidden_states=False,
            continuation_tokens=continuation_tokens,
            continuation_contexts=contexts,
        )
        contexts = dataset.continuation_contexts
        feature_logits = dataset.base_logits_matrix
        if feature_model is not None:
            if feature_tokenizer is None and tokenizer is None:
                raise ValueError(
                    "feature_tokenizer is required when draft-feature calibration uses "
                    "an instantiated feature model without a shared tokenizer"
                )
            feature_logits, _, feature_prompt_indices, feature_token_positions = (
                collect_context_model_outputs(
                    feature_model,
                    feature_tokenizer or tokenizer,
                    contexts,
                    collect_hidden_states=False,
                )
            )
            if (
                dataset.prompt_indices != feature_prompt_indices
                or dataset.token_positions != feature_token_positions
            ):
                raise ValueError("Correction features and shift labels are not context-aligned")
        self.calibrate_from_dataset(dataset, feature_logits)
        return self

    def calibrate_from_dataset(
        self,
        dataset: LogitShiftDataset,
        feature_logits_matrix: torch.Tensor | None = None,
    ) -> "LowRankCorrection":
        if feature_logits_matrix is not None and feature_logits_matrix.shape != (
            dataset.num_positions,
            dataset.vocabulary_size,
        ):
            raise ValueError("feature logits must match the shift dataset shape")
        self._fit_from_dataset(dataset, feature_logits_matrix)
        return self

    def _ridge_operator(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = features.dtype
        features = features.to(dtype=torch.float64)
        targets = targets.to(dtype=torch.float64)
        if features.shape[0] <= features.shape[1]:
            gram = features @ features.T
            ridge_scale = float(torch.trace(gram).item()) / max(gram.shape[0], 1)
            ridge_eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            dual = torch.linalg.solve(
                gram + ridge_eye * (self.ridge * max(ridge_scale, 1e-12)),
                targets,
            )
            return (features.T @ dual).to(dtype=original_dtype)
        gram = features.T @ features
        ridge_scale = float(torch.trace(gram).item()) / max(gram.shape[0], 1)
        ridge_eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return torch.linalg.solve(
            gram + ridge_eye * (self.ridge * max(ridge_scale, 1e-12)),
            features.T @ targets,
        ).to(dtype=original_dtype)

    def _fit_from_dataset(
        self,
        dataset: LogitShiftDataset,
        feature_logits_matrix: torch.Tensor | None = None,
    ) -> None:
        base_logits = center_logit_shift_rows(dataset.base_logits_matrix.float())
        raw_feature_logits = (
            feature_logits_matrix
            if feature_logits_matrix is not None
            else dataset.base_logits_matrix
        )
        feature_logits = center_logit_shift_rows(raw_feature_logits.float())
        shift = center_logit_shift_rows(dataset.shift_matrix.float())
        self.vocab_size = dataset.vocabulary_size
        self._dataset = dataset
        self._feature_logits_matrix = raw_feature_logits
        self.mean_base_logits = base_logits.mean(dim=0)
        self.mean_feature_logits = feature_logits.mean(dim=0)
        self.mean_shift = shift.mean(dim=0)
        centered_shift = shift - self.mean_shift
        basis, singular_values = truncated_right_singular_subspace(centered_shift, rank=self.rank)
        selected_rank = int(basis.shape[1])
        base_features = base_logits - self.mean_base_logits
        application_features = feature_logits - self.mean_feature_logits
        target_features = centered_shift @ basis
        input_to_coefficients = self._ridge_operator(application_features, target_features)
        self.basis = basis
        self.input_to_coefficients = input_to_coefficients
        self.base_input_to_coefficients = self._ridge_operator(base_features, target_features)
        self.singular_values = singular_values
        self.selected_rank = selected_rank
        self._tensor_cache.clear()

    def apply(
        self,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = hidden_state
        draft_logits = self._check_logits(draft_logits)
        if any(
            value is None
            for value in (
                self.mean_shift,
                self.mean_feature_logits,
                self.basis,
                self.input_to_coefficients,
            )
        ):
            raise RuntimeError("Correction must be calibrated before apply")
        mean_feature_logits = self._cached(
            "mean_feature_logits",
            self.mean_feature_logits,
            draft_logits.device,
            draft_logits.dtype,
        )
        basis = self._cached("basis", self.basis, draft_logits.device, draft_logits.dtype)
        input_to_coefficients = self._cached(
            "input_to_coefficients",
            self.input_to_coefficients,
            draft_logits.device,
            draft_logits.dtype,
        )
        mean_shift = self._cached(
            "mean_shift", self.mean_shift, draft_logits.device, draft_logits.dtype
        )
        gauge_fixed_logits = draft_logits - draft_logits.mean(dim=-1, keepdim=True)
        centered_logits = gauge_fixed_logits - mean_feature_logits
        correction_coefficients = centered_logits @ input_to_coefficients
        low_rank_shift = correction_coefficients @ basis.T
        return draft_logits + mean_shift + low_rank_shift

    def approximation_error(self, k: int | None = None) -> ApproximationErrorReport:
        if self._dataset is None or self.singular_values is None:
            raise RuntimeError("Correction must be calibrated before approximation_error")
        selected_rank = self.selected_rank
        if selected_rank is None:
            raise RuntimeError("Correction rank was not recorded during calibration")
        if k is not None and k != selected_rank:
            raise ValueError(
                "approximation_error(k) requires k to match the calibrated correction rank"
            )
        total_energy = float(self.singular_values.square().sum().item())
        retained_energy = float(self.singular_values[:selected_rank].square().sum().item())
        tail_energy = max(total_energy - retained_energy, 0.0)
        theoretical = (tail_energy / max(total_energy, 1e-12)) ** 0.5

        gauge_shift = center_logit_shift_rows(self._dataset.shift_matrix.float())
        centered_shift = gauge_shift - gauge_shift.mean(dim=0)
        if self.basis is None:
            raise RuntimeError("Correction basis is unavailable")
        basis = self.basis
        reconstruction = centered_shift @ basis @ basis.T
        empirical = torch.linalg.matrix_norm(
            centered_shift - reconstruction,
            ord="fro",
        ).item() / max(
            torch.linalg.matrix_norm(centered_shift, ord="fro").item(),
            1e-12,
        )
        centered_denominator = max(
            torch.linalg.matrix_norm(centered_shift, ord="fro").item(), 1e-12
        )
        if (
            self.mean_base_logits is None
            or self.mean_feature_logits is None
            or self.input_to_coefficients is None
            or self.base_input_to_coefficients is None
        ):
            raise RuntimeError("Correction operator is incomplete")
        gauge_base = center_logit_shift_rows(self._dataset.base_logits_matrix.float())
        centered_base = gauge_base - self.mean_base_logits
        gauge_features = center_logit_shift_rows(self._feature_logits_matrix.float())
        centered_features = gauge_features - self.mean_feature_logits
        predicted_coefficients = centered_features @ self.input_to_coefficients
        base_predicted_coefficients = centered_base @ self.base_input_to_coefficients
        target_coefficients = centered_shift @ basis
        coefficient_error = (
            torch.linalg.matrix_norm(
                target_coefficients - predicted_coefficients,
                ord="fro",
            ).item()
            / centered_denominator
        )
        predicted_centered_error = (theoretical**2 + coefficient_error**2) ** 0.5
        predicted_centered_shift = predicted_coefficients @ basis.T
        centered_operator_error = (
            torch.linalg.matrix_norm(
                centered_shift - predicted_centered_shift,
                ord="fro",
            ).item()
            / centered_denominator
        )
        base_coefficient_error = (
            torch.linalg.matrix_norm(
                target_coefficients - base_predicted_coefficients,
                ord="fro",
            ).item()
            / centered_denominator
        )
        base_predicted_shift = base_predicted_coefficients @ basis.T
        base_operator_error = (
            torch.linalg.matrix_norm(
                centered_shift - base_predicted_shift,
                ord="fro",
            ).item()
            / centered_denominator
        )
        raw_features = self._feature_logits_matrix.float()
        predicted = self.apply(raw_features)
        predicted_shift = center_logit_shift_rows(predicted - raw_features)
        operator_error = torch.linalg.matrix_norm(
            gauge_shift - predicted_shift,
            ord="fro",
        ).item() / max(
            torch.linalg.matrix_norm(gauge_shift, ord="fro").item(),
            1e-12,
        )
        gauge_residual = center_logit_shift_rows(
            predicted - self._dataset.adapted_logits_matrix.float(),
        )
        uncorrected_residual = center_logit_shift_rows(
            raw_features - self._dataset.adapted_logits_matrix.float(),
        )
        end_to_end_error = torch.linalg.matrix_norm(
            gauge_residual,
            ord="fro",
        ).item() / max(
            torch.linalg.matrix_norm(uncorrected_residual, ord="fro").item(),
            1e-12,
        )
        return ApproximationErrorReport(
            spectral_tail_relative_frobenius=float(theoretical),
            centered_shift_reconstruction_relative_frobenius=float(empirical),
            coefficient_regression_relative_frobenius=float(coefficient_error),
            predicted_centered_operator_relative_frobenius=float(predicted_centered_error),
            centered_operator_relative_frobenius=float(centered_operator_error),
            operator_calibration_relative_frobenius=float(operator_error),
            end_to_end_calibration_relative_frobenius=float(end_to_end_error),
            base_feature_coefficient_relative_frobenius=float(base_coefficient_error),
            base_feature_operator_relative_frobenius=float(base_operator_error),
            retained_energy_fraction=float(retained_energy / max(total_energy, 1e-12)),
            selected_rank=selected_rank,
        )


class ContextDependentCorrection(_BaseCorrection):
    requires_hidden_state = True

    def __init__(
        self,
        rank: int = 8,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 1e-3,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.mean_shift: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None
        self.network: torch.nn.Module | None = None
        self.hidden_size: int | None = None

    def calibrate(
        self,
        base_model: str | PreTrainedModel,
        adapted_model: str | PreTrainedModel,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase | None = None,
        feature_model: str | PreTrainedModel | None = None,
        feature_tokenizer: PreTrainedTokenizerBase | None = None,
        continuation_tokens: int = 16,
        continuation_contexts: ContinuationContextSet | None = None,
    ) -> "ContextDependentCorrection":
        feature_source = feature_model if feature_model is not None else base_model
        source_device = (
            next(feature_source.parameters()).device
            if isinstance(feature_source, torch.nn.Module)
            else torch.device("cpu")
        )
        contexts = continuation_contexts
        if contexts is None and isinstance(base_model, torch.nn.Module) and tokenizer is not None:
            contexts = build_continuation_contexts(
                base_model,
                tokenizer,
                prompts,
                max_new_tokens=continuation_tokens,
            )
        shift_dataset = collect_logit_shift_dataset(
            base_model=base_model,
            adapted_model=adapted_model,
            calibration_prompts=prompts,
            tokenizer=tokenizer,
            collect_hidden_states=False,
            continuation_tokens=continuation_tokens,
            continuation_contexts=contexts,
        )
        contexts = shift_dataset.continuation_contexts
        feature_matrix, feature_prompt_indices, feature_token_positions = (
            collect_hidden_state_matrix(
                model=feature_source,
                calibration_prompts=prompts,
                tokenizer=feature_tokenizer or tokenizer,
                continuation_tokens=continuation_tokens,
                continuation_contexts=contexts,
            )
        )
        if (
            shift_dataset.prompt_indices != feature_prompt_indices
            or shift_dataset.token_positions != feature_token_positions
        ):
            raise ValueError("Shift labels and feature hidden states are not context-aligned")
        return self.calibrate_from_dataset(
            shift_dataset,
            feature_matrix,
            training_device=source_device,
        )

    def calibrate_from_dataset(
        self,
        shift_dataset: LogitShiftDataset,
        feature_matrix: torch.Tensor,
        training_device: str | torch.device | None = None,
    ) -> "ContextDependentCorrection":
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] != shift_dataset.num_positions:
            raise ValueError("hidden-state features must align with the shift dataset rows")
        source_device = (
            torch.device(training_device) if training_device is not None else torch.device("cpu")
        )
        self.vocab_size = shift_dataset.vocabulary_size
        shift = center_logit_shift_rows(shift_dataset.shift_matrix.float())
        self.mean_shift = shift.mean(dim=0)
        centered_shift = shift - self.mean_shift
        self.basis, _ = truncated_right_singular_subspace(centered_shift, rank=self.rank)
        selected_rank = int(self.basis.shape[1])
        if selected_rank == 0:
            raise ValueError("Context-dependent correction requires non-constant shift variation")
        targets = centered_shift @ self.basis
        features = feature_matrix.float()
        self.hidden_size = features.shape[1]
        self.feature_mean = features.mean(dim=0, keepdim=True)
        self.feature_std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        normalized = ((features - self.feature_mean) / self.feature_std).to(source_device)
        targets = targets.to(source_device)

        torch.manual_seed(self.seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, selected_rank),
        ).to(source_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = network(normalized)
            loss = loss_fn(prediction, targets)
            loss.backward()
            optimizer.step()
        self.network = network.eval().to(source_device)
        return self

    def apply(
        self,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        draft_logits = self._check_logits(draft_logits)
        if any(
            value is None
            for value in (
                self.mean_shift,
                self.basis,
                self.feature_mean,
                self.feature_std,
                self.network,
            )
        ):
            raise RuntimeError("Correction must be calibrated before apply")
        if hidden_state is None:
            raise ValueError(
                "ContextDependentCorrection requires a hidden_state tensor at apply time"
            )
        feature_mean = self._cached(
            "feature_mean", self.feature_mean, hidden_state.device, hidden_state.dtype
        )
        feature_std = self._cached(
            "feature_std", self.feature_std, hidden_state.device, hidden_state.dtype
        )
        basis = self._cached("basis", self.basis, draft_logits.device, draft_logits.dtype)
        mean_shift = self._cached(
            "mean_shift", self.mean_shift, draft_logits.device, draft_logits.dtype
        )
        normalized = (hidden_state.to(dtype=torch.float32) - feature_mean) / feature_std
        network_device = next(self.network.parameters()).device
        with torch.no_grad():
            coefficients = self.network(normalized.to(device=network_device, dtype=torch.float32))
        correction = coefficients.to(draft_logits.device, draft_logits.dtype) @ basis.T
        return draft_logits + mean_shift + correction

    def measure_overhead_ms(
        self,
        repeats: int = 200,
        warmup: int = 20,
        device: str | torch.device | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> float:
        if self.hidden_size is None:
            raise RuntimeError("Correction must be calibrated before overhead measurement")
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        if hidden_state is None:
            hidden_state = torch.zeros(
                1, self.hidden_size, dtype=torch.float32, device=target_device
            )
        if self.network is None:
            raise RuntimeError("Correction must be calibrated before overhead measurement")
        self.network = self.network.to(target_device)
        return super().measure_overhead_ms(
            repeats=repeats,
            warmup=warmup,
            device=target_device,
            hidden_state=hidden_state,
        )


def summarize_shift_spectrum(dataset: LogitShiftDataset) -> dict[str, float | list[float]]:
    analysis = spectral_analysis(center_logit_shift_rows(dataset.shift_matrix.float()))
    return {
        "singular_values": analysis.singular_values,
        "cumulative_energy": analysis.cumulative_energy,
        "participation_ratio": analysis.participation_ratio,
        "stable_rank": analysis.stable_rank,
        "effective_rank_95": analysis.effective_rank_95,
        "effective_rank_99": analysis.effective_rank_99,
    }


DistributionOffsetCorrection = MeanShiftCorrection
