from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch


ErrorMetric = Literal["total_variation", "logit_span"]


@dataclass
class AcceptanceRecoveryPrediction:
    approximation_error: list[float]
    acceptance_lower_bound: list[float]
    guaranteed_recovery_delta: list[float]
    error_metric: ErrorMetric


@dataclass
class RecoveryComparisonResult:
    approximation_error: list[float]
    acceptance_lower_bound: list[float]
    empirical_acceptance: list[float]
    guaranteed_recovery_delta: list[float]
    empirical_recovery_delta: list[float]
    bound_coverage: float
    mean_bound_slack: float
    maximum_bound_violation: float
    error_metric: ErrorMetric

    def save_plot(self, output_path: str | Path, title: str = "Acceptance-Recovery Bound") -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        order = np.argsort(np.asarray(self.approximation_error))
        errors = np.asarray(self.approximation_error)[order]
        empirical_delta = np.asarray(self.empirical_recovery_delta)[order]
        guaranteed_delta = np.asarray(self.guaranteed_recovery_delta)[order]
        axes[0].scatter(errors, empirical_delta, color="#1d4ed8", label="empirical")
        axes[0].plot(errors, guaranteed_delta, color="#d55c4b", label="guaranteed lower bound")
        axes[0].set_xlabel(
            "Total variation" if self.error_metric == "total_variation" else "Residual logit span"
        )
        axes[0].set_ylabel("Acceptance recovery delta")
        axes[0].set_title("Recovery Guarantee")
        axes[0].legend(loc="best")

        lower = np.asarray(self.acceptance_lower_bound)
        empirical = np.asarray(self.empirical_acceptance)
        axes[1].scatter(lower, empirical, color="#0f766e", alpha=0.85)
        minimum = float(min(lower.min(), empirical.min()))
        maximum = float(max(lower.max(), empirical.max()))
        axes[1].plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black")
        axes[1].set_xlabel("Acceptance lower bound")
        axes[1].set_ylabel("Empirical acceptance")
        axes[1].set_title(f"Coverage={self.bound_coverage:.1%}, slack={self.mean_bound_slack:.3f}")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return path


def _as_array(value: float | list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("approximation_error must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError("approximation_error must be non-negative")
    return array


def total_variation_distance(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("target_probabilities and draft_probabilities must have matching shapes")
    return 0.5 * torch.sum(torch.abs(target_probabilities - draft_probabilities), dim=dim)


def rejection_sampling_acceptance(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Expected modified-rejection-sampling acceptance, equal to 1 - TV(p, q)."""
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("target_probabilities and draft_probabilities must have matching shapes")
    return torch.sum(torch.minimum(target_probabilities, draft_probabilities), dim=dim)


def acceptance_lower_bound_from_logit_residual(
    target_logits: torch.Tensor,
    approximate_logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Lower-bound acceptance using the shift-invariant span of the logit residual.

    If the residual-logit oscillation is s, the normalized likelihood-ratio
    oscillation is also s. The tight ratio-oscillation bound is
    TV <= tanh(s / 4), hence acceptance >= 1 - tanh(s / 4).
    """
    if target_logits.shape != approximate_logits.shape:
        raise ValueError("target_logits and approximate_logits must have matching shapes")
    residual = target_logits.float() - approximate_logits.float()
    span = residual.amax(dim=dim) - residual.amin(dim=dim)
    return 1.0 - torch.tanh(0.25 * span)


def predicted_acceptance_recovery(
    approximation_error: float | list[float] | np.ndarray,
    base_acceptance: float,
    error_metric: ErrorMetric = "total_variation",
) -> AcceptanceRecoveryPrediction:
    """Return a per-token acceptance lower bound under standard rejection sampling.

    For total variation the expression is the exact expected acceptance identity.
    For residual logit span s it uses the tight TV <= tanh(s / 4) bound.
    """
    errors = _as_array(approximation_error)
    if not 0.0 <= base_acceptance <= 1.0:
        raise ValueError("base_acceptance must lie in [0, 1]")
    if error_metric == "total_variation":
        if np.any(errors > 1.0 + 1e-12):
            raise ValueError("total variation must lie in [0, 1]")
        lower_bound = 1.0 - np.clip(errors, 0.0, 1.0)
    elif error_metric == "logit_span":
        lower_bound = 1.0 - np.tanh(0.25 * errors)
    else:
        raise ValueError(f"Unsupported error_metric: {error_metric}")
    guaranteed_delta = lower_bound - float(base_acceptance)
    return AcceptanceRecoveryPrediction(
        approximation_error=errors.tolist(),
        acceptance_lower_bound=lower_bound.tolist(),
        guaranteed_recovery_delta=guaranteed_delta.tolist(),
        error_metric=error_metric,
    )


def empirical_vs_predicted_recovery(
    approximation_error: float | list[float] | np.ndarray,
    base_acceptance: float,
    empirical_acceptance: list[float] | np.ndarray,
    error_metric: ErrorMetric = "total_variation",
    tolerance: float = 1e-6,
) -> RecoveryComparisonResult:
    prediction = predicted_acceptance_recovery(
        approximation_error=approximation_error,
        base_acceptance=base_acceptance,
        error_metric=error_metric,
    )
    empirical = _as_array(empirical_acceptance)
    lower_bound = _as_array(prediction.acceptance_lower_bound)
    if empirical.shape != lower_bound.shape:
        raise ValueError("empirical_acceptance must match approximation_error length")
    if np.any(empirical < -tolerance) or np.any(empirical > 1.0 + tolerance):
        raise ValueError("empirical_acceptance must lie in [0, 1]")
    slack = empirical - lower_bound
    violations = np.maximum(-slack, 0.0)
    return RecoveryComparisonResult(
        approximation_error=prediction.approximation_error,
        acceptance_lower_bound=lower_bound.tolist(),
        empirical_acceptance=empirical.tolist(),
        guaranteed_recovery_delta=prediction.guaranteed_recovery_delta,
        empirical_recovery_delta=(empirical - float(base_acceptance)).tolist(),
        bound_coverage=float(np.mean(slack >= -tolerance)),
        mean_bound_slack=float(np.mean(slack)),
        maximum_bound_violation=float(np.max(violations)),
        error_metric=error_metric,
    )
