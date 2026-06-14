from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    num_clusters: int
    repetitions: int


@dataclass(frozen=True)
class PiecewiseLinearFit:
    breakpoint: float
    intercept: float
    slope_before: float
    slope_after: float
    residual_sum_squares: float
    linear_residual_sum_squares: float
    bic: float
    linear_bic: float
    bic_improvement: float
    num_observations: int


def cluster_means(
    values: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[int | str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if observations.shape != clusters.shape:
        raise ValueError("values and cluster_ids must have matching one-dimensional shapes")
    if observations.size == 0:
        raise ValueError("values must not be empty")
    if not np.isfinite(observations).all():
        raise ValueError("values must be finite")
    unique_clusters = np.unique(clusters)
    means = np.asarray(
        [observations[clusters == cluster].mean() for cluster in unique_clusters],
        dtype=np.float64,
    )
    return unique_clusters, means


def cluster_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[int | str] | np.ndarray,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> BootstrapInterval:
    """Estimate an equal-cluster-weighted mean and percentile interval.

    Token positions from one prompt are correlated. This function first reduces
    each prompt to one mean, then resamples prompts rather than token rows.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    _, means = cluster_means(values, cluster_ids)
    estimate = float(means.mean())
    if means.size == 1:
        return BootstrapInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            confidence=confidence,
            num_clusters=1,
            repetitions=repetitions,
        )
    generator = np.random.default_rng(seed)
    sampled_indices = generator.integers(0, means.size, size=(repetitions, means.size))
    bootstrap_means = means[sampled_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        num_clusters=int(means.size),
        repetitions=repetitions,
    )


def fit_continuous_piecewise_linear(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    min_points_per_side: int = 3,
) -> PiecewiseLinearFit:
    """Select a continuous one-breakpoint regression by minimum BIC.

    This is an exploratory change-point diagnostic. BIC improvement is evidence
    for curvature relative to one straight line, not proof of a phase transition.
    """
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have matching one-dimensional shapes")
    if x_values.size < 2 * min_points_per_side:
        raise ValueError("Not enough observations for the requested segmented fit")
    if min_points_per_side < 2:
        raise ValueError("min_points_per_side must be at least 2")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("x and y must be finite")
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if np.unique(x_values).size != x_values.size:
        raise ValueError("x values must be unique")

    linear_design = np.column_stack([np.ones_like(x_values), x_values])
    linear_coefficients = np.linalg.lstsq(linear_design, y_values, rcond=None)[0]
    linear_residual = y_values - linear_design @ linear_coefficients
    linear_rss = float(linear_residual @ linear_residual)
    n = int(x_values.size)
    epsilon = np.finfo(np.float64).tiny
    linear_bic = n * np.log(max(linear_rss / n, epsilon)) + 2 * np.log(n)

    best: tuple[float, float, np.ndarray, float] | None = None
    candidate_indices = range(min_points_per_side - 1, n - min_points_per_side + 1)
    for index in candidate_indices:
        breakpoint = float(x_values[index])
        hinge = np.maximum(x_values - breakpoint, 0.0)
        design = np.column_stack([np.ones_like(x_values), x_values, hinge])
        coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        residual = y_values - design @ coefficients
        rss = float(residual @ residual)
        # The breakpoint is selected from the observations and is therefore
        # an additional fitted parameter beyond the three hinge coefficients.
        bic = n * np.log(max(rss / n, epsilon)) + 4 * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, breakpoint, coefficients, rss)
    if best is None:
        raise RuntimeError("No valid breakpoint candidate was evaluated")
    bic, breakpoint, coefficients, rss = best
    return PiecewiseLinearFit(
        breakpoint=breakpoint,
        intercept=float(coefficients[0]),
        slope_before=float(coefficients[1]),
        slope_after=float(coefficients[1] + coefficients[2]),
        residual_sum_squares=rss,
        linear_residual_sum_squares=linear_rss,
        bic=float(bic),
        linear_bic=float(linear_bic),
        bic_improvement=float(linear_bic - bic),
        num_observations=n,
    )
