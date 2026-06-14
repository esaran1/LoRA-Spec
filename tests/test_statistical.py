from __future__ import annotations

import numpy as np
import pytest

from lora_spec.statistical import cluster_bootstrap_mean, fit_continuous_piecewise_linear


def test_cluster_bootstrap_weights_prompts_not_token_counts() -> None:
    interval = cluster_bootstrap_mean(
        values=[0.0, 0.0, 0.0, 1.0],
        cluster_ids=["long", "long", "long", "short"],
        repetitions=500,
        seed=3,
    )

    assert interval.estimate == pytest.approx(0.5)
    assert interval.num_clusters == 2
    assert interval.lower <= interval.estimate <= interval.upper


def test_piecewise_fit_recovers_known_breakpoint() -> None:
    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    y = [0.0, 0.05, 0.1, 0.15, 0.2, 0.7, 1.2, 1.7]
    fit = fit_continuous_piecewise_linear(x, y)

    assert fit.breakpoint == pytest.approx(2.0)
    assert fit.slope_after > fit.slope_before
    assert fit.bic_improvement > 0.0


def test_piecewise_bic_penalizes_searched_breakpoint() -> None:
    x = np.arange(8, dtype=np.float64)
    y = np.array([0.0, 0.9, 2.1, 3.0, 4.2, 5.1, 5.9, 7.2])
    fit = fit_continuous_piecewise_linear(x, y)
    expected = x.size * np.log(fit.residual_sum_squares / x.size) + 4 * np.log(x.size)
    assert fit.bic == pytest.approx(expected)
