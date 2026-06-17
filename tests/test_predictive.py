from __future__ import annotations

import numpy as np
import pytest

from lora_spec.predictive import (
    LinearRegressionModel,
    MLPRegressionModel,
    MultivariateRegressionModel,
    leave_one_group_out_cv,
    leave_one_out_cv,
    r_squared_score,
)


def test_r_squared_score_is_one_for_perfect_predictions() -> None:
    assert r_squared_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_r_squared_rejects_misaligned_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="same number of values"):
        r_squared_score([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="finite"):
        r_squared_score([1.0, float("inf")], [1.0, 2.0])


def test_linear_and_multivariate_regression_fit_simple_relation() -> None:
    features = np.asarray([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]])
    targets = np.asarray([2.0, 4.0, 6.0, 8.0])
    linear = LinearRegressionModel().fit(features, targets)
    multi = MultivariateRegressionModel().fit(features, targets)
    assert linear.evaluate(features, targets).r_squared > 0.99
    assert multi.evaluate(features, targets).r_squared > 0.99


def test_multivariate_regression_is_invariant_to_feature_units() -> None:
    features = np.asarray([[1.0, 1e6], [2.0, 2e6], [3.0, 1e6], [4.0, 2e6]])
    targets = np.asarray([1.0, 2.0, 3.0, 4.0])
    scaled = features.copy()
    scaled[:, 1] /= 1e6

    original_prediction = (
        MultivariateRegressionModel(ridge=1e-3).fit(features, targets).predict(features)
    )
    scaled_prediction = MultivariateRegressionModel(ridge=1e-3).fit(scaled, targets).predict(scaled)

    assert np.allclose(original_prediction, scaled_prediction, atol=1e-8)


def test_mlp_and_loocv_run_on_small_cpu_dataset() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    mlp = MLPRegressionModel(hidden_dim=8, epochs=50, lr=0.05, seed=0).fit(features, targets)
    assert mlp.evaluate(features, targets).r_squared > 0.95
    cv = leave_one_out_cv(lambda: MultivariateRegressionModel(), features, targets)
    assert len(cv.predictions) == len(targets)


def test_leave_one_group_out_keeps_related_rows_in_same_fold() -> None:
    features = np.asarray([[0.0], [0.2], [2.0], [2.2]], dtype=np.float64)
    targets = np.asarray([0.0, 0.2, 2.0, 2.2], dtype=np.float64)
    result = leave_one_group_out_cv(
        lambda: MultivariateRegressionModel(),
        features,
        targets,
        groups=["adapter_a", "adapter_a", "adapter_b", "adapter_b"],
    )

    assert len(result.predictions) == 4


def test_cross_validation_rejects_too_few_or_nonfinite_rows() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        leave_one_out_cv(lambda: MultivariateRegressionModel(), [[1.0]], [1.0])

    with pytest.raises(ValueError, match="finite"):
        leave_one_out_cv(lambda: MultivariateRegressionModel(), [[1.0], [float("nan")]], [1.0, 2.0])


def test_regression_rejects_misaligned_targets_and_bad_feature_index() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        MultivariateRegressionModel().fit([[1.0], [2.0]], [1.0])

    with pytest.raises(ValueError, match="feature_index"):
        LinearRegressionModel(feature_index=3).fit([[1.0], [2.0]], [1.0, 2.0])
