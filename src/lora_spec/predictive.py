from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch


ArrayLike = np.ndarray | list[list[float]] | list[float]


@dataclass
class RegressionMetrics:
    r_squared: float
    mse: float
    predictions: list[float]
    targets: list[float]


def r_squared_score(targets: ArrayLike, predictions: ArrayLike) -> float:
    y_true = np.asarray(targets, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must contain the same number of values")
    if y_true.size == 0:
        raise ValueError("targets and predictions must not be empty")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("targets and predictions must be finite")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - (ss_res / ss_tot)


class Regressor(Protocol):
    fit: Callable[[ArrayLike, ArrayLike], "Regressor"]
    predict: Callable[[ArrayLike], np.ndarray]


def _validated_arrays(
    features: ArrayLike,
    targets: ArrayLike,
    min_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("features must be one- or two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must contain the same number of rows")
    if y.shape[0] < min_samples:
        raise ValueError(f"at least {min_samples} observations are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    return x, y


class _EvaluationMixin:
    def evaluate(self, features: ArrayLike, targets: ArrayLike) -> RegressionMetrics:
        predictions = self.predict(features)
        _, y_true = _validated_arrays(features, targets)
        if predictions.reshape(-1).shape[0] != y_true.shape[0]:
            raise ValueError("predictions and targets must contain the same number of rows")
        mse = float(np.mean((predictions.reshape(-1) - y_true) ** 2))
        return RegressionMetrics(
            r_squared=r_squared_score(y_true, predictions),
            mse=mse,
            predictions=predictions.reshape(-1).tolist(),
            targets=y_true.tolist(),
        )


class LinearRegressionModel(_EvaluationMixin):
    def __init__(self, feature_index: int = 0) -> None:
        self.feature_index = feature_index
        self.coefficients: np.ndarray | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> "LinearRegressionModel":
        x, y_vector = _validated_arrays(features, targets)
        y = y_vector.reshape(-1, 1)
        if self.feature_index < 0 or self.feature_index >= x.shape[1]:
            raise ValueError("feature_index is out of range")
        column = x[:, [self.feature_index]]
        design = np.concatenate([np.ones((column.shape[0], 1)), column], axis=1)
        self.coefficients = np.linalg.pinv(design) @ y
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before predict")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        column = x[:, [self.feature_index]]
        design = np.concatenate([np.ones((column.shape[0], 1)), column], axis=1)
        return (design @ self.coefficients).reshape(-1)


class MultivariateRegressionModel(_EvaluationMixin):
    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = ridge
        self.coefficients: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> "MultivariateRegressionModel":
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        x, y_vector = _validated_arrays(features, targets)
        y = y_vector.reshape(-1, 1)
        self.feature_mean = x.mean(axis=0, keepdims=True)
        self.feature_std = x.std(axis=0, keepdims=True)
        self.feature_std[self.feature_std < 1e-12] = 1.0
        normalized = (x - self.feature_mean) / self.feature_std
        design = np.concatenate([np.ones((x.shape[0], 1)), normalized], axis=1)
        ridge_eye = np.eye(design.shape[1]) * self.ridge
        ridge_eye[0, 0] = 0.0
        self.coefficients = np.linalg.solve(design.T @ design + ridge_eye, design.T @ y)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        if self.coefficients is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model must be fit before predict")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        normalized = (x - self.feature_mean) / self.feature_std
        design = np.concatenate([np.ones((x.shape[0], 1)), normalized], axis=1)
        return (design @ self.coefficients).reshape(-1)


class MLPRegressionModel(_EvaluationMixin):
    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 300,
        lr: float = 1e-2,
        seed: int = 7,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.network: torch.nn.Module | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> "MLPRegressionModel":
        if self.hidden_dim < 1 or self.epochs < 1 or self.lr <= 0.0:
            raise ValueError("hidden_dim, epochs, and lr must be positive")
        x64, y64 = _validated_arrays(features, targets)
        x = x64.astype(np.float32)
        y = y64.astype(np.float32).reshape(-1, 1)

        self.feature_mean = x.mean(axis=0, keepdims=True)
        self.feature_std = x.std(axis=0, keepdims=True) + 1e-6
        x_norm = (x - self.feature_mean) / self.feature_std

        torch.manual_seed(self.seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(x_norm.shape[1], self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        features_tensor = torch.from_numpy(x_norm)
        targets_tensor = torch.from_numpy(y)

        network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions = network(features_tensor)
            loss = loss_fn(predictions, targets_tensor)
            loss.backward()
            optimizer.step()

        self.network = network.eval()
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        if self.network is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model must be fit before predict")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x_norm = (x - self.feature_mean) / self.feature_std
        with torch.no_grad():
            predictions = self.network(torch.from_numpy(x_norm)).cpu().numpy()
        return predictions.reshape(-1)


def leave_one_out_cv(
    model_factory: Callable[[], Regressor],
    features: ArrayLike,
    targets: ArrayLike,
) -> RegressionMetrics:
    x, y = _validated_arrays(features, targets, min_samples=2)
    predictions = np.zeros_like(y)
    for index in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[index] = False
        model = model_factory().fit(x[train_mask], y[train_mask])
        predictions[index] = model.predict(x[[index]])[0]
    mse = float(np.mean((predictions - y) ** 2))
    return RegressionMetrics(
        r_squared=r_squared_score(y, predictions),
        mse=mse,
        predictions=predictions.tolist(),
        targets=y.tolist(),
    )


def leave_one_group_out_cv(
    model_factory: Callable[[], Regressor],
    features: ArrayLike,
    targets: ArrayLike,
    groups: list[str] | np.ndarray,
) -> RegressionMetrics:
    x, y = _validated_arrays(features, targets, min_samples=2)
    group_values = np.asarray(groups, dtype=str).reshape(-1)
    if len(y) != len(group_values):
        raise ValueError("groups must contain exactly one value per target")
    unique_groups = np.unique(group_values)
    if unique_groups.size < 2:
        raise ValueError("Grouped cross-validation requires at least two groups")
    predictions = np.zeros_like(y)
    for group in unique_groups:
        test_mask = group_values == group
        train_mask = ~test_mask
        if not np.any(train_mask) or not np.any(test_mask):
            raise ValueError(f"Invalid grouped fold for {group}")
        model = model_factory().fit(x[train_mask], y[train_mask])
        predictions[test_mask] = model.predict(x[test_mask])
    mse = float(np.mean((predictions - y) ** 2))
    return RegressionMetrics(
        r_squared=r_squared_score(y, predictions),
        mse=mse,
        predictions=predictions.tolist(),
        targets=y.tolist(),
    )


def save_scatter_plot(
    targets: ArrayLike,
    predictions: ArrayLike,
    output_path: str | Path,
    title: str,
) -> Path:
    target_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction_values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(target_values, prediction_values, alpha=0.8)
    minimum = float(min(target_values.min(), prediction_values.min()))
    maximum = float(max(target_values.max(), prediction_values.max()))
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black")
    axis.set_xlabel("Observed degradation")
    axis.set_ylabel("Predicted degradation")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
