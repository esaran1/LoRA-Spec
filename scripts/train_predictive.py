from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lora_spec.predictive import (
    LinearRegressionModel,
    MLPRegressionModel,
    MultivariateRegressionModel,
    leave_one_group_out_cv,
    leave_one_out_cv,
    save_scatter_plot,
)
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    set_seed,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive degradation models.")
    add_common_args(parser)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/predictive")
    parser.add_argument("--group-key", type=str, default="adapter_source")
    parser.add_argument("--secondary-group-key", type=str, default="model_family")
    return parser.parse_args()


def _load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("Predictive modeling requires at least three rows")
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target"] for row in rows], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("features must form a non-empty two-dimensional matrix")
    if targets.shape != (features.shape[0],):
        raise ValueError("targets must contain exactly one scalar per feature row")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Predictive inputs must contain only finite values")
    return features, targets, rows


def _groups_from_rows(rows: list[dict[str, object]], key: str) -> list[str]:
    groups: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"Row {index} is missing required grouped-CV key {key!r}")
        groups.append(str(value))
    if len(set(groups)) < 2:
        raise ValueError(f"Grouped-CV key {key!r} must contain at least two groups")
    return groups


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "train_predictive")
    config_data = resolve_config(args.config, args.override)
    seed = int(get_config_value(config_data, args, "seed"))
    set_seed(seed)
    input_json_value = get_config_value(config_data, args, "input_json")
    if not input_json_value:
        raise ValueError("input_json must be provided")
    input_json = str(input_json_value)
    output_dir = Path(str(get_config_value(config_data, args, "output_dir")))
    group_key = str(get_config_value(config_data, args, "group_key"))
    secondary_group_key = str(get_config_value(config_data, args, "secondary_group_key"))
    features, targets, rows = _load_dataset(input_json)
    primary_groups = _groups_from_rows(rows, group_key)
    secondary_groups = _groups_from_rows(rows, secondary_group_key)

    linear = LinearRegressionModel().fit(features, targets)
    multi = MultivariateRegressionModel().fit(features, targets)
    input_sha256 = hashlib.sha256(Path(input_json).read_bytes()).hexdigest()
    mlp = MLPRegressionModel(seed=seed).fit(features, targets)

    linear_cv = leave_one_out_cv(lambda: LinearRegressionModel(), features, targets)
    multi_cv = leave_one_out_cv(lambda: MultivariateRegressionModel(), features, targets)
    mlp_cv = leave_one_out_cv(
        lambda: MLPRegressionModel(seed=seed, epochs=150),
        features,
        targets,
    )
    linear_grouped_cv = leave_one_group_out_cv(
        lambda: LinearRegressionModel(), features, targets, primary_groups
    )
    multi_grouped_cv = leave_one_group_out_cv(
        lambda: MultivariateRegressionModel(), features, targets, primary_groups
    )
    mlp_grouped_cv = leave_one_group_out_cv(
        lambda: MLPRegressionModel(seed=seed, epochs=150),
        features,
        targets,
        primary_groups,
    )
    multi_model_family_cv = leave_one_group_out_cv(
        lambda: MultivariateRegressionModel(),
        features,
        targets,
        secondary_groups,
    )

    scatter_path = save_scatter_plot(
        targets=targets,
        predictions=multi_grouped_cv.predictions,
        output_path=output_dir / "multivariate_scatter.png",
        title="Source-held-out acceptance degradation",
    )
    output = write_json_result(
        payload={
            "linear": linear.evaluate(features, targets).__dict__,
            "multivariate": multi.evaluate(features, targets).__dict__,
            "mlp": mlp.evaluate(features, targets).__dict__,
            "linear_loocv": linear_cv.__dict__,
            "multivariate_loocv": multi_cv.__dict__,
            "mlp_loocv": mlp_cv.__dict__,
            "linear_grouped_cv": linear_grouped_cv.__dict__,
            "multivariate_grouped_cv": multi_grouped_cv.__dict__,
            "mlp_grouped_cv": mlp_grouped_cv.__dict__,
            "multivariate_model_family_cv": multi_model_family_cv.__dict__,
            "grouping": {
                "primary_key": group_key,
                "primary_groups": sorted(set(primary_groups)),
                "secondary_key": secondary_group_key,
                "secondary_groups": sorted(set(secondary_groups)),
            },
            "scatter_plot": str(scatter_path),
        },
        output_dir=output_dir,
        stem="predictive_models",
        config={
            "input_json": input_json,
            "input_sha256": input_sha256,
            "num_rows": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "group_key": group_key,
            "secondary_group_key": secondary_group_key,
            "seed": seed,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved predictive modeling results to %s", output)


if __name__ == "__main__":
    main()
