from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lora_spec.predictive import (
    LinearRegressionModel,
    MLPRegressionModel,
    MultivariateRegressionModel,
    leave_one_out_cv,
    save_scatter_plot,
)
from lora_spec.utils import add_common_args, get_config_value, resolve_config, setup_logging, write_json_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive degradation models.")
    add_common_args(parser)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/predictive")
    return parser.parse_args()


def _load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target"] for row in rows], dtype=np.float64)
    return features, targets


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "train_predictive")
    config_data = resolve_config(args.config, args.override)
    input_json_value = get_config_value(config_data, args, "input_json")
    if not input_json_value:
        raise ValueError("input_json must be provided")
    input_json = str(input_json_value)
    output_dir = Path(str(get_config_value(config_data, args, "output_dir")))
    features, targets = _load_dataset(input_json)

    linear = LinearRegressionModel().fit(features, targets)
    multi = MultivariateRegressionModel().fit(features, targets)
    mlp = MLPRegressionModel(seed=args.seed).fit(features, targets)

    linear_cv = leave_one_out_cv(lambda: LinearRegressionModel(), features, targets)
    multi_cv = leave_one_out_cv(lambda: MultivariateRegressionModel(), features, targets)
    mlp_cv = leave_one_out_cv(lambda: MLPRegressionModel(seed=args.seed, epochs=150), features, targets)

    scatter_path = save_scatter_plot(
        targets=targets,
        predictions=multi.predict(features),
        output_path=output_dir / "multivariate_scatter.png",
        title="Observed vs predicted acceptance degradation",
    )
    output = write_json_result(
        payload={
            "linear": linear.evaluate(features, targets).__dict__,
            "multivariate": multi.evaluate(features, targets).__dict__,
            "mlp": mlp.evaluate(features, targets).__dict__,
            "linear_loocv": linear_cv.__dict__,
            "multivariate_loocv": multi_cv.__dict__,
            "mlp_loocv": mlp_cv.__dict__,
            "scatter_plot": str(scatter_path),
        },
        output_dir=output_dir,
        stem="predictive_models",
        config={"input_json": input_json},
        cwd=Path.cwd(),
    )
    logger.info("Saved predictive modeling results to %s", output)


if __name__ == "__main__":
    main()
