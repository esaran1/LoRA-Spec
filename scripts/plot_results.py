from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from lora_spec.utils import add_common_args, get_config_value, resolve_config, setup_logging, write_json_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 1 acceptance and throughput comparisons.")
    add_common_args(parser)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "plot_results")
    config_data = resolve_config(args.config, args.override)
    input_json = get_config_value(config_data, args, "input_json")
    if not input_json:
        raise ValueError("input_json must be provided")
    output_dir = Path(str(get_config_value(config_data, args, "output_dir")))
    payload = json.loads(Path(str(input_json)).read_text(encoding="utf-8"))
    baseline = payload["baseline"]
    adapted = payload["adapted"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["baseline", "adapted"], [baseline["acceptance_rate_overall"], adapted["acceptance_rate_overall"]])
    axes[0].set_title("Acceptance rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(["baseline", "adapted"], [baseline["throughput_tps"], adapted["throughput_tps"]])
    axes[1].set_title("Throughput (tok/s)")

    figure.tight_layout()
    output_path = output_dir / "phase1_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    result_path = write_json_result(
        payload={
            "input_json": str(Path(str(input_json)).resolve()),
            "plot_path": str(output_path.resolve()),
            "acceptance_delta": adapted["acceptance_rate_overall"] - baseline["acceptance_rate_overall"],
            "throughput_delta_tps": adapted["throughput_tps"] - baseline["throughput_tps"],
        },
        output_dir=output_dir,
        stem="phase1_plot",
        config={"input_json": str(input_json)},
        cwd=Path.cwd(),
    )
    logger.info("Saved plot to %s", output_path)
    logger.info("Saved plot manifest to %s", result_path)


if __name__ == "__main__":
    main()
