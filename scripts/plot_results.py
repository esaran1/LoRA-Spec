from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lora_spec.utils import (
    add_common_args,
    ensure_dir,
    get_config_value,
    resolve_config,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LoRA-Spec figures from result JSON artifacts."
    )
    add_common_args(parser)
    parser.add_argument("--input-json", action="append", default=[])
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--glob", type=str, default="**/*.json")
    parser.add_argument("--output-dir", type=str, default="results/plots")
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_inputs(input_dir: Path, pattern: str, explicit_inputs: list[str]) -> list[Path]:
    discovered = {Path(value).resolve() for value in explicit_inputs}
    if input_dir.exists():
        discovered.update(path.resolve() for path in input_dir.glob(pattern) if path.is_file())
    return sorted(discovered)


def _payload_timestamp(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    return str(metadata.get("timestamp", ""))


def _latest_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("payloads must not be empty")
    return max(payloads, key=_payload_timestamp)


def _classify_payload(payload: dict[str, Any]) -> str:
    experiment_type = payload.get("experiment_type")
    if experiment_type in {
        "measure_logit_shift_rank",
        "validate_correction_theory",
        "phase_transition_sweep",
        "subspace_sharing",
    }:
        return str(experiment_type)
    if "runs" in payload and isinstance(payload["runs"], list):
        return "characterize"
    if {"linear", "multivariate", "mlp"} <= payload.keys():
        return "predictive"
    if {"mean_shift", "low_rank", "context_dependent", "baseline"} <= payload.keys():
        return "correction"
    if {"baseline", "adapted"} <= payload.keys():
        baseline = payload["baseline"]
        if isinstance(baseline, dict) and "completed_requests" in baseline:
            return "serving"
        if isinstance(baseline, dict) and "acceptance_rate_overall" in baseline:
            return "phase1"
    return "unknown"


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_phase1(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    baseline = latest["baseline"]
    adapted = latest["adapted"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["baseline", "adapted"],
        [baseline["acceptance_rate_overall"], adapted["acceptance_rate_overall"]],
        color=["#5b7c99", "#d55c4b"],
    )
    axes[0].set_title("Phase 1 Acceptance")
    axes[0].set_ylim(0, 1)
    axes[1].bar(
        ["baseline", "adapted"],
        [baseline["throughput_tps"], adapted["throughput_tps"]],
        color=["#5b7c99", "#d55c4b"],
    )
    axes[1].set_title("Phase 1 Throughput")
    axes[1].set_ylabel("tokens/sec")
    return [_save_figure(figure, output_dir / "phase1_comparison.png")]


def _extract_characterize_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for run in payload.get("runs", []):
            result = run.get("result", {})
            baseline = result.get("baseline", {})
            adapted = result.get("adapted", {})
            comparison = result.get("comparison", {})
            experiment = run.get("experiment", {})
            experiment_config = experiment.get("experiment", experiment)
            adapter = experiment_config.get("adapter", {})
            rows.append(
                {
                    "model_pair_name": run.get("model_pair_name"),
                    "adapter_name": run.get("adapter_name"),
                    "rank": adapter.get("rank"),
                    "domain": adapter.get("domain"),
                    "epochs": adapter.get("epochs"),
                    "acceptance_delta": comparison.get("acceptance_delta"),
                    "throughput_delta_tps": comparison.get("throughput_delta_tps"),
                    "baseline_per_position": baseline.get("acceptance_rate_per_position", []),
                    "adapted_per_position": adapted.get("acceptance_rate_per_position", []),
                }
            )
    return [row for row in rows if row["acceptance_delta"] is not None]


def _plot_characterize(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    rows = _extract_characterize_rows(payloads)
    if not rows:
        return []
    paths: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        epoch_label = "unknown" if row["epochs"] is None else str(row["epochs"])
        grouped[(str(row["model_pair_name"]), str(row["domain"]), epoch_label)].append(row)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(grouped), 1)))
    for color, (key, group_rows) in zip(palette, sorted(grouped.items())):
        model_pair, domain, epochs = key
        ordered = sorted(group_rows, key=lambda item: int(item["rank"]))
        label = f"{model_pair} | {domain} | {epochs} ep"
        axes[0].plot(
            [int(row["rank"]) for row in ordered],
            [float(row["acceptance_delta"]) for row in ordered],
            marker="o",
            linewidth=1.5,
            label=label,
            color=color,
        )
        axes[1].plot(
            [int(row["rank"]) for row in ordered],
            [float(row["throughput_delta_tps"]) for row in ordered],
            marker="o",
            linewidth=1.5,
            label=label,
            color=color,
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Acceptance Delta vs Rank")
    axes[0].set_xlabel("LoRA rank")
    axes[0].set_ylabel("adapted - baseline")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Throughput Delta vs Rank")
    axes[1].set_xlabel("LoRA rank")
    axes[1].set_ylabel("tokens/sec delta")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    paths.append(_save_figure(figure, output_dir / "characterize_rank_trends.png"))

    max_position = max(len(row["baseline_per_position"]) for row in rows)
    if max_position > 0:
        baseline_matrix = []
        adapted_matrix = []
        for row in rows:
            base_values = list(row["baseline_per_position"])
            adapted_values = list(row["adapted_per_position"])
            baseline_matrix.append(base_values + [np.nan] * (max_position - len(base_values)))
            adapted_matrix.append(adapted_values + [np.nan] * (max_position - len(adapted_values)))
        baseline_mean = np.nanmean(np.asarray(baseline_matrix, dtype=np.float64), axis=0)
        adapted_mean = np.nanmean(np.asarray(adapted_matrix, dtype=np.float64), axis=0)
        figure, axis = plt.subplots(figsize=(7, 4.5))
        positions = np.arange(1, max_position + 1)
        axis.plot(positions, baseline_mean, marker="o", label="baseline", color="#5b7c99")
        axis.plot(positions, adapted_mean, marker="o", label="adapted", color="#d55c4b")
        axis.set_title("Per-Position Acceptance")
        axis.set_xlabel("draft token position")
        axis.set_ylabel("acceptance rate")
        axis.set_ylim(0, 1)
        axis.legend(loc="best")
        paths.append(_save_figure(figure, output_dir / "characterize_per_position_acceptance.png"))
    return paths


def _plot_predictive(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    models = [
        ("linear_grouped_cv", "Linear source-held-out", "#4f46e5"),
        ("multivariate_grouped_cv", "Multivariate source-held-out", "#0f766e"),
        ("mlp_grouped_cv", "MLP source-held-out", "#b45309"),
    ]
    figure, axes = plt.subplots(1, len(models), figsize=(15, 4.5))
    paths: list[Path] = []
    for axis, (key, title, color) in zip(axes, models):
        metrics = latest.get(key)
        if not isinstance(metrics, dict):
            axis.set_visible(False)
            continue
        targets = np.asarray(metrics.get("targets", []), dtype=np.float64)
        predictions = np.asarray(metrics.get("predictions", []), dtype=np.float64)
        if targets.size == 0 or predictions.size == 0:
            axis.set_visible(False)
            continue
        axis.scatter(targets, predictions, alpha=0.85, color=color)
        minimum = float(min(targets.min(), predictions.min()))
        maximum = float(max(targets.max(), predictions.max()))
        axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black")
        axis.set_title(f"{title}\n$R^2$={metrics.get('r_squared', float('nan')):.3f}")
        axis.set_xlabel("observed")
        axis.set_ylabel("predicted")
    paths.append(_save_figure(figure, output_dir / "predictive_grouped_cv_scatter.png"))
    return paths


def _plot_correction(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    labels = ["baseline", "mean_shift", "low_rank", "context_dependent"]
    names = [label.replace("_", "\n") for label in labels if label in latest]
    kl_values = [float(latest[label]["kl_divergence"]) for label in labels if label in latest]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    palette = ["#64748b", "#0f766e", "#1d4ed8", "#b45309"][: len(names)]
    axes[0].bar(names, kl_values, color=palette)
    axes[0].set_title("Correction KL Divergence")
    axes[0].set_ylabel("KL")
    approximation = latest.get("low_rank_approximation", {})
    axes[1].bar(
        ["spectral tail", "coefficient fit", "operator fit"],
        [
            float(approximation.get("spectral_tail_relative_frobenius", 0.0)),
            float(approximation.get("coefficient_regression_relative_frobenius", 0.0)),
            float(approximation.get("centered_operator_relative_frobenius", 0.0)),
        ],
        color=["#1d4ed8", "#d55c4b", "#0f766e"],
    )
    axes[1].set_title("Low-Rank Approximation Error")
    axes[1].set_ylabel("relative Frobenius error")
    paths = [_save_figure(figure, output_dir / "correction_divergence_bars.png")]

    proxy = latest.get("speculative_proxy")
    if isinstance(proxy, dict) and proxy:
        proxy_labels = []
        acceptance_values = []
        depth1_values = []
        for key in labels:
            if key not in proxy:
                continue
            proxy_labels.append(key.replace("_", "\n"))
            acceptance_values.append(float(proxy[key]["acceptance_rate_overall"]))
            depth_curve = proxy[key].get("acceptance_by_depth", [])
            depth1_values.append(float(depth_curve[0]) if depth_curve else float("nan"))
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].bar(proxy_labels, acceptance_values, color=palette[: len(proxy_labels)])
        axes[0].set_title("Overall Acceptance")
        axes[0].set_ylim(0, 1)
        axes[1].bar(proxy_labels, depth1_values, color=palette[: len(proxy_labels)])
        axes[1].set_title("Depth-1 Acceptance")
        axes[1].set_ylim(0, 1)
        paths.append(_save_figure(figure, output_dir / "correction_speculative_proxy.png"))
    return paths


def _plot_serving(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    rows = []
    for payload in payloads:
        baseline = payload.get("baseline", {})
        adapted = payload.get("adapted", {})
        if not baseline or not adapted:
            continue
        rows.append(
            {
                "pattern": baseline.get("pattern", "unknown"),
                "baseline_tps": float(baseline.get("throughput_tps", 0.0)),
                "adapted_tps": float(adapted.get("throughput_tps", 0.0)),
                "baseline_p95": float(baseline.get("p95_latency_ms", 0.0)),
                "adapted_p95": float(adapted.get("p95_latency_ms", 0.0)),
            }
        )
    if not rows:
        return []
    rows.sort(key=lambda row: row["pattern"])
    labels = [row["pattern"] for row in rows]
    x = np.arange(len(labels))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        x - width / 2,
        [row["baseline_tps"] for row in rows],
        width=width,
        label="baseline",
        color="#5b7c99",
    )
    axes[0].bar(
        x + width / 2,
        [row["adapted_tps"] for row in rows],
        width=width,
        label="adapted",
        color="#d55c4b",
    )
    axes[0].set_title("Serving Throughput by Traffic Pattern")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("tokens/sec")
    axes[0].legend(loc="best")
    axes[1].bar(
        x - width / 2,
        [row["baseline_p95"] for row in rows],
        width=width,
        label="baseline",
        color="#5b7c99",
    )
    axes[1].bar(
        x + width / 2,
        [row["adapted_p95"] for row in rows],
        width=width,
        label="adapted",
        color="#d55c4b",
    )
    axes[1].set_title("Serving p95 Latency by Traffic Pattern")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("ms")
    axes[1].legend(loc="best")
    return [_save_figure(figure, output_dir / "serving_pattern_comparison.png")]


def _plot_measure_logit_shift_rank(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    rows = latest.get("rows", [])
    if not rows:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_pair_name"])].append(row)
    palette = plt.cm.Set2(np.linspace(0, 1, max(len(grouped), 1)))
    for color, (model_pair, group_rows) in zip(palette, sorted(grouped.items())):
        ordered = sorted(
            group_rows, key=lambda item: (float(item["magnitude_scale"]), int(item["adapter_rank"]))
        )
        axes[0].scatter(
            [int(row["adapter_rank"]) for row in ordered],
            [int(row["effective_rank"]) for row in ordered],
            color=color,
            label=model_pair,
        )
        axes[1].scatter(
            [float(row["adapter_properties"]["frobenius_norm_sum"]) for row in ordered],
            [int(row["effective_rank"]) for row in ordered],
            color=color,
            label=model_pair,
        )
    axes[0].set_title("Effective Rank vs Adapter Rank")
    axes[0].set_xlabel("Adapter rank")
    axes[0].set_ylabel("Effective rank")
    axes[1].set_title("Effective Rank vs Adapter Magnitude")
    axes[1].set_xlabel("Frobenius norm of BA")
    axes[1].set_ylabel("Effective rank")
    axes[1].legend(loc="best")
    return [_save_figure(figure, output_dir / "theory_effective_rank.png")]


def _plot_validate_correction_theory(
    payloads: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    rows = latest.get("rows", [])
    if not rows:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ordered = sorted(rows, key=lambda item: int(item["rank"]))
    axes[0].plot(
        [int(row["rank"]) for row in ordered],
        [float(row["heldout_normalized_logit_error"]) for row in ordered],
        marker="o",
        color="#1d4ed8",
    )
    axes[0].set_title("Approximation Error vs Correction Rank")
    axes[0].set_xlabel("Correction rank k")
    axes[0].set_ylabel("held-out normalized logit error")
    axes[1].plot(
        [int(row["rank"]) for row in ordered],
        [
            float(
                row.get("prompt_cluster_bootstrap", {})
                .get("rejection_sampling_acceptance", {})
                .get("estimate", row["expected_rejection_sampling_acceptance"])
            )
            for row in ordered
        ],
        marker="o",
        color="#0f766e",
        label="exact expected acceptance",
    )
    acceptance_intervals = [
        row.get("prompt_cluster_bootstrap", {}).get("rejection_sampling_acceptance")
        for row in ordered
    ]
    if all(isinstance(interval, dict) for interval in acceptance_intervals):
        estimates = np.asarray([float(interval["estimate"]) for interval in acceptance_intervals])
        axes[1].errorbar(
            [int(row["rank"]) for row in ordered],
            estimates,
            yerr=np.asarray(
                [
                    estimates
                    - np.asarray([float(interval["lower"]) for interval in acceptance_intervals]),
                    np.asarray([float(interval["upper"]) for interval in acceptance_intervals])
                    - estimates,
                ]
            ),
            fmt="none",
            capsize=3,
            color="#0f766e",
        )
    axes[1].plot(
        [int(row["rank"]) for row in ordered],
        [float(row["logit_acceptance_lower_bound"]) for row in ordered],
        marker="x",
        color="#d55c4b",
        label="logit-span lower bound",
    )
    axes[1].axhline(
        float(latest.get("baseline_expected_rejection_sampling_acceptance", 0.0)),
        linestyle="--",
        color="black",
        linewidth=1,
        label="uncorrected baseline",
    )
    axes[1].set_title("Acceptance and Certified Lower Bound")
    axes[1].set_xlabel("Correction rank k")
    axes[1].set_ylabel("expected per-token acceptance")
    axes[1].legend(loc="best")
    return [_save_figure(figure, output_dir / "theory_correction_validation.png")]


def _plot_phase_transition_sweep(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    rows = sorted(latest.get("rows", []), key=lambda item: float(item["magnitude_scale"]))
    if not rows:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    scales = [float(row["magnitude_scale"]) for row in rows]
    heldout_values = [
        float(
            row.get(
                "heldout_prompt_weighted_normalized_logit_error",
                row["heldout_normalized_logit_error"],
            )
        )
        for row in rows
    ]
    axes[0].plot(
        scales,
        heldout_values,
        marker="o",
        label="held-out logit error",
        color="#1d4ed8",
    )
    intervals = [row.get("heldout_prompt_cluster_bootstrap") for row in rows]
    if all(isinstance(interval, dict) for interval in intervals):
        estimates = np.asarray(heldout_values)
        axes[0].errorbar(
            scales,
            estimates,
            yerr=np.asarray(
                [
                    estimates - np.asarray([float(interval["lower"]) for interval in intervals]),
                    np.asarray([float(interval["upper"]) for interval in intervals]) - estimates,
                ]
            ),
            fmt="none",
            capsize=3,
            color="#1d4ed8",
        )
    breakpoint = latest.get("breakpoint_analysis")
    if isinstance(breakpoint, dict):
        axes[0].axvline(
            float(breakpoint["breakpoint"]),
            linestyle="--",
            color="#7c2d12",
            linewidth=1.2,
            label=f"exploratory break, ΔBIC={float(breakpoint['bic_improvement']):.1f}",
        )
    axes[0].plot(
        scales,
        [float(row["nonlinearity_frobenius_fraction"]) for row in rows],
        marker="o",
        label="nonlinearity",
        color="#b45309",
    )
    axes[0].set_title("Error Growth vs Adapter Magnitude")
    axes[0].set_xlabel("Magnitude scale")
    axes[0].set_ylabel("relative error")
    axes[0].legend(loc="best")
    axes[1].plot(
        scales,
        [float(row["greedy_proxy_acceptance_recovery"]) for row in rows],
        marker="o",
        color="#0f766e",
    )
    axes[1].set_title("Greedy Proxy Recovery vs Adapter Magnitude")
    axes[1].set_xlabel("Magnitude scale")
    axes[1].set_ylabel("greedy proxy acceptance delta")
    return [_save_figure(figure, output_dir / "theory_phase_transition.png")]


def _plot_subspace_sharing(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = _latest_payload(payloads)
    rows = latest.get("rows", [])
    if not rows:
        return []
    adapter_names = sorted(
        {str(row["adapter_a"]) for row in rows} | {str(row["adapter_b"]) for row in rows}
    )
    index = {name: position for position, name in enumerate(adapter_names)}
    matrix = np.ones((len(adapter_names), len(adapter_names)), dtype=np.float64)
    for row in rows:
        i = index[str(row["adapter_a"])]
        j = index[str(row["adapter_b"])]
        value = float(row["mean_cosine"])
        matrix[i, j] = value
        matrix[j, i] = value
    figure, axis = plt.subplots(figsize=(6 + 0.4 * len(adapter_names), 5))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(np.arange(len(adapter_names)), adapter_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(adapter_names)), adapter_names)
    unique_sources = int(latest.get("unique_adapter_sources", 0))
    title = "Cross-Adapter Dominant-Subspace Mean Cosine"
    if unique_sources < 2:
        title = "Magnitude-Control Subspace Mean Cosine"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return [_save_figure(figure, output_dir / "theory_subspace_sharing.png")]


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "plot_results")
    config_data = resolve_config(args.config, args.override)
    output_dir = ensure_dir(str(get_config_value(config_data, args, "output_dir")))
    input_dir = Path(str(get_config_value(config_data, args, "input_dir")))
    glob_pattern = str(get_config_value(config_data, args, "glob"))
    explicit_inputs = get_config_value(config_data, args, "input_json", args.input_json)
    if isinstance(explicit_inputs, str):
        explicit_inputs = [explicit_inputs]
    input_paths = _discover_inputs(input_dir, glob_pattern, list(explicit_inputs or []))
    if not input_paths:
        raise ValueError("No result JSON files were found for plotting")

    categorized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_paths: dict[str, list[str]] = defaultdict(list)
    for path in input_paths:
        payload = _load_payload(path)
        category = _classify_payload(payload)
        if category == "unknown":
            continue
        categorized[category].append(payload)
        used_paths[category].append(str(path))

    plot_paths: list[str] = []
    plot_paths.extend(str(path) for path in _plot_phase1(categorized["phase1"], output_dir))
    plot_paths.extend(
        str(path) for path in _plot_characterize(categorized["characterize"], output_dir)
    )
    plot_paths.extend(str(path) for path in _plot_predictive(categorized["predictive"], output_dir))
    plot_paths.extend(str(path) for path in _plot_correction(categorized["correction"], output_dir))
    plot_paths.extend(str(path) for path in _plot_serving(categorized["serving"], output_dir))
    plot_paths.extend(
        str(path)
        for path in _plot_measure_logit_shift_rank(
            categorized["measure_logit_shift_rank"], output_dir
        )
    )
    plot_paths.extend(
        str(path)
        for path in _plot_validate_correction_theory(
            categorized["validate_correction_theory"], output_dir
        )
    )
    plot_paths.extend(
        str(path)
        for path in _plot_phase_transition_sweep(categorized["phase_transition_sweep"], output_dir)
    )
    plot_paths.extend(
        str(path) for path in _plot_subspace_sharing(categorized["subspace_sharing"], output_dir)
    )

    if not plot_paths:
        raise ValueError("No supported result payloads were found for plotting")

    manifest = {
        "plots": plot_paths,
        "plot_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in plot_paths
        },
        "input_files_by_category": dict(used_paths),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_paths
            if str(path) in {used for paths in used_paths.values() for used in paths}
        },
    }
    result_path = write_json_result(
        payload=manifest,
        output_dir=output_dir,
        stem="plot_manifest",
        config={
            "input_dir": str(input_dir),
            "glob": glob_pattern,
            "explicit_inputs": explicit_inputs,
            "input_sha256": manifest["input_sha256"],
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved %d figures to %s", len(plot_paths), output_dir)
    logger.info("Saved plot manifest to %s", result_path)


if __name__ == "__main__":
    main()
