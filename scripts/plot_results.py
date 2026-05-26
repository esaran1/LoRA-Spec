from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lora_spec.utils import add_common_args, ensure_dir, get_config_value, resolve_config, setup_logging, write_json_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LoRA-Spec figures from result JSON artifacts.")
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


def _classify_payload(payload: dict[str, Any]) -> str:
    if "runs" in payload and isinstance(payload["runs"], list):
        return "characterize"
    if {"linear", "multivariate", "mlp"} <= payload.keys():
        return "predictive"
    if {"distribution_offset", "low_rank", "jacobian", "baseline"} <= payload.keys():
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
    latest = payloads[-1]
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
            adapter = experiment.get("adapter", {})
            rows.append(
                {
                    "model_pair_name": run.get("model_pair_name"),
                    "adapter_name": run.get("adapter_name"),
                    "rank": adapter.get("rank"),
                    "domain": adapter.get("domain"),
                    "epochs": adapter.get("epochs"),
                    "baseline_acceptance": baseline.get("acceptance_rate_overall"),
                    "adapted_acceptance": adapted.get("acceptance_rate_overall"),
                    "acceptance_delta": comparison.get("acceptance_delta"),
                    "baseline_throughput": baseline.get("throughput_tps"),
                    "adapted_throughput": adapted.get("throughput_tps"),
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

    rows_sorted = sorted(rows, key=lambda row: (int(row["rank"] or 0), str(row["domain"]), str(row["model_pair_name"])))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ranks = [int(row["rank"]) for row in rows_sorted]
    acceptance_delta = [float(row["acceptance_delta"]) for row in rows_sorted]
    throughput_delta = [float(row["throughput_delta_tps"]) for row in rows_sorted]
    colors = {"medical": "#0f766e", "code": "#1d4ed8", "chat": "#b45309", "math": "#7c3aed"}
    for domain in sorted({str(row["domain"]) for row in rows_sorted}):
        domain_rows = [row for row in rows_sorted if str(row["domain"]) == domain]
        axes[0].plot(
            [int(row["rank"]) for row in domain_rows],
            [float(row["acceptance_delta"]) for row in domain_rows],
            marker="o",
            label=domain,
            color=colors.get(domain, None),
        )
        axes[1].plot(
            [int(row["rank"]) for row in domain_rows],
            [float(row["throughput_delta_tps"]) for row in domain_rows],
            marker="o",
            label=domain,
            color=colors.get(domain, None),
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Acceptance Delta vs Rank")
    axes[0].set_xlabel("LoRA rank")
    axes[0].set_ylabel("adapted - baseline")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Throughput Delta vs Rank")
    axes[1].set_xlabel("LoRA rank")
    axes[1].set_ylabel("tokens/sec delta")
    axes[1].legend(loc="best")
    paths.append(_save_figure(figure, output_dir / "characterize_rank_trends.png"))

    max_position = max(len(row["baseline_per_position"]) for row in rows_sorted)
    if max_position > 0:
        baseline_matrix = []
        adapted_matrix = []
        for row in rows_sorted:
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
    latest = payloads[-1]
    models = [
        ("linear_loocv", "Linear LOOCV", "#4f46e5"),
        ("multivariate_loocv", "Multivariate LOOCV", "#0f766e"),
        ("mlp_loocv", "MLP LOOCV", "#b45309"),
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
    paths.append(_save_figure(figure, output_dir / "predictive_loocv_scatter.png"))
    return paths


def _plot_correction(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not payloads:
        return []
    latest = payloads[-1]
    labels = ["baseline", "distribution_offset", "low_rank", "jacobian"]
    kl_values = [float(latest[label]["kl_divergence"]) for label in labels if label in latest]
    js_values = [float(latest[label]["js_divergence"]) for label in labels if label in latest]
    names = [label.replace("_", "\n") for label in labels if label in latest]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(names, kl_values, color=["#64748b", "#0f766e", "#1d4ed8", "#b45309"][: len(names)])
    axes[0].set_title("Correction KL Divergence")
    axes[0].set_ylabel("KL")
    axes[1].bar(names, js_values, color=["#64748b", "#0f766e", "#1d4ed8", "#b45309"][: len(names)])
    axes[1].set_title("Correction JSD")
    axes[1].set_ylabel("JSD")
    paths = [_save_figure(figure, output_dir / "correction_divergence_bars.png")]

    proxy = latest.get("speculative_proxy")
    if isinstance(proxy, dict):
        proxy_labels = []
        acceptance_values = []
        tptc_values = []
        for key in labels:
            if key not in proxy:
                continue
            proxy_labels.append(key.replace("_", "\n"))
            acceptance_values.append(float(proxy[key]["acceptance_rate_overall"]))
            tptc_values.append(float(proxy[key]["tokens_per_target_call"]))
        if proxy_labels:
            figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            palette = ["#64748b", "#0f766e", "#1d4ed8", "#b45309"][: len(proxy_labels)]
            axes[0].bar(proxy_labels, acceptance_values, color=palette)
            axes[0].set_title("Acceptance Recovery Proxy")
            axes[0].set_ylabel("overall acceptance")
            axes[0].set_ylim(0, 1)
            axes[1].bar(proxy_labels, tptc_values, color=palette)
            axes[1].set_title("Tokens per Target Call")
            axes[1].set_ylabel("proxy throughput")
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
    axes[0].bar(x - width / 2, [row["baseline_tps"] for row in rows], width=width, label="baseline", color="#5b7c99")
    axes[0].bar(x + width / 2, [row["adapted_tps"] for row in rows], width=width, label="adapted", color="#d55c4b")
    axes[0].set_title("Serving Throughput by Traffic Pattern")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("tokens/sec")
    axes[0].legend(loc="best")
    axes[1].bar(x - width / 2, [row["baseline_p95"] for row in rows], width=width, label="baseline", color="#5b7c99")
    axes[1].bar(x + width / 2, [row["adapted_p95"] for row in rows], width=width, label="adapted", color="#d55c4b")
    axes[1].set_title("Serving p95 Latency by Traffic Pattern")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("ms")
    axes[1].legend(loc="best")
    return [_save_figure(figure, output_dir / "serving_pattern_comparison.png")]


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
    plot_paths.extend(str(path) for path in _plot_characterize(categorized["characterize"], output_dir))
    plot_paths.extend(str(path) for path in _plot_predictive(categorized["predictive"], output_dir))
    plot_paths.extend(str(path) for path in _plot_correction(categorized["correction"], output_dir))
    plot_paths.extend(str(path) for path in _plot_serving(categorized["serving"], output_dir))

    if not plot_paths:
        raise ValueError("No supported result payloads were found for plotting")

    manifest = {
        "plots": plot_paths,
        "input_files_by_category": dict(used_paths),
    }
    result_path = write_json_result(
        payload=manifest,
        output_dir=output_dir,
        stem="plot_manifest",
        config={
            "input_dir": str(input_dir),
            "glob": glob_pattern,
            "explicit_inputs": explicit_inputs,
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved %d figures to %s", len(plot_paths), output_dir)
    logger.info("Saved plot manifest to %s", result_path)


if __name__ == "__main__":
    main()
