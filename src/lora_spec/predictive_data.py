from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


FEATURE_NAMES = [
    "frobenius_norm_sum",
    "spectral_norm_sum",
    "max_spectral_norm",
    "adapted_parameter_fraction",
    "calibration_kl_divergence",
    "calibration_js_divergence",
    "adapter_rank",
    "magnitude_scale",
]


def _artifact_key(provenance: dict[str, Any]) -> tuple[str, str]:
    return str(provenance["source"]), str(provenance["resolved_revision"])


def model_family(model_pair_name: str, target_source: str) -> str:
    lowered = f"{model_pair_name} {target_source}".lower()
    for family in ("llama", "qwen", "gemma", "mistral"):
        if family in lowered:
            return family
    return model_pair_name.split("_", 1)[0]


def load_property_index(
    directory: str | Path,
) -> dict[tuple[str, str, str, str, float], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, float], dict[str, Any]] = {}
    for path in sorted(Path(directory).glob("adapter_props_*.json")):
        source_bytes = path.read_bytes()
        payload = json.loads(source_bytes.decode("utf-8"))
        config = payload.get("full_config", {})
        provenance = config.get("artifact_provenance", {})
        adapter = provenance.get("adapter")
        base = provenance.get("base_model")
        if not isinstance(adapter, dict) or not isinstance(base, dict):
            continue
        adapter_source, adapter_revision = _artifact_key(adapter)
        base_source, base_revision = _artifact_key(base)
        key = (
            adapter_source,
            adapter_revision,
            base_source,
            base_revision,
            float(config.get("magnitude_scale", 1.0)),
        )
        if key in index:
            raise ValueError(f"Duplicate adapter-property artifact for key {key}")
        index[key] = {
            **payload,
            "source_path": str(path),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    if not index:
        raise ValueError(f"No usable adapter-property artifacts found in {directory}")
    return index


def build_predictive_rows(
    characterize_payload: dict[str, Any],
    property_index: dict[tuple[str, str, str, str, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in characterize_payload.get("runs", []):
        provenance = run.get("artifact_provenance", {})
        adapter = provenance.get("adapter")
        target = provenance.get("target_model")
        experiment = run.get("experiment", {})
        adapter_config = experiment.get("adapter", {})
        if not isinstance(adapter, dict) or not isinstance(target, dict):
            raise ValueError("Characterization run is missing immutable artifact provenance")
        adapter_source, adapter_revision = _artifact_key(adapter)
        target_source, target_revision = _artifact_key(target)
        magnitude_scale = float(adapter_config.get("magnitude_scale", 1.0))
        key = (
            adapter_source,
            adapter_revision,
            target_source,
            target_revision,
            magnitude_scale,
        )
        if key not in property_index:
            raise KeyError(f"No adapter-property artifact matches characterization key {key}")
        property_payload = property_index[key]
        property_sha256 = property_payload.get("source_sha256")
        if not isinstance(property_sha256, str) or len(property_sha256) != 64:
            raise ValueError(f"Adapter-property artifact lacks a source SHA-256 for key {key}")
        properties = property_payload["properties"]
        divergence = property_payload.get("divergence")
        if not isinstance(divergence, dict):
            raise ValueError(
                f"Adapter-property artifact lacks calibration divergence for key {key}"
            )
        comparison = run["result"]["comparison"]
        rows.append(
            {
                "features": [
                    float(properties["frobenius_norm_sum"]),
                    float(properties["spectral_norm_sum"]),
                    float(properties["max_spectral_norm"]),
                    float(properties["adapted_parameter_fraction"]),
                    float(divergence["kl_divergence"]),
                    float(divergence["js_divergence"]),
                    float(adapter_config["rank"]),
                    magnitude_scale,
                ],
                "target": -float(comparison["acceptance_delta"]),
                "adapter_source": f"{adapter_source}@{adapter_revision}",
                "model_family": model_family(str(run["model_pair_name"]), target_source),
                "model_pair_name": run["model_pair_name"],
                "adapter_name": run["adapter_name"],
                "magnitude_scale": magnitude_scale,
                "characterization_config_hash": run.get("config_hash"),
                "adapter_properties_config_hash": property_payload.get("config_hash"),
                "adapter_properties_path": property_payload["source_path"],
                "adapter_properties_sha256": property_sha256,
            }
        )
    if len(rows) < 3:
        raise ValueError("Predictive dataset requires at least three matched characterization runs")
    return rows
