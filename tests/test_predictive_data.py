from __future__ import annotations

from lora_spec.predictive_data import build_predictive_rows


def test_build_predictive_rows_joins_immutable_artifacts() -> None:
    adapter = {"source": "org/adapter", "resolved_revision": "adapter-sha"}
    target = {"source": "org/llama", "resolved_revision": "target-sha"}
    runs = []
    for index in range(3):
        runs.append(
            {
                "model_pair_name": "llama_pair",
                "adapter_name": f"adapter_{index}",
                "artifact_provenance": {"adapter": adapter, "target_model": target},
                "experiment": {"adapter": {"rank": 8, "magnitude_scale": 1.0}},
                "result": {"comparison": {"acceptance_delta": -0.1 - index * 0.01}},
                "config_hash": f"run-{index}",
            }
        )
    property_index = {
        ("org/adapter", "adapter-sha", "org/llama", "target-sha", 1.0): {
            "properties": {
                "frobenius_norm_sum": 1.0,
                "spectral_norm_sum": 2.0,
                "max_spectral_norm": 0.5,
                "adapted_parameter_fraction": 0.01,
            },
            "divergence": {"kl_divergence": 0.2, "js_divergence": 0.1},
            "config_hash": "props",
            "source_path": "props.json",
            "source_sha256": "a" * 64,
        }
    }

    rows = build_predictive_rows({"runs": runs}, property_index)

    assert len(rows) == 3
    assert rows[0]["target"] == 0.1
    assert rows[0]["adapter_source"] == "org/adapter@adapter-sha"
    assert rows[0]["model_family"] == "llama"
    assert len(rows[0]["features"]) == 8
    assert rows[0]["adapter_properties_sha256"] == "a" * 64
