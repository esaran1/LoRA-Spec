from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_plot_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
    spec = importlib.util.spec_from_file_location("plot_results_test_module", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load plot_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_characterization_extraction_reads_nested_experiment_config() -> None:
    module = _load_plot_module()
    rows = module._extract_characterize_rows(
        [
            {
                "runs": [
                    {
                        "model_pair_name": "pair",
                        "adapter_name": "adapter",
                        "experiment": {
                            "experiment": {"adapter": {"rank": 8, "domain": "code", "epochs": 1}}
                        },
                        "result": {
                            "baseline": {"acceptance_rate_per_position": [0.9]},
                            "adapted": {"acceptance_rate_per_position": [0.8]},
                            "comparison": {
                                "acceptance_delta": -0.1,
                                "throughput_delta_tps": -2.0,
                            },
                        },
                    }
                ]
            }
        ]
    )

    assert rows[0]["rank"] == 8
    assert rows[0]["domain"] == "code"
    assert rows[0]["epochs"] == 1
