from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_prompt_verification_script_runs_from_source_checkout_without_editable_install() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/verify_prompt_splits.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified" in result.stderr


def test_rank_script_help_does_not_require_hf_experiment_dependencies() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/measure_logit_shift_rank.py", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--synthetic-smoke-test" in result.stdout


def test_rank_script_synthetic_smoke_test_writes_result(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    output_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/measure_logit_shift_rank.py",
            "--synthetic-smoke-test",
            "--seed",
            "13",
            "--output-dir",
            str(output_dir),
            "--plots-dir",
            str(plots_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    result_files = sorted(
        path
        for path in output_dir.glob("measure_logit_shift_rank_smoke_*.json")
        if not path.name.endswith(".source.json")
    )
    assert len(result_files) == 1
    assert result_files[0].with_name(f"{result_files[0].stem}.source.json").exists()
    assert (plots_dir / "synthetic_smoke__spectrum.png").exists()
