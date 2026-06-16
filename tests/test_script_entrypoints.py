from __future__ import annotations

import os
import subprocess
import sys


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
