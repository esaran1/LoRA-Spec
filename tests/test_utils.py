from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_spec import utils
from lora_spec.utils import mean_ci95, write_json_result


def test_write_json_result_normalizes_nonfinite_values(tmp_path: Path) -> None:
    output = write_json_result(
        payload={"nan": float("nan"), "positive_infinity": float("inf")},
        output_dir=tmp_path,
        stem="finite_json",
        config={"seed": 7},
        cwd=tmp_path,
    )
    raw = output.read_text(encoding="utf-8")
    parsed = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert parsed["nan"] is None
    assert parsed["positive_infinity"] is None


def test_write_json_result_supports_exact_resume_path(tmp_path: Path) -> None:
    exact_path = tmp_path / "runs" / "fixed.json"
    output = write_json_result(
        payload={"value": 1},
        output_dir=tmp_path,
        stem="ignored",
        config={"seed": 3},
        cwd=tmp_path,
        exact_path=exact_path,
    )
    assert output == exact_path
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["metadata"]["timestamp"]
    assert parsed["full_config"] == {"seed": 3}


def test_mean_ci95_is_centered_on_sample_mean() -> None:
    mean, lower, upper = mean_ci95([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert lower < mean < upper
    assert mean - lower == upper - mean


def test_result_writer_does_not_leave_temporary_files(tmp_path: Path) -> None:
    output = write_json_result(
        payload={"value": 1},
        output_dir=tmp_path,
        stem="atomic",
        config={"seed": 1},
    )

    assert output.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert len(output.stem.split("_")) >= 3


def test_result_writer_captures_dirty_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        utils,
        "capture_git_source_snapshot",
        lambda cwd=None: {"format": "test", "tracked_patch_base64": "AA=="},
    )
    output = write_json_result(
        payload={"value": 1},
        output_dir=tmp_path,
        stem="snapshot",
        config={"seed": 1},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    snapshot_path = output.parent / payload["metadata"]["source_snapshot_path"]
    assert snapshot_path.exists()
    assert len(payload["metadata"]["source_snapshot_sha256"]) == 64


def test_source_snapshot_rejects_sensitive_untracked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env.local").write_text("TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(utils, "get_git_dirty", lambda cwd=None: True)

    def fake_check_output(command: list[str], **kwargs: object) -> bytes | str:
        _ = kwargs
        if command[:3] == ["git", "diff", "--binary"]:
            return b""
        if command[:3] == ["git", "ls-files", "--others"]:
            return b".env.local\0"
        if command[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return str(tmp_path)
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(utils.subprocess, "check_output", fake_check_output)

    with pytest.raises(ValueError, match="potentially sensitive"):
        utils.capture_git_source_snapshot(tmp_path)
