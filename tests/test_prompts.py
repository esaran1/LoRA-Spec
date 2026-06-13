from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from lora_spec.prompts import (
    load_frozen_prompt_texts,
    load_prompt_records,
    prompt_file_provenance,
    verify_prompt_manifest,
    verify_prompt_release_lock,
)


def _write_split(path: Path, split: str, texts: list[str]) -> str:
    lines = [
        json.dumps(
            {
                "id": f"{split}-{index}",
                "split": split,
                "domain": "test",
                "task": "test",
                "source": "unit-test",
                "text": text,
            },
            separators=(",", ":"),
        )
        for index, text in enumerate(texts)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    root: Path,
    calibration_texts: list[str] | None = None,
    evaluation_texts: list[str] | None = None,
    maximum_cross_split_jaccard: float = 0.1,
) -> Path:
    calibration_values = calibration_texts or ["alpha"]
    evaluation_values = evaluation_texts or ["beta", "gamma"]
    calibration_hash = _write_split(
        root / "calibration.jsonl",
        "calibration",
        calibration_values,
    )
    evaluation_hash = _write_split(
        root / "evaluation.jsonl",
        "evaluation",
        evaluation_values,
    )
    manifest = {
        "schema_version": 1,
        "name": "unit-test-v1",
        "created_at": "2026-06-13",
        "description": "test",
        "construction": "test",
        "normalization": "UTF-8 JSONL with LF endings",
        "source_license": "test",
        "overlap_policy": {
            "ngram_size": 3,
            "maximum_cross_split_jaccard": maximum_cross_split_jaccard,
        },
        "splits": {
            "calibration": {
                "path": "calibration.jsonl",
                "sha256": calibration_hash,
                "records": len(calibration_values),
                "domains": {"test": len(calibration_values)},
            },
            "evaluation": {
                "path": "evaluation.jsonl",
                "sha256": evaluation_hash,
                "records": len(evaluation_values),
                "domains": {"test": len(evaluation_values)},
            },
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    release_lock = {
        "schema_version": 1,
        "manifest_name": manifest["name"],
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_sha256": {
            "calibration": calibration_hash,
            "evaluation": evaluation_hash,
        },
    }
    (root / "release.lock.json").write_text(
        json.dumps(release_lock, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_verify_prompt_manifest_and_load_texts(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    verification = verify_prompt_manifest(manifest_path)
    assert verification.total_records == 3
    assert verification.split_counts == {"calibration": 1, "evaluation": 2}
    assert load_frozen_prompt_texts(
        tmp_path / "evaluation.jsonl",
        expected_split="evaluation",
    ) == ["beta", "gamma"]


def test_verify_prompt_manifest_rejects_modified_file(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    with (tmp_path / "calibration.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_prompt_manifest(manifest_path)


def test_load_frozen_prompts_enforces_split_role(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="Expected a calibration prompt split"):
        load_frozen_prompt_texts(
            tmp_path / "evaluation.jsonl",
            expected_split="calibration",
        )


def test_provenance_rejects_unregistered_file(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    unregistered = tmp_path / "unregistered.jsonl"
    _write_split(unregistered, "calibration", ["unregistered"])
    with pytest.raises(ValueError, match="not registered"):
        prompt_file_provenance(
            unregistered,
            expected_split="calibration",
            manifest_path=manifest_path,
        )


def test_release_lock_rejects_manifest_edit(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["description"] = "silently changed"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its release lock"):
        verify_prompt_release_lock(manifest_path)


def test_manifest_rejects_near_duplicate_cross_split_prompts(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        calibration_texts=["one two three four"],
        evaluation_texts=["one two three five"],
        maximum_cross_split_jaccard=0.1,
    )
    with pytest.raises(ValueError, match="overlap exceeds policy"):
        verify_prompt_manifest(manifest_path)


def test_prompt_records_reject_blank_lines_and_duplicate_keys(tmp_path: Path) -> None:
    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_text(
        '{"id":"x","split":"calibration","domain":"x","task":"x",'
        '"source":"x","text":"valid"}\n\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Blank JSONL line"):
        load_prompt_records(blank_path)

    duplicate_key_path = tmp_path / "duplicate.jsonl"
    duplicate_key_path.write_text(
        '{"id":"x","id":"y","split":"calibration","domain":"x",'
        '"task":"x","source":"x","text":"valid"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_prompt_records(duplicate_key_path)


def test_checked_in_pilot_manifest_is_valid() -> None:
    manifest_path = Path("data/prompts/pilot_v1/manifest.json")
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "22d6c1a462f56a48dbf9ba20195bb6c89b4b59d33375aef3594986d776908abe"
    )
    lock_path = Path("data/prompts/pilot_v1/release.lock.json")
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == (
        "c434db374a1b169168bcd069b139ef73483e048089c60761980879161f47dde2"
    )
    verify_prompt_release_lock(manifest_path)
    verification = verify_prompt_manifest(manifest_path)
    assert verification.split_counts == {"calibration": 16, "evaluation": 32}
    assert verification.total_records == 48
    provenance = prompt_file_provenance(
        "data/prompts/pilot_v1/calibration.jsonl",
        expected_split="calibration",
    )
    assert provenance["path"] == "data/prompts/pilot_v1/calibration.jsonl"
    assert provenance["manifest_path"] == "data/prompts/pilot_v1/manifest.json"
    assert provenance["split"] == "calibration"
    assert verification.maximum_cross_split_ngram_jaccard < 0.1

    registry = yaml.safe_load(Path("configs/prompts.yaml").read_text(encoding="utf-8"))
    pilot = registry["prompt_splits"]["pilot_v1"]
    assert pilot["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert pilot["release_lock_sha256"] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert pilot["calibration_sha256"] == verification.split_hashes["calibration"]
    assert pilot["evaluation_sha256"] == verification.split_hashes["evaluation"]
