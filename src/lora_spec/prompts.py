from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PromptSplitName = Literal["calibration", "evaluation"]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FrozenPromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: PromptSplitName
    domain: str = Field(min_length=1)
    task: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("id", "domain", "task", "source", "text")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace characters")
        return value


class PromptSplitEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: Sha256
    records: int = Field(ge=1)
    domains: dict[str, Annotated[int, Field(ge=1)]]

    @field_validator("path")
    @classmethod
    def reject_unsafe_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.name:
            raise ValueError("split path must be a filename in the manifest directory")
        return value

    @field_validator("domains")
    @classmethod
    def reject_empty_domains(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("domains must not be empty")
        if any(not domain.strip() for domain in value):
            raise ValueError("domain names must not be blank")
        return value


class PromptOverlapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ngram_size: int = Field(ge=1)
    maximum_cross_split_jaccard: float = Field(ge=0.0, le=1.0)


class PromptManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str = Field(min_length=1)
    created_at: date
    description: str = Field(min_length=1)
    construction: str = Field(min_length=1)
    normalization: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    overlap_policy: PromptOverlapPolicy
    splits: dict[PromptSplitName, PromptSplitEntry]

    @field_validator("name", "description", "construction", "normalization", "source_license")
    @classmethod
    def reject_blank_manifest_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace characters")
        return value


class PromptReleaseLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    manifest_name: str = Field(min_length=1)
    manifest_sha256: Sha256
    split_sha256: dict[PromptSplitName, Sha256]


@dataclass(frozen=True)
class PromptSplitVerification:
    manifest_path: Path
    manifest_name: str
    manifest_sha256: str
    release_lock_path: Path
    release_lock_sha256: str
    split_paths: dict[PromptSplitName, Path]
    split_hashes: dict[PromptSplitName, str]
    split_counts: dict[PromptSplitName, int]
    domain_counts: dict[PromptSplitName, dict[str, int]]
    maximum_cross_split_ngram_jaccard: float
    total_records: int


@dataclass(frozen=True)
class RegisteredPromptSplit:
    name: PromptSplitName
    path: Path
    sha256: str
    records: list[FrozenPromptRecord]
    verification: PromptSplitVerification


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json(raw: str, source: str) -> object:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _word_ngrams(text: str, ngram_size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[\w]+", _normalized_text(text), flags=re.UNICODE)
    if len(tokens) < ngram_size:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)
    }


def _maximum_cross_split_jaccard(
    calibration: list[FrozenPromptRecord],
    evaluation: list[FrozenPromptRecord],
    ngram_size: int,
) -> float:
    maximum = 0.0
    calibration_ngrams = [
        (_word_ngrams(record.text, ngram_size), record.id) for record in calibration
    ]
    evaluation_ngrams = [
        (_word_ngrams(record.text, ngram_size), record.id) for record in evaluation
    ]
    for calibration_set, _ in calibration_ngrams:
        for evaluation_set, _ in evaluation_ngrams:
            union = calibration_set | evaluation_set
            score = len(calibration_set & evaluation_set) / len(union) if union else 1.0
            maximum = max(maximum, score)
    return maximum


def load_prompt_records(path: str | Path) -> list[FrozenPromptRecord]:
    prompt_path = Path(path)
    if prompt_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Frozen prompt files must use JSONL: {prompt_path}")
    records: list[FrozenPromptRecord] = []
    for line_number, line in enumerate(
        prompt_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"Blank JSONL line in {prompt_path}:{line_number}")
        payload = _load_json(line, f"{prompt_path}:{line_number}")
        try:
            records.append(FrozenPromptRecord.model_validate(payload))
        except ValueError as exc:
            raise ValueError(
                f"Invalid prompt record in {prompt_path}:{line_number}: {exc}"
            ) from exc
    if not records:
        raise ValueError(f"No prompt records found in {prompt_path}")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
        raise ValueError(f"Duplicate prompt IDs in {prompt_path}: {duplicates}")
    normalized = [_normalized_text(record.text) for record in records]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate normalized prompt text in {prompt_path}")
    return records


def load_prompt_manifest(path: str | Path) -> PromptManifest:
    manifest_path = Path(path)
    payload = _load_json(manifest_path.read_text(encoding="utf-8"), str(manifest_path))
    return PromptManifest.model_validate(payload)


def load_prompt_release_lock(path: str | Path) -> PromptReleaseLock:
    lock_path = Path(path)
    payload = _load_json(lock_path.read_text(encoding="utf-8"), str(lock_path))
    return PromptReleaseLock.model_validate(payload)


def _validate_prompt_release_lock(
    manifest_path: str | Path,
    manifest: PromptManifest,
    lock_path: str | Path | None = None,
) -> tuple[Path, str, str]:
    resolved_manifest = Path(manifest_path).resolve()
    resolved_lock = (
        Path(lock_path).resolve() if lock_path else resolved_manifest.parent / "release.lock.json"
    )
    if not resolved_lock.exists():
        raise FileNotFoundError(f"Frozen prompt release lock not found: {resolved_lock}")
    release_lock = load_prompt_release_lock(resolved_lock)
    actual_manifest_hash = file_sha256(resolved_manifest)
    if release_lock.manifest_name != manifest.name:
        raise ValueError(
            f"Release lock names {release_lock.manifest_name}, but manifest names {manifest.name}",
        )
    if release_lock.manifest_sha256 != actual_manifest_hash:
        raise ValueError(
            "Prompt manifest does not match its release lock: "
            f"expected {release_lock.manifest_sha256}, got {actual_manifest_hash}",
        )
    manifest_split_hashes = {
        split_name: entry.sha256 for split_name, entry in manifest.splits.items()
    }
    if release_lock.split_sha256 != manifest_split_hashes:
        raise ValueError("Prompt split hashes do not match the release lock")
    return resolved_lock, actual_manifest_hash, file_sha256(resolved_lock)


def verify_prompt_release_lock(
    manifest_path: str | Path,
    lock_path: str | Path | None = None,
) -> None:
    resolved_manifest = Path(manifest_path).resolve()
    manifest = load_prompt_manifest(resolved_manifest)
    _validate_prompt_release_lock(resolved_manifest, manifest, lock_path)


def verify_prompt_manifest(path: str | Path) -> PromptSplitVerification:
    manifest_path = Path(path).resolve()
    manifest = load_prompt_manifest(manifest_path)
    release_lock_path, manifest_sha256, release_lock_sha256 = _validate_prompt_release_lock(
        manifest_path,
        manifest,
    )
    all_ids: set[str] = set()
    all_texts: set[str] = set()
    split_hashes: dict[PromptSplitName, str] = {}
    split_paths: dict[PromptSplitName, Path] = {}
    split_counts: dict[PromptSplitName, int] = {}
    domain_counts: dict[PromptSplitName, dict[str, int]] = {}
    records_by_split: dict[PromptSplitName, list[FrozenPromptRecord]] = {}

    for split_name in ("calibration", "evaluation"):
        entry = manifest.splits.get(split_name)
        if entry is None:
            raise ValueError(f"Manifest is missing required split: {split_name}")
        split_path = (manifest_path.parent / entry.path).resolve()
        if split_path.parent != manifest_path.parent:
            raise ValueError(f"Prompt split must remain in manifest directory: {entry.path}")
        actual_hash = file_sha256(split_path)
        if actual_hash != entry.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {split_name}: expected {entry.sha256}, got {actual_hash}",
            )
        records = load_prompt_records(split_path)
        if any(record.split != split_name for record in records):
            raise ValueError(f"Split label mismatch in {split_path}")
        if len(records) != entry.records:
            raise ValueError(
                f"Record count mismatch for {split_name}: expected {entry.records}, got {len(records)}",
            )
        actual_domains = dict(sorted(Counter(record.domain for record in records).items()))
        if actual_domains != dict(sorted(entry.domains.items())):
            raise ValueError(
                f"Domain count mismatch for {split_name}: expected {entry.domains}, got {actual_domains}",
            )
        split_ids = {record.id for record in records}
        split_texts = {_normalized_text(record.text) for record in records}
        overlapping_ids = all_ids & split_ids
        overlapping_texts = all_texts & split_texts
        if overlapping_ids:
            raise ValueError(f"Prompt IDs overlap across splits: {sorted(overlapping_ids)}")
        if overlapping_texts:
            raise ValueError("Normalized prompt text overlaps across splits")
        all_ids.update(split_ids)
        all_texts.update(split_texts)
        split_hashes[split_name] = actual_hash
        split_paths[split_name] = split_path
        split_counts[split_name] = len(records)
        domain_counts[split_name] = actual_domains
        records_by_split[split_name] = records

    maximum_overlap = _maximum_cross_split_jaccard(
        records_by_split["calibration"],
        records_by_split["evaluation"],
        ngram_size=manifest.overlap_policy.ngram_size,
    )
    if maximum_overlap > manifest.overlap_policy.maximum_cross_split_jaccard:
        raise ValueError(
            "Cross-split prompt overlap exceeds policy: "
            f"observed {maximum_overlap:.6f}, allowed "
            f"{manifest.overlap_policy.maximum_cross_split_jaccard:.6f}",
        )

    return PromptSplitVerification(
        manifest_path=manifest_path,
        manifest_name=manifest.name,
        manifest_sha256=manifest_sha256,
        release_lock_path=release_lock_path,
        release_lock_sha256=release_lock_sha256,
        split_paths=split_paths,
        split_hashes=split_hashes,
        split_counts=split_counts,
        domain_counts=domain_counts,
        maximum_cross_split_ngram_jaccard=maximum_overlap,
        total_records=sum(split_counts.values()),
    )


def resolve_registered_prompt_split(
    path: str | Path,
    expected_split: PromptSplitName,
    manifest_path: str | Path | None = None,
) -> RegisteredPromptSplit:
    prompt_path = Path(path).resolve()
    resolved_manifest = (
        Path(manifest_path).resolve() if manifest_path else prompt_path.parent / "manifest.json"
    )
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Frozen prompt manifest not found: {resolved_manifest}")
    verification = verify_prompt_manifest(resolved_manifest)

    matches: list[PromptSplitName] = []
    for split_name, registered_path in verification.split_paths.items():
        if registered_path == prompt_path:
            matches.append(split_name)
    if not matches:
        raise ValueError(f"Prompt file is not registered by manifest: {prompt_path}")
    if len(matches) != 1:
        raise ValueError(f"Prompt file is registered under multiple split roles: {matches}")
    split_name = matches[0]
    if split_name != expected_split:
        raise ValueError(
            f"Expected a {expected_split} prompt split, but {prompt_path} is registered as {split_name}",
        )
    records = load_prompt_records(prompt_path)
    actual_hash = file_sha256(prompt_path)
    expected_hash = verification.split_hashes[split_name]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Prompt split changed while being loaded: expected {expected_hash}, got {actual_hash}",
        )
    return RegisteredPromptSplit(
        name=split_name,
        path=prompt_path,
        sha256=expected_hash,
        records=records,
        verification=verification,
    )


def load_frozen_prompt_texts(
    path: str | Path,
    expected_split: PromptSplitName,
    manifest_path: str | Path | None = None,
) -> list[str]:
    registered = resolve_registered_prompt_split(
        path,
        manifest_path=manifest_path,
        expected_split=expected_split,
    )
    return [record.text for record in registered.records]


def prompt_file_provenance(
    path: str | Path,
    expected_split: PromptSplitName,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    display_prompt_path = Path(path)
    display_manifest_path = (
        Path(manifest_path) if manifest_path else display_prompt_path.parent / "manifest.json"
    )
    prompt_path = display_prompt_path.resolve()
    resolved_manifest = display_manifest_path.resolve()
    registered = resolve_registered_prompt_split(
        prompt_path,
        manifest_path=resolved_manifest,
        expected_split=expected_split,
    )
    records = registered.records
    verification = registered.verification
    return {
        "path": str(display_prompt_path),
        "sha256": registered.sha256,
        "split": registered.name,
        "records": len(records),
        "domains": dict(sorted(Counter(record.domain for record in records).items())),
        "manifest_path": str(display_manifest_path),
        "manifest_name": verification.manifest_name,
        "manifest_sha256": verification.manifest_sha256,
        "release_lock_path": str(
            display_manifest_path.parent / verification.release_lock_path.name
        ),
        "release_lock_sha256": verification.release_lock_sha256,
    }


def select_frozen_prompts(
    path: str | Path,
    expected_split: PromptSplitName,
    num_prompts: int,
    seed: int,
    manifest_path: str | Path | None = None,
) -> tuple[list[str], dict[str, object]]:
    if num_prompts < 1:
        raise ValueError("num_prompts must be positive")
    registered = resolve_registered_prompt_split(
        path,
        expected_split=expected_split,
        manifest_path=manifest_path,
    )
    if num_prompts > len(registered.records):
        raise ValueError(
            f"Requested {num_prompts} prompts, but frozen {expected_split} split has "
            f"{len(registered.records)}",
        )
    indices = list(range(len(registered.records)))
    random.Random(seed).shuffle(indices)
    selected = [registered.records[index] for index in indices[:num_prompts]]
    provenance = {
        **prompt_file_provenance(
            path,
            expected_split=expected_split,
            manifest_path=manifest_path,
        ),
        "selected_prompt_ids": [record.id for record in selected],
        "selection_seed": seed,
    }
    return [record.text for record in selected], provenance
