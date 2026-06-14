from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:  # Optional for local-only analysis and CPU tests.
    HfApi = None  # type: ignore[assignment,misc]
    snapshot_download = None  # type: ignore[assignment]


RepositoryType = Literal["model", "dataset", "space"]


@dataclass(frozen=True)
class ArtifactProvenance:
    source: str
    repository_type: RepositoryType | Literal["local"]
    requested_revision: str | None
    resolved_revision: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @property
    def revision_for_loading(self) -> str | None:
        return None if self.repository_type == "local" else self.resolved_revision


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Local artifact directory contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_revision(
    source: str | Path,
    revision: str | None = None,
    repository_type: RepositoryType = "model",
    api: Any | None = None,
) -> ArtifactProvenance:
    artifact_path = Path(source).expanduser()
    if artifact_path.exists():
        resolved_path = artifact_path.resolve()
        resolved_revision = (
            hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            if resolved_path.is_file()
            else _directory_sha256(resolved_path)
        )
        return ArtifactProvenance(
            source=str(source),
            repository_type="local",
            requested_revision=revision,
            resolved_revision=resolved_revision,
        )

    if api is None:
        if HfApi is None:
            raise ImportError("Remote artifact resolution requires the huggingface-hub package")
        api = HfApi()
    client = api
    info = client.repo_info(
        repo_id=str(source),
        revision=revision,
        repo_type=repository_type,
    )
    resolved_revision = getattr(info, "sha", None)
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise ValueError(f"Hugging Face did not return a commit SHA for {source}")
    return ArtifactProvenance(
        source=str(source),
        repository_type=repository_type,
        requested_revision=revision,
        resolved_revision=resolved_revision,
    )


def materialize_artifact(provenance: ArtifactProvenance) -> str:
    source_path = Path(provenance.source).expanduser()
    if provenance.repository_type == "local":
        return str(source_path.resolve())
    if snapshot_download is None:
        raise ImportError("Remote artifact materialization requires the huggingface-hub package")
    return snapshot_download(
        repo_id=provenance.source,
        revision=provenance.resolved_revision,
        repo_type=provenance.repository_type,
    )


def tokenizers_are_equivalent(
    reference: Any,
    candidate: Any,
    probe_prompts: list[str] | None = None,
) -> bool:
    if len(reference) != len(candidate):
        return False
    for field in (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
        "mask_token_id",
        "sep_token_id",
        "cls_token_id",
    ):
        if getattr(reference, field, None) != getattr(candidate, field, None):
            return False
    if not hasattr(reference, "get_vocab") or not hasattr(candidate, "get_vocab"):
        return False
    if reference.get_vocab() != candidate.get_vocab():
        return False
    reference_added = getattr(reference, "get_added_vocab", lambda: {})()
    candidate_added = getattr(candidate, "get_added_vocab", lambda: {})()
    if reference_added != candidate_added:
        return False
    for prompt in probe_prompts or []:
        reference_ids = reference(prompt, add_special_tokens=True)["input_ids"]
        candidate_ids = candidate(prompt, add_special_tokens=True)["input_ids"]
        if reference_ids != candidate_ids:
            return False
    return True
