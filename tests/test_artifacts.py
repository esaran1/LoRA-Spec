from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lora_spec.artifacts import resolve_artifact_revision, tokenizers_are_equivalent


class FakeTokenizer:
    def __init__(self, vocabulary: dict[str, int], eos_token_id: int = 2) -> None:
        self.vocabulary = vocabulary
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.unk_token_id = 3
        self.mask_token_id = None
        self.sep_token_id = None
        self.cls_token_id = None

    def __len__(self) -> int:
        return len(self.vocabulary)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def __call__(self, prompt: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [self.vocabulary.get(token, self.unk_token_id) for token in prompt.split()]
        return {"input_ids": ([self.bos_token_id] + ids if add_special_tokens else ids)}


def test_tokenizer_equivalence_rejects_equal_size_permuted_vocabulary() -> None:
    reference = FakeTokenizer({"a": 0, "b": 1, "c": 2, "d": 3})
    candidate = FakeTokenizer({"a": 1, "b": 0, "c": 2, "d": 3})
    assert not tokenizers_are_equivalent(reference, candidate, ["a b"])
    assert tokenizers_are_equivalent(reference, FakeTokenizer(reference.vocabulary), ["a b"])


def test_resolve_artifact_revision_records_remote_commit() -> None:
    class FakeApi:
        def repo_info(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs == {"repo_id": "org/model", "revision": "release", "repo_type": "model"}
            return SimpleNamespace(sha="a" * 40)

    provenance = resolve_artifact_revision("org/model", revision="release", api=FakeApi())
    assert provenance.resolved_revision == "a" * 40
    assert provenance.requested_revision == "release"


def test_resolve_artifact_revision_hashes_local_directory(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"weights")
    first = resolve_artifact_revision(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"changed")
    second = resolve_artifact_revision(tmp_path)
    assert first.repository_type == "local"
    assert first.resolved_revision != second.resolved_revision
