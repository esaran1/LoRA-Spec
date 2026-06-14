from __future__ import annotations

import json
import os
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from lora_spec.serving import (
    TrafficPatternConfig,
    TrafficRequest,
    _stable_lora_id,
    build_traffic_requests,
    create_openai_server_request_executor,
    initialize_vllm,
    linear_percentile,
    load_and_verify_server_provenance,
    run_concurrent_benchmark,
    warmup_subset,
)


def test_traffic_config_rejects_invalid_measurement_parameters() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        TrafficPatternConfig(pattern="uniform", concurrency=0, requests_per_tenant=1)
    with pytest.raises(ValueError, match="hot_tenant_fraction"):
        TrafficPatternConfig(
            pattern="skewed_80_20",
            concurrency=1,
            requests_per_tenant=1,
            hot_tenant_fraction=0.0,
        )


def test_linear_percentile_matches_type_seven_interpolation() -> None:
    assert linear_percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    assert linear_percentile([3.0], 0.95) == 3.0


def test_stable_lora_id_is_deterministic_and_positive() -> None:
    first = _stable_lora_id("tenant_01")
    second = _stable_lora_id("tenant_01")

    assert first == second
    assert 1 <= first < 2**63
    assert first != _stable_lora_id("tenant_02")


def test_http_executor_selects_preloaded_adapter_by_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"text": "ok"}], "usage": {"completion_tokens": 2}}
            ).encode()

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        _ = timeout
        captured.update(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    executor = create_openai_server_request_executor(
        server_url="http://localhost:8000",
        model="base-model",
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
    )
    text, tokens = executor(
        TrafficRequest(
            request_id="r1",
            tenant_id="tenant_00",
            prompt="hello",
            adapter_model_name="medical-lora",
        )
    )

    assert text == "ok"
    assert tokens == 2
    assert captured["model"] == "medical-lora"
    assert "extra_body" not in captured


def test_benchmark_fails_closed_on_request_error() -> None:
    requests = build_traffic_requests(
        ["prompt"],
        [None],
        TrafficPatternConfig(pattern="uniform", concurrency=1, requests_per_tenant=1),
    )

    def fail(_: TrafficRequest) -> tuple[str, int]:
        raise RuntimeError("server unavailable")

    with pytest.raises(RuntimeError, match="failed 1/1 requests"):
        run_concurrent_benchmark(requests, fail, concurrency=1)


def test_server_provenance_must_match_exact_artifacts(tmp_path: Path) -> None:
    expected = {"target_model": {"resolved_revision": "abc"}, "draft_model": {}, "adapters": []}
    path = tmp_path / "server.json"
    path.write_text(
        json.dumps(
            {
                "vllm_version": "0.15.1",
                "artifact_provenance": expected,
                "adapter_model_names": [],
            }
        ),
        encoding="utf-8",
    )
    payload, digest = load_and_verify_server_provenance(str(path), expected, [])
    assert payload["artifact_provenance"] == expected
    assert len(digest) == 64


def test_warmup_subset_resets_arrival_offsets() -> None:
    requests = build_traffic_requests(
        ["a", "b"],
        [None, None],
        TrafficPatternConfig(pattern="uniform", concurrency=2, requests_per_tenant=2),
    )
    warmup = warmup_subset(requests, per_tenant=1)
    assert len(warmup) == 2
    assert all(request.arrival_offset_s == 0.0 for request in warmup)


def test_initialize_vllm_uses_v1_draft_config_and_in_process_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class LLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = LLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    initialize_vllm("target", "draft", speculation_length=5, enable_lora=True)

    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert captured["enable_lora"] is True
    assert captured["max_lora_rank"] == 16
    assert captured["speculative_config"] == {
        "model": "draft",
        "method": "draft_model",
        "num_speculative_tokens": 5,
    }


def test_initialize_vllm_rounds_rank_four_to_supported_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class LLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = LLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    initialize_vllm("target", "draft", enable_lora=True, max_lora_rank=4)
    assert captured["max_lora_rank"] == 8


def test_initialize_vllm_overrides_conflicting_multiprocessing_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LLM:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = LLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

    initialize_vllm("target", "draft")

    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
