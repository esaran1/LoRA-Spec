from __future__ import annotations

import importlib.metadata
import inspect

import pytest


def test_pinned_vllm_sampler_instrumentation_contract() -> None:
    pytest.importorskip("vllm")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vLLM distribution metadata is unavailable")
    if version != "0.15.1":
        pytest.skip("integration contract is pinned to vllm==0.15.1")
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    parameters = inspect.signature(RejectionSampler.forward).parameters
    assert tuple(parameters)[:5] == (
        "self",
        "metadata",
        "draft_probs",
        "logits",
        "sampling_metadata",
    )
    parse_parameters = inspect.signature(RejectionSampler.parse_output).parameters
    assert tuple(parse_parameters)[:4] == (
        "output_token_ids",
        "vocab_size",
        "discard_req_indices",
        "logprobs_tensors",
    )
