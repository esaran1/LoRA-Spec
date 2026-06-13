from __future__ import annotations

import importlib.metadata

import pytest
import torch

from lora_spec.metrics import patch_vllm_rejection_sampler


def test_pinned_vllm_sampler_instrumentation_contract() -> None:
    pytest.importorskip("vllm")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vLLM distribution metadata is unavailable")
    if version != "0.5.3.post1":
        pytest.skip("integration contract is pinned to vllm==0.5.3.post1")
    from vllm.model_executor.layers.rejection_sampler import RejectionSampler

    sampler = RejectionSampler(disable_bonus_tokens=False, strict_mode=True)
    draft_probs = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]],
        dtype=torch.float32,
    )
    draft_token_ids = torch.tensor([[0, 1]], dtype=torch.long)
    bonus_token_ids = torch.tensor([[2]], dtype=torch.long)
    with patch_vllm_rejection_sampler(speculation_length=2) as (accumulator, backend):
        output = sampler.forward(
            target_probs=draft_probs.clone(),
            bonus_token_ids=bonus_token_ids,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            generators=[None],
        )
    summary = accumulator.summarize()
    assert backend.endswith("RejectionSampler._get_accepted")
    assert output.tolist() == [[0, 1, 2]]
    assert summary.accepted_drafted_tokens == 2
    assert summary.total_drafted_tokens == 2
    assert summary.bonus_tokens == 1
