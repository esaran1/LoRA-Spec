from __future__ import annotations

import pytest
import torch

from lora_spec.acceptance_theory import (
    acceptance_lower_bound_from_logit_residual,
    empirical_vs_predicted_recovery,
    predicted_acceptance_recovery,
    rejection_sampling_acceptance,
    total_variation_distance,
)


def test_rejection_acceptance_equals_one_minus_total_variation() -> None:
    target = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    draft = torch.tensor([[0.5, 0.2, 0.3], [0.2, 0.2, 0.6]])
    acceptance = rejection_sampling_acceptance(target, draft)
    total_variation = total_variation_distance(target, draft)
    assert torch.allclose(acceptance, 1.0 - total_variation, atol=1e-7)


def test_logit_span_bound_is_shift_invariant_and_valid() -> None:
    target_logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, -0.1, 0.7]])
    approximate_logits = torch.tensor([[1.6, 0.7, -0.8], [0.0, 0.0, 0.5]])
    target = torch.softmax(target_logits, dim=-1)
    approximate = torch.softmax(approximate_logits, dim=-1)
    empirical = rejection_sampling_acceptance(target, approximate)
    lower_bound = acceptance_lower_bound_from_logit_residual(target_logits, approximate_logits)
    shifted_bound = acceptance_lower_bound_from_logit_residual(
        target_logits + 13.0,
        approximate_logits - 4.0,
    )
    assert torch.all(lower_bound <= empirical + 1e-6)
    assert torch.allclose(lower_bound, shifted_bound, atol=1e-7)


def test_total_variation_prediction_is_exact_acceptance_identity() -> None:
    prediction = predicted_acceptance_recovery(
        approximation_error=[0.0, 0.25, 0.5, 1.0],
        base_acceptance=0.4,
        error_metric="total_variation",
    )
    assert prediction.acceptance_lower_bound == pytest.approx([1.0, 0.75, 0.5, 0.0])
    assert prediction.guaranteed_recovery_delta == pytest.approx([0.6, 0.35, 0.1, -0.4])


def test_empirical_comparison_reports_bound_coverage() -> None:
    comparison = empirical_vs_predicted_recovery(
        approximation_error=[0.1, 0.2, 0.3],
        base_acceptance=0.4,
        empirical_acceptance=[0.92, 0.82, 0.75],
        error_metric="total_variation",
    )
    assert comparison.bound_coverage == pytest.approx(1.0)
    assert comparison.maximum_bound_violation == pytest.approx(0.0)
    assert comparison.mean_bound_slack > 0.0
