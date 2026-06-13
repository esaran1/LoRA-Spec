from __future__ import annotations

import pytest
import torch

from lora_spec.theory import (
    center_logit_shift_rows,
    effective_rank,
    nonlinearity_residual,
    spectral_analysis,
    subspace_overlap,
    truncated_right_singular_subspace,
)


def test_center_logit_shift_rows_removes_softmax_gauge() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 2.0, 8.0]])
    centered = center_logit_shift_rows(matrix)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(torch.softmax(centered, dim=-1), torch.softmax(matrix, dim=-1))


def test_effective_rank_matches_synthetic_low_rank_matrix() -> None:
    left = torch.randn(8, 2)
    right = torch.randn(2, 6)
    matrix = left @ right
    assert effective_rank(matrix, threshold=0.99) == 2
    analysis = spectral_analysis(matrix)
    assert analysis.effective_rank_99 == 2
    assert analysis.stable_rank <= 2.01


def test_truncated_right_singular_subspace_reconstructs_known_rank() -> None:
    left = torch.randn(7, 2)
    right = torch.randn(2, 11)
    matrix = left @ right
    basis, singular_values = truncated_right_singular_subspace(matrix, rank=2)
    reconstructed = matrix @ basis @ basis.T
    assert basis.shape == (11, 2)
    assert singular_values.shape[0] == 7
    assert torch.linalg.norm(matrix - reconstructed) / torch.linalg.norm(matrix) < 1e-5


def test_subspace_overlap_detects_shared_and_orthogonal_subspaces() -> None:
    basis_a = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    )
    basis_b = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    )
    basis_c = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )
    matrix_a = torch.randn(10, 2) @ basis_a.T
    matrix_b = torch.randn(10, 2) @ basis_b.T
    matrix_c = torch.randn(10, 2) @ basis_c.T

    shared = subspace_overlap(matrix_a, matrix_b, rank=2)
    orthogonal = subspace_overlap(matrix_a, matrix_c, rank=2)
    assert shared.mean_cosine == pytest.approx(1.0, rel=1e-4)
    assert orthogonal.mean_cosine < 0.1


def test_nonlinearity_residual_zero_for_exact_first_order_match() -> None:
    matrix = torch.randn(6, 5)
    residual = nonlinearity_residual(matrix, matrix.clone())
    assert residual.frobenius_fraction == pytest.approx(0.0, abs=1e-8)
    assert residual.cosine_similarity_mean == pytest.approx(1.0, rel=1e-6)
