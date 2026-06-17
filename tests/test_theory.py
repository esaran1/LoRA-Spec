from __future__ import annotations

import pytest
import torch

import lora_spec.theory as theory_module
from lora_spec.theory import (
    ContinuationContextSet,
    FactoredParameterDelta,
    LogitShiftDataset,
    center_logit_shift_rows,
    collect_context_model_outputs,
    effective_rank,
    first_order_logit_shift,
    nonlinearity_residual,
    parameter_delta_from_models,
    singular_value_spectrum,
    spectral_analysis,
    spectral_sample_size_sensitivity,
    subspace_overlap,
    subspace_overlap_from_bases,
    truncated_right_singular_subspace,
    validate_continuation_contexts_for_prompts,
)


def test_center_logit_shift_rows_removes_softmax_gauge() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 2.0, 8.0]])
    centered = center_logit_shift_rows(matrix)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(torch.softmax(centered, dim=-1), torch.softmax(matrix, dim=-1))


def test_continuation_context_hash_covers_generation_metadata() -> None:
    base = ContinuationContextSet(
        input_ids=(torch.tensor([1, 2, 3], dtype=torch.long),),
        prompt_lengths=(2,),
        continuation_lengths=(1,),
        trajectory_model="base_target",
        generation_policy="greedy",
    )
    adapted = ContinuationContextSet(
        input_ids=base.input_ids,
        prompt_lengths=base.prompt_lengths,
        continuation_lengths=base.continuation_lengths,
        trajectory_model="adapted_target",
        generation_policy=base.generation_policy,
    )

    assert base.sha256() != adapted.sha256()
    assert base.to_dict()["input_ids"] == [[1, 2, 3]]


def test_continuation_context_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        ContinuationContextSet(
            input_ids=(torch.tensor([1, 2, 3], dtype=torch.long),),
            prompt_lengths=(2,),
            continuation_lengths=(2,),
            trajectory_model="test",
            generation_policy="fixed",
        )


def test_explicit_contexts_must_match_prompt_count() -> None:
    contexts = ContinuationContextSet(
        input_ids=(torch.tensor([1, 2, 3], dtype=torch.long),),
        prompt_lengths=(2,),
        continuation_lengths=(1,),
        trajectory_model="test",
        generation_policy="fixed",
    )

    with pytest.raises(ValueError, match="1 continuation trajectories for 2 prompts"):
        validate_continuation_contexts_for_prompts(
            contexts,
            ["prompt a", "prompt b"],
            context="unit_test",
        )


def test_logit_shift_dataset_rejects_misaligned_shapes() -> None:
    contexts = ContinuationContextSet(
        input_ids=(torch.tensor([1, 2, 3], dtype=torch.long),),
        prompt_lengths=(2,),
        continuation_lengths=(1,),
        trajectory_model="test",
        generation_policy="fixed",
    )

    with pytest.raises(ValueError, match="base_logits_matrix must match"):
        LogitShiftDataset(
            shift_matrix=torch.zeros(1, 3),
            base_logits_matrix=torch.zeros(1, 4),
            adapted_logits_matrix=torch.zeros(1, 3),
            hidden_state_matrix=None,
            prompt_indices=[0],
            token_positions=[1],
            vocabulary_size=3,
            continuation_contexts=contexts,
        )


def test_effective_rank_matches_synthetic_low_rank_matrix() -> None:
    left = torch.randn(8, 2)
    right = torch.randn(2, 6)
    matrix = left @ right
    assert effective_rank(matrix, threshold=0.99) == 2
    analysis = spectral_analysis(matrix)
    assert analysis.effective_rank_99 == 2
    assert analysis.stable_rank <= 2.01


def test_spectrum_preserves_small_ill_conditioned_component() -> None:
    generator = torch.Generator().manual_seed(17)
    left, _ = torch.linalg.qr(torch.randn(12, 3, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(64, 3, generator=generator, dtype=torch.float64))
    expected = torch.tensor([1.0, 1e-3, 1e-5], dtype=torch.float64)
    matrix = (left @ torch.diag(expected) @ right.T).float()
    observed = singular_value_spectrum(matrix)[:3]
    assert torch.allclose(observed, expected, rtol=2e-3, atol=1e-8)


def test_spectral_analysis_computes_the_spectrum_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = theory_module.singular_value_spectrum
    calls = 0

    def counted(matrix: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(theory_module, "singular_value_spectrum", counted)
    spectral_analysis(torch.randn(8, 5))

    assert calls == 1


def test_spectral_sample_size_sensitivity_uses_nested_prompt_clusters() -> None:
    matrix = torch.randn(12, 5)
    cluster_ids = [index // 3 for index in range(12)]
    sensitivity = spectral_sample_size_sensitivity(
        matrix,
        cluster_ids,
        sample_sizes=[2, 4],
        seed=4,
        repetitions=3,
    )

    assert [row["num_clusters"] for row in sensitivity] == [2, 4]
    assert [row["num_rows"] for row in sensitivity] == [6, 12]
    assert sensitivity[0]["repetitions"] == 3
    assert sensitivity[1]["repetitions"] == 1
    assert sensitivity[-1]["rank_ceiling"] == 5


def test_truncated_right_singular_subspace_reconstructs_known_rank() -> None:
    left = torch.randn(7, 2)
    right = torch.randn(2, 11)
    matrix = left @ right
    basis, singular_values = truncated_right_singular_subspace(matrix, rank=2)
    reconstructed = matrix @ basis @ basis.T
    assert basis.shape == (11, 2)
    assert singular_values.shape[0] == 7
    assert torch.linalg.norm(matrix - reconstructed) / torch.linalg.norm(matrix) < 1e-5


def test_truncated_subspace_preserves_weak_direction_at_full_vocabulary_scale() -> None:
    matrix = torch.zeros((2, 128_000), dtype=torch.float32)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 0.01

    basis, singular_values = truncated_right_singular_subspace(matrix, rank=2)

    assert basis.shape == (128_000, 2)
    assert singular_values[:2].tolist() == pytest.approx([1.0, 0.01], rel=1e-6)
    reconstructed = matrix @ basis @ basis.T
    assert torch.linalg.norm(matrix - reconstructed) / torch.linalg.norm(matrix) < 1e-6


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
    assert shared.overlap_fraction_a == pytest.approx(1.0, rel=1e-4)
    assert orthogonal.mean_cosine < 0.1


def test_subspace_overlap_accounts_for_unmatched_dimensions() -> None:
    basis_a = torch.eye(4)[:, :3]
    basis_b = torch.eye(4)[:, :2]
    overlap = subspace_overlap_from_bases(basis_a, basis_b)

    assert overlap.rank_a == 3
    assert overlap.rank_b == 2
    assert overlap.overlap_fraction_a == pytest.approx(2.0 / 3.0)
    assert overlap.overlap_fraction_b == pytest.approx(1.0)
    assert overlap.chordal_distance > 0.0


def test_nonlinearity_residual_zero_for_exact_first_order_match() -> None:
    matrix = torch.randn(6, 5)
    residual = nonlinearity_residual(matrix, matrix.clone())
    assert residual.frobenius_fraction == pytest.approx(0.0, abs=1e-8)
    assert residual.cosine_similarity_mean == pytest.approx(1.0, rel=1e-6)


def test_first_order_shift_accepts_factored_deltas_with_bounded_groups() -> None:
    class Output:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.lm_head = torch.nn.Linear(3, 4, bias=False)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> Output:
            _ = attention_mask
            return Output(self.lm_head(self.embedding(input_ids)))

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(
            self,
            prompts: list[str],
            return_tensors: str,
            padding: bool,
            truncation: bool,
        ) -> dict[str, torch.Tensor]:
            _ = return_tensors, padding, truncation
            rows = [[int(piece) for piece in prompt.split()] for prompt in prompts]
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.ones(len(rows), len(rows[0]), dtype=torch.long),
            }

    torch.manual_seed(3)
    model = TinyModel().eval()
    matrix_b = torch.randn(4, 1)
    matrix_a = torch.randn(1, 3)
    dense_delta = matrix_b @ matrix_a
    shift = first_order_logit_shift(
        base_model=model,
        delta_W={"lm_head.weight": FactoredParameterDelta(((matrix_b, matrix_a, 1.0),))},
        calibration_prompts=["0 1 2"],
        tokenizer=TinyTokenizer(),
        max_tangent_bytes=dense_delta.numel() * dense_delta.element_size(),
        continuation_contexts=ContinuationContextSet(
            input_ids=(torch.tensor([0, 1, 2]),),
            prompt_lengths=(2,),
            continuation_lengths=(1,),
            trajectory_model="test",
            generation_policy="fixed",
        ),
    )
    expected = model.embedding(torch.tensor([1])) @ dense_delta.T

    assert torch.allclose(shift, expected, atol=1e-5)


def test_first_order_shift_rejects_single_tangent_over_budget() -> None:
    class Output:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> Output:
            _ = attention_mask
            return Output(torch.nn.functional.one_hot(input_ids, 3).float() @ self.weight)

    with pytest.raises(ValueError, match="exceeding max_tangent_bytes"):
        first_order_logit_shift(
            TinyModel(),
            {"weight": torch.ones(3, 3)},
            ["unused"],
            tokenizer=object(),  # type: ignore[arg-type]
            max_tangent_bytes=1,
        )


def test_parameter_delta_rejects_dora_as_plain_lora() -> None:
    class Config:
        peft_type = "LORA"
        use_dora = True
        bias = "none"
        modules_to_save = None

    base = torch.nn.Linear(2, 2, bias=False)
    adapted = torch.nn.Linear(2, 2, bias=False)
    adapted.peft_config = {"default": Config()}  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="plain LoRA only"):
        parameter_delta_from_models(base, adapted)  # type: ignore[arg-type]


@pytest.mark.parametrize("peft_type", ["XLORA", "ADALORA", "PeftType.VBLORA"])
def test_parameter_delta_rejects_non_plain_lora_variants(peft_type: str) -> None:
    class Config:
        use_dora = False
        bias = "none"
        modules_to_save = None

    config = Config()
    config.peft_type = peft_type
    base = torch.nn.Linear(2, 2, bias=False)
    adapted = torch.nn.Linear(2, 2, bias=False)
    adapted.peft_config = {"default": config}  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match=f"peft_type={peft_type.split('.')[-1]}"):
        parameter_delta_from_models(base, adapted)  # type: ignore[arg-type]


def test_context_collection_starts_at_first_continuation_prediction() -> None:
    class Output:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits
            self.hidden_states = None

    class PositionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_hidden_states: bool = False,
        ) -> Output:
            _ = attention_mask, output_hidden_states
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).float()
            logits = positions.view(1, -1, 1).expand(input_ids.shape[0], -1, 3)
            return Output(logits + self.anchor)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    logits, _, prompt_indices, token_positions = collect_context_model_outputs(
        PositionModel(),  # type: ignore[arg-type]
        Tokenizer(),  # type: ignore[arg-type]
        ContinuationContextSet(
            input_ids=(torch.tensor([3, 2, 1, 0]),),
            prompt_lengths=(2,),
            continuation_lengths=(2,),
            trajectory_model="test",
            generation_policy="fixed",
        ),
    )

    assert logits[:, 0].tolist() == [1.0, 2.0]
    assert prompt_indices == [0, 0]
    assert token_positions == [1, 2]
