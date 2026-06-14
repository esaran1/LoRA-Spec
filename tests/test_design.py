from __future__ import annotations

from lora_spec.design import audit_experiment_design


def test_design_audit_detects_rank_domain_confounding_and_missing_cells() -> None:
    payload = {
        "study_stage": "pilot",
        "required_coverage": {
            "ranks": [4, 8],
            "domains": ["code", "math"],
            "epochs": [1],
            "model_pairs": ["family_a", "family_b"],
        },
        "adapters": {
            "code_r4": {
                "rank": 4,
                "domain": "code",
                "epochs": 1,
                "hf_path": "org/code-r4",
            }
        },
        "experiments": [{"model_pair": "family_a", "adapter": "code_r4"}],
    }
    report = audit_experiment_design(payload)
    assert not report.paper_ready
    assert report.missing_ranks == [8]
    assert report.missing_domains == ["math"]
    assert report.rank_domain_confounding == {"4": ["code"]}
    assert report.missing_model_pairs == ["family_b"]
    assert len(report.missing_factorial_cells) == 7


def test_design_audit_accepts_complete_crossed_design() -> None:
    adapters = {}
    experiments = []
    for rank in (4, 8):
        for domain in ("code", "math"):
            for model in ("family_a", "family_b"):
                name = f"{domain}_r{rank}_{model}"
                adapters[name] = {
                    "rank": rank,
                    "domain": domain,
                    "epochs": 1,
                    "hf_path": f"org/{name}",
                }
                experiments.append({"model_pair": model, "adapter": name})
    report = audit_experiment_design(
        {
            "study_stage": "paper",
            "required_coverage": {
                "ranks": [4, 8],
                "domains": ["code", "math"],
                "epochs": [1],
                "model_pairs": ["family_a", "family_b"],
                "replicates_per_cell": 1,
            },
            "adapters": adapters,
            "experiments": experiments,
        }
    )
    assert report.paper_ready


def test_design_audit_requires_independent_replication_for_paper_design() -> None:
    report = audit_experiment_design(
        {
            "study_stage": "paper",
            "required_coverage": {
                "ranks": [4],
                "domains": ["code"],
                "epochs": [1],
                "model_pairs": ["family_a"],
                "replicates_per_cell": 2,
            },
            "adapters": {
                "a": {
                    "rank": 4,
                    "domain": "code",
                    "epochs": 1,
                    "hf_path": "org/shared",
                },
                "a_scaled": {
                    "rank": 4,
                    "domain": "code",
                    "epochs": 1,
                    "hf_path": "org/shared",
                    "magnitude_scale": 2.0,
                },
            },
            "experiments": [
                {"model_pair": "family_a", "adapter": "a"},
                {"model_pair": "family_a", "adapter": "a_scaled"},
            ],
        }
    )

    assert not report.paper_ready
    assert report.insufficient_replication_cells[0]["independent_replicates"] == 1


def test_design_audit_accepts_explicit_independent_replicates() -> None:
    report = audit_experiment_design(
        {
            "study_stage": "paper",
            "required_coverage": {
                "ranks": [4],
                "domains": ["code"],
                "epochs": [1],
                "model_pairs": ["family_a"],
                "replicates_per_cell": 2,
            },
            "adapters": {
                "seed_1": {
                    "rank": 4,
                    "domain": "code",
                    "epochs": 1,
                    "hf_path": "org/shared",
                    "replicate_id": "code-r4-seed-1",
                },
                "seed_2": {
                    "rank": 4,
                    "domain": "code",
                    "epochs": 1,
                    "hf_path": "org/shared",
                    "replicate_id": "code-r4-seed-2",
                },
            },
            "experiments": [
                {"model_pair": "family_a", "adapter": "seed_1"},
                {"model_pair": "family_a", "adapter": "seed_2"},
            ],
        }
    )

    assert report.paper_ready
    assert not report.insufficient_replication_cells
    assert report.independent_adapter_sources == 1
    assert report.independent_replicate_identities == 2


def test_design_audit_rejects_marginal_but_not_fully_crossed_model_coverage() -> None:
    adapters = {
        "code_r4": {"rank": 4, "domain": "code", "epochs": 1, "hf_path": "org/a"},
        "math_r4": {"rank": 4, "domain": "math", "epochs": 1, "hf_path": "org/b"},
        "code_r8": {"rank": 8, "domain": "code", "epochs": 1, "hf_path": "org/c"},
        "math_r8": {"rank": 8, "domain": "math", "epochs": 1, "hf_path": "org/d"},
    }
    experiments = [
        {"model_pair": "family_a", "adapter": "code_r4"},
        {"model_pair": "family_b", "adapter": "math_r4"},
        {"model_pair": "family_b", "adapter": "code_r8"},
        {"model_pair": "family_a", "adapter": "math_r8"},
    ]
    report = audit_experiment_design(
        {
            "study_stage": "paper",
            "required_coverage": {
                "ranks": [4, 8],
                "domains": ["code", "math"],
                "epochs": [1],
                "model_pairs": ["family_a", "family_b"],
            },
            "adapters": adapters,
            "experiments": experiments,
        }
    )

    assert not report.paper_ready
    assert len(report.missing_factorial_cells) == 4
