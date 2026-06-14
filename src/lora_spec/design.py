from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentDesignReport:
    paper_ready: bool
    study_stage: str
    independent_adapter_sources: int
    independent_replicate_identities: int
    configured_experiments: int
    required_replicates_per_cell: int
    missing_ranks: list[int]
    missing_domains: list[str]
    missing_epochs: list[int]
    missing_model_pairs: list[str]
    missing_required_axes: list[str]
    missing_factorial_cells: list[dict[str, Any]]
    insufficient_replication_cells: list[dict[str, Any]]
    incompatible_experiments: list[dict[str, str]]
    rank_domain_confounding: dict[str, list[str]]
    rank_model_confounding: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_experiment_design(
    payload: dict[str, Any],
    models_payload: dict[str, Any] | None = None,
) -> ExperimentDesignReport:
    adapters = payload.get("adapters", {})
    experiments = payload.get("experiments", [])
    required = payload.get("required_coverage", {})
    if not isinstance(adapters, dict) or not isinstance(experiments, list):
        raise ValueError("Adapter design must contain adapters and experiments")

    required_ranks = sorted(int(value) for value in required.get("ranks", []))
    required_domains = sorted(str(value) for value in required.get("domains", []))
    required_epochs = sorted(int(value) for value in required.get("epochs", []))
    required_model_pairs = sorted(str(value) for value in required.get("model_pairs", []))
    default_replicates = 3 if str(payload.get("study_stage", "")).lower() == "paper" else 1
    required_replicates = int(required.get("replicates_per_cell", default_replicates))
    if required_replicates < 1:
        raise ValueError("required_coverage.replicates_per_cell must be positive")
    observed_ranks: set[int] = set()
    observed_domains: set[str] = set()
    observed_epochs: set[int] = set()
    observed_cells: set[tuple[str, int, str, int]] = set()
    observed_model_pairs: set[str] = set()
    rank_domains: dict[int, set[str]] = {}
    rank_models: dict[int, set[str]] = {}
    adapter_sources: set[str] = set()
    replicate_identities: set[str] = set()
    cell_replicates: dict[tuple[str, int, str, int], set[str]] = {}
    incompatible_experiments: list[dict[str, str]] = []
    model_pairs = (models_payload or {}).get("model_pairs", {})

    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment entry must be a mapping")
        adapter_name = str(experiment["adapter"])
        if adapter_name not in adapters:
            raise KeyError(f"Unknown adapter in experiment design: {adapter_name}")
        adapter = adapters[adapter_name]
        rank = int(adapter["rank"])
        domain = str(adapter["domain"])
        epoch_value = adapter.get("epochs")
        model_pair = str(experiment["model_pair"])
        replicate_identity = str(
            adapter.get("replicate_id")
            or "|".join(
                (
                    str(adapter["hf_path"]),
                    str(adapter.get("revision", "unspecified")),
                    str(adapter.get("training_seed", "unspecified")),
                )
            )
        )
        adapter_source = "|".join(
            (str(adapter["hf_path"]), str(adapter.get("revision", "unspecified")))
        )
        observed_ranks.add(rank)
        observed_domains.add(domain)
        observed_model_pairs.add(model_pair)
        adapter_sources.add(adapter_source)
        replicate_identities.add(replicate_identity)
        adapter_target = adapter.get("target_model")
        model_target = (
            model_pairs.get(model_pair, {}).get("target_model")
            if isinstance(model_pairs, dict)
            else None
        )
        if model_pairs and model_pair not in model_pairs:
            incompatible_experiments.append(
                {
                    "model_pair": model_pair,
                    "adapter": adapter_name,
                    "adapter_target": str(adapter_target or "unknown"),
                    "model_target": "missing_model_pair",
                }
            )
        if adapter_target and model_target and str(adapter_target) != str(model_target):
            incompatible_experiments.append(
                {
                    "model_pair": model_pair,
                    "adapter": adapter_name,
                    "adapter_target": str(adapter_target),
                    "model_target": str(model_target),
                }
            )
        rank_domains.setdefault(rank, set()).add(domain)
        rank_models.setdefault(rank, set()).add(model_pair)
        if epoch_value is not None:
            epoch = int(epoch_value)
            observed_epochs.add(epoch)
            observed_cells.add((model_pair, rank, domain, epoch))
            cell_replicates.setdefault((model_pair, rank, domain, epoch), set()).add(
                replicate_identity
            )

    required_cells = set(
        itertools.product(
            required_model_pairs,
            required_ranks,
            required_domains,
            required_epochs,
        )
    )
    missing_cells = [
        {
            "model_pair": model_pair,
            "rank": rank,
            "domain": domain,
            "epochs": epochs,
        }
        for model_pair, rank, domain, epochs in sorted(required_cells - observed_cells)
    ]
    insufficient_replication_cells = [
        {
            "model_pair": model_pair,
            "rank": rank,
            "domain": domain,
            "epochs": epochs,
            "independent_replicates": len(
                cell_replicates.get((model_pair, rank, domain, epochs), set())
            ),
            "required_replicates": required_replicates,
        }
        for model_pair, rank, domain, epochs in sorted(required_cells & observed_cells)
        if len(cell_replicates.get((model_pair, rank, domain, epochs), set())) < required_replicates
    ]
    rank_domain_confounding = {
        str(rank): sorted(domains)
        for rank, domains in sorted(rank_domains.items())
        if required_domains and domains != set(required_domains)
    }
    rank_model_confounding = {
        str(rank): sorted(models)
        for rank, models in sorted(rank_models.items())
        if required_model_pairs and models != set(required_model_pairs)
    }
    missing_ranks = sorted(set(required_ranks) - observed_ranks)
    missing_domains = sorted(set(required_domains) - observed_domains)
    missing_epochs = sorted(set(required_epochs) - observed_epochs)
    missing_model_pairs = sorted(set(required_model_pairs) - observed_model_pairs)
    required_axes = {
        "ranks": required_ranks,
        "domains": required_domains,
        "epochs": required_epochs,
        "model_pairs": required_model_pairs,
    }
    missing_required_axes = [name for name, values in required_axes.items() if not values]
    paper_ready = not any(
        (
            missing_ranks,
            missing_domains,
            missing_epochs,
            missing_model_pairs,
            missing_required_axes,
            missing_cells,
            insufficient_replication_cells,
            incompatible_experiments,
            rank_domain_confounding,
            rank_model_confounding,
        )
    )
    return ExperimentDesignReport(
        paper_ready=paper_ready,
        study_stage=str(payload.get("study_stage", "unspecified")),
        independent_adapter_sources=len(adapter_sources),
        independent_replicate_identities=len(replicate_identities),
        configured_experiments=len(experiments),
        required_replicates_per_cell=required_replicates,
        missing_ranks=missing_ranks,
        missing_domains=missing_domains,
        missing_epochs=missing_epochs,
        missing_model_pairs=missing_model_pairs,
        missing_required_axes=missing_required_axes,
        missing_factorial_cells=missing_cells,
        insufficient_replication_cells=insufficient_replication_cells,
        incompatible_experiments=incompatible_experiments,
        rank_domain_confounding=rank_domain_confounding,
        rank_model_confounding=rank_model_confounding,
    )
