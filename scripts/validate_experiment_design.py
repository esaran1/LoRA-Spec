from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from lora_spec.design import audit_experiment_design
from lora_spec.prompts import file_sha256
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    load_yaml,
    resolve_config,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit adapter sweep coverage and confounding.")
    add_common_args(parser)
    parser.add_argument("--adapters-config", type=str, default="configs/adapters.yaml")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/design")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "validate_experiment_design")
    config_data = resolve_config(args.config, args.override)
    adapters_config = str(get_config_value(config_data, args, "adapters_config"))
    models_config = str(get_config_value(config_data, args, "models_config"))
    strict = bool(get_config_value(config_data, args, "strict", args.strict))
    report = audit_experiment_design(load_yaml(adapters_config), load_yaml(models_config))
    output = write_json_result(
        payload={"design_report": report.to_dict()},
        output_dir=str(get_config_value(config_data, args, "output_dir")),
        stem="experiment_design",
        config={
            "adapters_config": adapters_config,
            "models_config": models_config,
            "adapters_config_sha256": file_sha256(adapters_config),
            "models_config_sha256": file_sha256(models_config),
            "strict": strict,
            "seed": int(get_config_value(config_data, args, "seed")),
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved experiment-design audit to %s", output)
    if not report.paper_ready:
        message = (
            "Adapter design is pilot-only: missing ranks=%s domains=%s epochs=%s "
            "model_pairs=%s required_axes=%s; rank/domain confounding=%s; "
            "under-replicated cells=%s incompatible experiments=%s"
        )
        values = (
            report.missing_ranks,
            report.missing_domains,
            report.missing_epochs,
            report.missing_model_pairs,
            report.missing_required_axes,
            report.rank_domain_confounding,
            report.insufficient_replication_cells,
            report.incompatible_experiments,
        )
        if strict:
            raise ValueError(message % values)
        logger.warning(message, *values)


if __name__ == "__main__":
    main()
