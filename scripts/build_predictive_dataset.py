from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lora_spec.predictive_data import FEATURE_NAMES, build_predictive_rows, load_property_index
from lora_spec.utils import (
    add_common_args,
    get_config_value,
    resolve_config,
    setup_logging,
    write_json_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join characterization and adapter-property artifacts."
    )
    add_common_args(parser)
    parser.add_argument("--characterize-json", type=str, default=None)
    parser.add_argument("--adapter-props-dir", type=str, default="results/adapter_props")
    parser.add_argument("--output-dir", type=str, default="results/predictive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "build_predictive_dataset")
    config_data = resolve_config(args.config, args.override)
    characterize_json = get_config_value(config_data, args, "characterize_json")
    if not characterize_json:
        raise ValueError("characterize_json must be provided")
    characterize_path = Path(str(characterize_json))
    characterize_bytes = characterize_path.read_bytes()
    characterize_sha256 = hashlib.sha256(characterize_bytes).hexdigest()
    characterize_payload = json.loads(characterize_bytes.decode("utf-8"))
    property_directory = str(get_config_value(config_data, args, "adapter_props_dir"))
    property_index = load_property_index(property_directory)
    rows = build_predictive_rows(characterize_payload, property_index)
    output = write_json_result(
        payload={
            "feature_names": FEATURE_NAMES,
            "target_name": "acceptance_degradation",
            "rows": rows,
        },
        output_dir=str(get_config_value(config_data, args, "output_dir")),
        stem="predictive_dataset",
        config={
            "characterize_json": str(characterize_path),
            "characterize_sha256": characterize_sha256,
            "characterize_config_hash": characterize_payload.get("config_hash"),
            "adapter_props_dir": property_directory,
            "adapter_property_artifacts": sorted(
                {
                    (str(row["adapter_properties_path"]), str(row["adapter_properties_sha256"]))
                    for row in rows
                }
            ),
            "matched_rows": len(rows),
            "seed": int(get_config_value(config_data, args, "seed")),
        },
        cwd=Path.cwd(),
    )
    logger.info("Saved %d-row predictive dataset to %s", len(rows), output)


if __name__ == "__main__":
    main()
