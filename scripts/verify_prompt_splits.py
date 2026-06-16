from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from lora_spec.prompts import verify_prompt_manifest
from lora_spec.utils import add_common_args, get_config_value, resolve_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify frozen prompt splits and SHA-256 hashes.")
    add_common_args(parser)
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/prompts/pilot_v1/manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose, "verify_prompt_splits")
    config_data = resolve_config(args.config, args.override)
    manifest_path = str(get_config_value(config_data, args, "manifest"))
    verification = verify_prompt_manifest(manifest_path)
    logger.info("Verified prompt manifest %s", verification.manifest_name)
    for split_name in sorted(verification.split_hashes):
        logger.info(
            "%s: records=%d sha256=%s domains=%s",
            split_name,
            verification.split_counts[split_name],
            verification.split_hashes[split_name],
            verification.domain_counts[split_name],
        )
    logger.info(
        "Maximum cross-split n-gram Jaccard: %.6f",
        verification.maximum_cross_split_ngram_jaccard,
    )
    logger.info("Verified %d total frozen prompts", verification.total_records)


if __name__ == "__main__":
    main()
