# LoRA-Spec Config Authority

This directory is the source of truth for all LoRA-Spec experiments. Scripts and notebooks may pass `--config` and `--override` flags for execution mechanics, but experimental choices must be traceable back to these YAML files before results are used in the paper.

## Authoritative Files

| File | Owns | Notes |
|---|---|---|
| `models.yaml` | Target/draft model pairs, Hugging Face paths, revisions, tensor parallel degree | Add new architectures here before running model-family claims. |
| `adapters.yaml` | Adapter IDs, ranks, domains, epochs, magnitude scales, sweep cells | Add new adapters here before collecting rank, correction, or serving results. |
| `prompts.yaml` | Frozen calibration/evaluation prompt split paths and SHA-256 hashes | Prompt files must be verified with `scripts/verify_prompt_splits.py` before use. |
| `baselines.yaml` | Baseline methods, implementation status, enabled state | Disabled or unimplemented baselines cannot appear as completed comparisons. |
| `serving.yaml` | Multi-tenant traffic patterns, concurrency, burst settings | Serving benchmarks must cite the named traffic pattern used. |

## Rules

1. Do not hard-code model paths, adapter paths, prompt files, baseline names, traffic patterns, seeds, or sweep cells in notebooks.
2. If a CLI override changes a scientific variable, record that override in the run ledger before launching the run.
3. Every result JSON must include the config hash, git hash, timestamp, seed, GPU type, and full resolved config.
4. Calibration prompts are for fitting corrections or estimating geometry. Evaluation prompts are for reporting metrics.
5. Pilot prompt split `pilot_v1` is frozen for early validation only; larger final-study splits should be added as new named entries, not by modifying `pilot_v1`.
6. Community adapter metadata should include the Hugging Face path and revision once a run is promoted from pilot to paper-grade.

## Verification Commands

Run these on a MacBook before launching GPU jobs:

```bash
python scripts/verify_prompt_splits.py --verbose
python scripts/validate_experiment_design.py --config configs/adapters.yaml --verbose
```

Run this before using a result in a figure or table:

```bash
python scripts/plot_results.py --help
```

