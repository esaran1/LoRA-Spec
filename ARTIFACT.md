# Artifact Guide

This document describes how to reproduce the main LoRA-Spec figures and tables from raw runs.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,analysis]
```

For GPU experiments:

```bash
pip install -e .[analysis,colab]
```

## Determinism

Every script accepts `--seed`, writes a JSON artifact with:

- full config
- config hash
- git hash
- timestamp
- runtime metadata including GPU type when CUDA is available
- requested and resolved Hugging Face revisions, or a content hash for local artifacts
- an exact binary Git patch plus base64-encoded untracked source files when the worktree is dirty

Dirty-source capture rejects likely secret files, untracked files larger than 5 MiB, and snapshots larger than 25 MiB. Commit source before paper runs; these bounds prevent accidental credential or checkpoint embedding in result artifacts.

Use a fixed prompt file for exact reruns of the theory experiments.

Remote model and adapter names are resolved to immutable Hub commit SHAs before model construction. The resolved SHA, prompt hashes, selected prompt IDs, and measurement settings participate in the experiment config hash. Reproduction should use the resolved revisions recorded in the artifact, not the current repository head.

Verify the checked-in frozen pilot split and its hashes before running:

```bash
python scripts/verify_prompt_splits.py --verbose
```

The authoritative manifest is `data/prompts/pilot_v1/manifest.json`, independently pinned by `data/prompts/pilot_v1/release.lock.json`. Every theory result config records the split role, split SHA-256, manifest SHA-256, release-lock SHA-256, record count, and domain counts. Calibration and evaluation roles are enforced at load time. The split is project-authored and suitable for pilot hypothesis validation; final paper claims require a larger benchmark-derived frozen suite.

## Interpretation Rules

- `expected_rejection_sampling_acceptance` is the full-vocabulary overlap `sum_v min(p_v, q_v) = 1 - TV(p, q)` on held-out contexts.
- `logit_acceptance_lower_bound` is the certified bound `1 - tanh(span / 4)` derived from the residual-logit oscillation, not a fitted predictor.
- `greedy_proxy_acceptance_rate` is a sequence-level diagnostic and must not be presented as standard rejection-sampling acceptance.
- Results with `spectrum_is_approximate` or `subspace_is_approximate` use a shared random vocabulary projection. Report the projection dimension and seed.
- Projected effective-rank results use multiple independent Gaussian sketches. Report every sketch estimate and its range; do not treat the representative plotted spectrum as exact uncertainty-free evidence.
- Repeated sketches do not measure projection-dimension bias. Paper runs must also report `projection_dimension_sensitivity` from at least three projection dimensions.
- Rank and correction analyses use row-mean-centered logits to remove the softmax-invariant scalar gauge.
- Rank, divergence, first-order, subspace, and correction analyses share deterministic base-target continuation contexts. The first measured row is the final prompt token predicting the first continuation token.
- Every continuation context set is hashed from token IDs and prompt/continuation boundaries; compare this hash before combining or reproducing runs.
- Held-out token metrics report equal-prompt-weighted cluster bootstrap intervals; token positions from one prompt are not treated as independent samples.
- Effective-rank artifacts report the row/dimension rank ceiling and nested prompt-sample-size sensitivity. Ceiling-saturated ranks are inconclusive.
- Magnitude breakpoints are exploratory continuous segmented regressions with a linear-model BIC comparison, not standalone proof of a phase transition.
- Correction calibration labels are always computed from `adapted target - base target`; draft logits or hidden states are only application-time features.
- Per-position vLLM acceptance is conditional on reaching that drafted position. `acceptance_by_depth[d]` is the fraction of eligible speculative steps whose first `d + 1` proposals all survive; artifacts also record the exact numerator and denominator for every depth.
- Phase 1 comparisons use randomized paired condition order. Report the replicate-level measurements and paired confidence intervals; the pooled counts are descriptive summaries, not independent replicates.
- Predictive results use leave-one-adapter-source-out folds, with all magnitude variants from one checkpoint kept together. Model-family-held-out evaluation is reported separately.

## Runtime Scope

The serving environment pins `vllm==0.15.1`, the first tagged runtime used by this artifact that combines ordinary draft-model speculation with the LoRA-capable V1 engine. The harness forces `VLLM_ENABLE_V1_MULTIPROCESSING=0` before engine construction because acceptance instrumentation is process-local. It derives accepted-prefix lengths after vLLM's own output parsing has removed placeholders, out-of-vocabulary entries, and discarded requests, and fails closed if sampler outputs are not parsed in-process. Analytical correction scripts report exact distribution overlap plus an explicitly labeled greedy sequence proxy; they do not claim corrected vLLM serving until a proposer-side correction integration is implemented and measured.

Run the pinned vLLM server only on localhost or a trusted research network. It is pinned for a validated measurement contract, not as a recommendation for a public-facing service; any runtime upgrade requires rerunning the sampler contract tests and GPU integration checks.

Micro-LoRA training partitions the frozen calibration split deterministically into disjoint training and validation subsets and selects the best checkpoint by validation full-vocabulary KL, including the zero-initialized draft LoRA as the epoch-zero baseline. The frozen evaluation split remains untouched until final comparison.

The current tensor-parallel SLURM template measures logit-shift rank and does not measure speculative acceptance. Do not infer TP=4 acceptance behavior from it. External server benchmarks require a server provenance manifest that exactly matches the client-resolved target, draft, adapter revisions, vLLM version, and registered adapter names.

For HTTP serving benchmarks, preload every adapter when starting vLLM and use the same names in the benchmark:

`server_provenance.json` must contain `vllm_version`, `adapter_model_names`, and an
`artifact_provenance` object copied from the client-resolved target, draft, and adapter records.

```bash
vllm serve <base-model> \
  --enable-lora \
  --max-lora-rank 64 \
  --speculative-config '{"model":"<draft-model>","method":"draft_model","num_speculative_tokens":4}' \
  --lora-modules code-lora=<immutable-local-adapter-path>

python scripts/benchmark_serving.py \
  --server-url http://localhost:8000 \
  --target-model <base-model> \
  --draft-model <draft-model> \
  --adapter-path <adapter-repository-or-path> \
  --adapter-model-name code-lora \
  --server-provenance-json server_provenance.json \
  --prompts-file data/prompts/pilot_v1/evaluation.jsonl
```

The benchmark warms baseline and adapted conditions, then fails closed if any measured request errors. Do not disable this behavior for paper measurements.
Latency percentiles use the Hyndman-Fan type-7 linear-interpolation estimator. In-process runs are serialized and labeled `in_process_serial`; only runs labeled `http_concurrent` are valid concurrent-serving measurements.

## Figure Map

### Figure 1: Effective Rank vs Adapter Rank and Magnitude

1. Prepare a calibration prompt file with one prompt per line.
2. Run:

```bash
python scripts/measure_logit_shift_rank.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --projection-dimensions 128,256,512 \
  --verbose
```

3. Plot:

```bash
python scripts/plot_results.py --input-dir results --output-dir results/plots --verbose
```

Output figure: `results/plots/theory_effective_rank.png`

Before a full sweep, validate the design:

```bash
python scripts/validate_experiment_design.py \
  --adapters-config configs/adapters.yaml \
  --strict \
  --verbose
```

The checked-in manifest is a pilot and is expected to fail strict mode until the crossed rank/domain/epoch/model adapter set has been produced with the configured number of independent training replicates per cell. Give independently trained checkpoints explicit `replicate_id` values; magnitude-scaled views of one checkpoint share a replicate identity.

### Figure 2: Approximation Error vs Acceptance Recovery

```bash
python scripts/validate_correction_theory.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --adapted-adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --eval-prompts-file data/prompts/pilot_v1/evaluation.jsonl \
  --rank-values 0,1,2,4,8,16 \
  --verbose
```

Output figure: `results/plots/theory_correction_validation.png`

### Figure 3: Adapter-Magnitude Phase Transition

```bash
python scripts/phase_transition_sweep.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --eval-prompts-file data/prompts/pilot_v1/evaluation.jsonl \
  --magnitude-values 0.0,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
  --verbose
```

Output figure: `results/plots/theory_phase_transition.png`

### Figure 4: Shared Correction Subspace

```bash
python scripts/subspace_sharing.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --model-pair llama3_8b_1b \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --verbose
```

Output figure: `results/plots/theory_subspace_sharing.png`

The checked-in pilot manifest currently contains one unique adapter source per compatible target and several magnitude-scaled replicas. Those replicas are a magnitude-invariance control, not evidence of cross-adapter universality. A paper claim about a shared universal subspace requires at least two independently trained adapters with the same base target; verify `unique_adapter_sources >= 2` in the result artifact.

## Existing System Figures

- `scripts/validate_hypothesis.py`: initial speculative-decoding degradation check
- `scripts/characterize.py`: broader empirical degradation characterization
- `scripts/benchmark_serving.py`: multi-tenant serving effects
- `scripts/train_micro_lora.py`: training-based upper bound when analytical correction fails

## SLURM

Use the provided templates in `slurm/` for GPU runs. Each template records:

- git hash
- conda environment
- GPU model
- CUDA version

`slurm/sweep_70b.sh` performs projected logit-rank analysis with `device_map=auto`; it is not a tensor-parallel vLLM acceptance benchmark.

## Expected Outputs

All result artifacts are JSON under `results/` and all figures are written under `results/plots/`.
