# LoRA-Spec

LoRA-Spec studies speculative decoding under LoRA adaptation from a theory-first perspective. The central object is the adapter-induced next-token logit shift

`delta_z(x) = z_adapted(x) - z_base(x)`.

Because LoRA is a low-rank weight perturbation, `W' = W + BA`, the project tests whether `delta_z(x)` is approximately low-rank in practice. LoRA rank alone does not imply a hard rank bound on the cross-context logit-shift matrix because the model Jacobian varies with context.

## Research Spine

The repository supports three core claims:

1. Determine when the empirical logit-shift matrix is low-rank or near-low-rank.
2. Evaluate a PCA-output-subspace ridge operator whose calibration residual decomposes into spectral-tail and coefficient-regression terms.
3. Test whether a reproducible regime boundary exists where the shift becomes too nonlinear or too large for analytical correction, and training becomes necessary.

System benchmarks remain in the repo, but they now validate the theoretical story rather than define it.

All logit-space rank results use a row-mean-centered gauge because softmax distributions are invariant to a per-context scalar offset. Correction operators are calibrated on `adapted target - base target`; draft outputs are application-time features, not part of the adapter-shift definition.
The context distribution is also explicit: the base target greedily generates a frozen continuation trajectory, and measurements begin at the final prompt token that predicts the first continuation token. Prompt-interior teacher-forced logits are not used as a proxy for speculative proposal contexts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,analysis]
pytest -q
```

For Colab Pro or GPU experiments with vLLM:

```bash
pip install -e .[analysis,colab]
```

## Main Theory Experiments

Verify the frozen pilot prompt splits before running experiments:

```bash
python scripts/verify_prompt_splits.py --verbose
```

The default `pilot_v1` split contains 16 calibration and 32 held-out evaluation prompts, balanced across code, medical, chat, and math. Its byte-level SHA-256 hashes are pinned in `data/prompts/pilot_v1/manifest.json` and independently locked by `release.lock.json`. Scripts enforce calibration/evaluation roles and reject unregistered files. This pilot is intended to validate the research direction before constructing the final benchmark suite.

Model and adapter identifiers are resolved to immutable Hugging Face commit SHAs at run time and included in the result config hash. Local artifacts are content-hashed. Full-vocabulary comparisons additionally require exact tokenizer vocabulary and special-token equivalence.

Run the paired Phase 1 validation on the frozen evaluation split:

```bash
python scripts/validate_hypothesis.py \
  --target-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS \
  --adapter-domain code \
  --prompts-file data/prompts/pilot_v1/evaluation.jsonl \
  --measurement-repetitions 3 \
  --verbose
```

The baseline/adapted order is randomized within each repetition. Results retain replicate-level acceptance, throughput, and TTFT measurements plus paired 95% confidence intervals.

Measure effective rank of the logit-shift matrix:

```bash
python scripts/measure_logit_shift_rank.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --model-pair llama3_8b_1b \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --projection-repetitions 3 \
  --verbose
```

When the exact full-vocabulary matrix exceeds the configured memory limit, the script uses independent Gaussian sketches and reports all sketch-level rank estimates and ranges. Those outputs are explicitly marked approximate.
Use `--projection-dimensions 128,256,512` for the required projection-dimension sensitivity check. Repeated sketches estimate sketch randomness; varying the projection dimension tests compression bias.
The artifact also reports the matrix rank ceiling and nested prompt-sample-size sensitivity. An effective rank near the row ceiling is inconclusive evidence of low rank and requires a larger calibration set.

Audit the configured factorial coverage before interpreting a sweep as paper evidence:

```bash
python scripts/validate_experiment_design.py \
  --adapters-config configs/adapters.yaml \
  --strict \
  --verbose
```

The checked-in adapter manifest is intentionally labeled `pilot` and does not pass strict mode. It contains magnitude controls for early theory validation, not the fully crossed rank/domain/epoch/model design required for causal comparisons.

Validate the exact rejection-sampling overlap identity and the residual-logit lower bound against held-out acceptance:

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

Sweep adapter magnitude for phase-transition analysis:

```bash
python scripts/phase_transition_sweep.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --eval-prompts-file data/prompts/pilot_v1/evaluation.jsonl \
  --verbose
```

The sweep reports prompt-cluster bootstrap intervals and an exploratory continuous segmented-regression breakpoint. A positive BIC improvement supports curvature over a single linear trend; it is not by itself proof of a sharp phase transition.

Measure shared dominant subspaces across adapters:

```bash
python scripts/subspace_sharing.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --model-pair llama3_8b_1b \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --verbose
```

## Existing System-Facing Experiments

- `scripts/validate_hypothesis.py`: initial degradation check under speculative decoding
- `scripts/characterize.py`: broader degradation characterization
- `scripts/analytical_correction.py`: single-run correction evaluation with theory-grounded operators
- `scripts/train_micro_lora.py`: training-based upper bound when analytical correction breaks down
- `scripts/benchmark_serving.py`: multi-tenant serving experiments

Distillation uses the frozen calibration split, and serving uses the frozen evaluation split. Both resolve model and adapter references to immutable revisions. The current vLLM rejection-sampler hook supports only single-process acceptance measurement; tensor-parallel 70B jobs are limited to theory/model-analysis experiments until worker-side metric reduction is implemented.

HTTP serving requires adapters preloaded with vLLM `--lora-modules`, unique `--adapter-model-name` values, and a matching `--server-provenance-json`. Both conditions are warmed, and request failures invalidate the run.

Predictive input rows must include `features`, `target`, `adapter_source`, and `model_family`. Publication-facing metrics use grouped source-held-out and model-family-held-out evaluation; row-wise LOOCV is retained only as a diagnostic.

Build those rows directly from immutable characterization and adapter-property artifacts:

```bash
python scripts/build_predictive_dataset.py \
  --characterize-json results/characterize/<aggregate>.json \
  --adapter-props-dir results/adapter_props \
  --verbose
```

## Plotting

Generate figures from JSON artifacts across theory and systems phases:

```bash
python scripts/plot_results.py \
  --input-dir results \
  --output-dir results/plots \
  --verbose
```

## Configs

- `configs/models.yaml`: model-pair definitions, including a third family beyond Llama and Qwen
- `configs/adapters.yaml`: adapter manifests with explicit magnitude sweeps
- `configs/baselines.yaml`: decoding, analytical, and training-based baselines
- `configs/serving.yaml`: traffic-pattern settings for serving benchmarks

## Artifact Reproduction

See [ARTIFACT.md](ARTIFACT.md) for figure-by-figure reproduction commands.

The baseline manifest explicitly marks registered external methods that are not yet integrated as disabled. Do not report autoregressive, DistillSpec-style, or EAGLE-family comparisons until their entries are runnable and their provenance is captured by the same artifact path.
