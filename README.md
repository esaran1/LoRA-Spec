# LoRA-Spec

LoRA-Spec studies speculative decoding under LoRA adaptation from a theory-first perspective. The central object is the adapter-induced next-token logit shift

`delta_z(x) = z_adapted(x) - z_base(x)`.

Because LoRA is a low-rank weight perturbation, `W' = W + BA`, the project tests whether `delta_z(x)` is approximately low-rank in practice. LoRA rank alone does not imply a hard rank bound on the cross-context logit-shift matrix because the model Jacobian varies with context.

## Research Spine

The repository supports three core claims:

1. Determine when the empirical logit-shift matrix is low-rank or near-low-rank.
2. Evaluate a PCA-output-subspace ridge operator whose calibration residual decomposes into spectral-tail and coefficient-regression terms.
3. There is a phase boundary where the shift becomes too nonlinear or too large for analytical correction, and training becomes necessary.

System benchmarks remain in the repo, but they now validate the theoretical story rather than define it.

All logit-space rank results use a row-mean-centered gauge because softmax distributions are invariant to a per-context scalar offset. Correction operators are calibrated on `adapted target - base target`; draft outputs are application-time features, not part of the adapter-shift definition.

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

Measure effective rank of the logit-shift matrix:

```bash
python scripts/measure_logit_shift_rank.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --model-pair llama3_8b_1b \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --verbose
```

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
