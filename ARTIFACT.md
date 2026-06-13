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

Use a fixed prompt file for exact reruns of the theory experiments.

## Interpretation Rules

- `expected_rejection_sampling_acceptance` is the full-vocabulary overlap `sum_v min(p_v, q_v) = 1 - TV(p, q)` on held-out contexts.
- `logit_acceptance_lower_bound` is a certified lower bound derived from the residual-logit span, not a fitted predictor.
- `greedy_proxy_acceptance_rate` is a sequence-level diagnostic and must not be presented as standard rejection-sampling acceptance.
- Results with `spectrum_is_approximate` or `subspace_is_approximate` use a shared random vocabulary projection. Report the projection dimension and seed.
- Rank and correction analyses use row-mean-centered logits to remove the softmax-invariant scalar gauge.
- Correction calibration labels are always computed from `adapted target - base target`; draft logits or hidden states are only application-time features.
- Per-position vLLM acceptance is conditional on reaching that drafted position. `acceptance_by_depth[d]` is the fraction of eligible speculative steps whose first `d + 1` proposals all survive.

## Runtime Scope

The pinned `vllm==0.5.3.post1` environment is retained to make the sampler instrumentation reproducible. Use it only with trusted checkpoints on a trusted research network. The hook supports the pinned single-process sampler path and fails closed when decisions execute in an unobserved worker process. Analytical correction scripts currently report exact distribution overlap plus an explicitly labeled greedy sequence proxy; they do not claim corrected vLLM serving until a proposer-side correction integration is implemented and measured.

## Figure Map

### Figure 1: Effective Rank vs Adapter Rank and Magnitude

1. Prepare a calibration prompt file with one prompt per line.
2. Run:

```bash
python scripts/measure_logit_shift_rank.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --prompts-file data/calibration_prompts.txt \
  --verbose
```

3. Plot:

```bash
python scripts/plot_results.py --input-dir results --output-dir results/plots --verbose
```

Output figure: `results/plots/theory_effective_rank.png`

### Figure 2: Approximation Error vs Acceptance Recovery

```bash
python scripts/validate_correction_theory.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --adapted-adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS \
  --prompts-file data/calibration_prompts.txt \
  --eval-prompts-file data/eval_prompts.txt \
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
  --prompts-file data/calibration_prompts.txt \
  --eval-prompts-file data/eval_prompts.txt \
  --magnitude-values 0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
  --verbose
```

Output figure: `results/plots/theory_phase_transition.png`

### Figure 4: Shared Correction Subspace

```bash
python scripts/subspace_sharing.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --model-pair llama3_8b_1b \
  --prompts-file data/calibration_prompts.txt \
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

Use the provided templates in `slurm/` for multi-GPU runs. Each template records:

- git hash
- conda environment
- GPU model
- CUDA version

## Expected Outputs

All result artifacts are JSON under `results/` and all figures are written under `results/plots/`.
