# LoRA-Spec Run Ledger

This ledger tracks planned and completed experiments. Add a row before launching a run, then fill in result paths and notes after completion. YAML configs remain the source of truth; this file is the operational record.

Status values: `planned`, `running`, `completed`, `failed`, `superseded`.

## Core ICLR Validation Runs

| Run ID | Status | Phase | Script | Model Pair | Adapter | Prompt Split | Seed | Required Resource | Expected Output | Result Path | Notes |
|---|---|---|---|---|---|---|---:|---|---|---|---|
| LS-RANK-001 | planned | Effective rank smoke | `scripts/measure_logit_shift_rank.py` | `llama3_8b_base_1b` | `medical_rank64_scale100` | `pilot_v1` calibration | 13 | GPU-A100 | Singular spectrum, effective rank, adapter magnitude |  | First real viability check for low-rank thesis. |
| LS-RANK-002 | planned | Domain contrast | `scripts/measure_logit_shift_rank.py` | `llama3_8b_1b` | `code_rank16_scale100` | `pilot_v1` calibration | 13 | GPU-A100 | Effective rank vs domain/rank |  | Tests whether code adapter also yields compressible shift. |
| LS-RANK-003 | planned | Domain contrast | `scripts/measure_logit_shift_rank.py` | `llama3_nephilim_8b_1b` | `chat_rank32_scale100` | `pilot_v1` calibration | 13 | GPU-A100 | Effective rank vs domain/rank |  | Tests chat adapter and non-base target path. |
| PHASE-001 | planned | Magnitude sweep | `scripts/phase_transition_sweep.py` | `llama3_8b_base_1b` | `medical_rank64_scale025` through `medical_rank64_scale200` | `pilot_v1` calibration/evaluation | 13 | GPU-A100 | Nonlinearity residual vs magnitude |  | Detects analytical-correction failure regime. |
| SUBSPACE-001 | planned | Shared subspace | `scripts/subspace_sharing.py` | Llama 8B family | Medical/code/chat anchors | `pilot_v1` calibration | 13 | GPU-A100 | Pairwise principal-angle matrix |  | Tests common correction subspace claim. |
| SPEC-001 | planned | Acceptance baseline | `scripts/validate_hypothesis.py` | `llama3_8b_base_1b` | none | `pilot_v1` evaluation | 13 | GPU-A100 | Base speculative acceptance/TPS/TTFT |  | Baseline for LoRA degradation. |
| SPEC-002 | planned | LoRA degradation | `scripts/validate_hypothesis.py` | `llama3_8b_base_1b` | `medical_rank64_scale100` | `pilot_v1` evaluation | 13 | GPU-A100 | LoRA-shifted acceptance/TPS/TTFT |  | Compare directly to `SPEC-001`. |
| CORR-001 | planned | Correction theory | `scripts/validate_correction_theory.py` | `llama3_8b_base_1b` | `medical_rank64_scale100` | `pilot_v1` calibration/evaluation | 13 | GPU-A100 | Approximation error and recovery vs correction rank |  | First link between geometry and acceptance recovery. |
| MICRO-001 | planned | Training upper bound | `scripts/train_micro_lora.py` | `llama3_8b_base_1b` | `medical_rank64_scale100` | `pilot_v1` calibration/evaluation | 13 | GPU-A100 | Draft micro-LoRA checkpoint and recovery estimate |  | Run only after analytical correction has a baseline. |
| SERVE-001 | planned | Multi-tenant pilot | `scripts/benchmark_serving.py` | `llama3_8b_base_1b` | medical/code/chat anchors | `pilot_v1` evaluation | 13 | GPU-A100 | Uniform/skewed/bursty acceptance and throughput |  | Systems pilot after single-tenant results. |

## Scale-Up Runs

| Run ID | Status | Phase | Script | Model Pair | Adapter | Prompt Split | Seed | Required Resource | Expected Output | Result Path | Notes |
|---|---|---|---|---|---|---|---:|---|---|---|---|
| SCALE-70B-001 | planned | Scale validation | `scripts/measure_logit_shift_rank.py` | `llama3_70b_8b` | paper-grade Llama 70B adapter | final split | 13 | GPU-multi | Effective rank at 70B scale |  | Requires 4x A100/H100 and pinned adapter revision. |
| SCALE-QWEN-001 | planned | Architecture validation | `scripts/measure_logit_shift_rank.py` | `qwen25_7b_05b` | paper-grade Qwen adapter | final split | 13 | GPU-A100 | Effective rank on Qwen family |  | Shows result is not Llama-specific. |
| SCALE-GEMMA-001 | planned | Architecture validation | `scripts/measure_logit_shift_rank.py` | `gemma2_9b_2b` | paper-grade Gemma adapter | final split | 13 | GPU-A100 | Effective rank on Gemma family |  | Confirms third-family breadth. |

## MacBook-Only Maintenance Runs

| Run ID | Status | Phase | Command | Required Resource | Expected Output | Notes |
|---|---|---|---|---|---|---|
| SMOKE-001 | completed | Rank script smoke | `python scripts/measure_logit_shift_rank.py --synthetic-smoke-test --seed 13 --output-dir results/smoke --plots-dir results/smoke/plots` | MacBook | Synthetic rank JSON and spectrum plot | Result: `results/smoke/measure_logit_shift_rank_smoke_20260622T223353427277Z_97112ea899a980b7.json`; no HF downloads or GPU used. |
| CHECK-001 | planned | Repo health | `pytest -q` | MacBook | CPU-safe test pass | Run before every push. |
| CHECK-002 | planned | Style | `ruff check . && ruff format --check .` | MacBook | Lint and formatting pass | Run before every commit. |
| CHECK-003 | planned | Prompt integrity | `python scripts/verify_prompt_splits.py --verbose` | MacBook | Prompt hashes verified | Required before launching GPU runs. |
| CHECK-004 | planned | Design validation | `python scripts/validate_experiment_design.py --config configs/adapters.yaml --verbose` | MacBook | Coverage/design report | Confirms configs are coherent. |
