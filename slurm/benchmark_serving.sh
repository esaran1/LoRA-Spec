#!/bin/bash
#SBATCH --job-name=lora-spec-serve
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "hostname=$(hostname)"
echo "git_hash=$(git rev-parse HEAD)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "python=$(which python)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "cuda=$(python -c 'import torch; print(torch.version.cuda or \"unknown\")')"
python --version
nvidia-smi

PATTERNS=("uniform" "skewed_80_20" "bursty")
PATTERN="${PATTERNS[$SLURM_ARRAY_TASK_ID]}"

: "${VLLM_SERVER_URL:?Set VLLM_SERVER_URL to a concurrently serving vLLM endpoint}"
: "${SERVER_PROVENANCE_JSON:?Set SERVER_PROVENANCE_JSON to the server-side provenance manifest}"

CMD=(
  python scripts/benchmark_serving.py
  --target-model meta-llama/Meta-Llama-3-8B-Instruct
  --draft-model meta-llama/Llama-3.2-1B-Instruct
  --adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS
  --adapter-model-name code-lora
  --server-url "${VLLM_SERVER_URL}"
  --server-provenance-json "${SERVER_PROVENANCE_JSON}"
  --prompts-file data/prompts/pilot_v1/evaluation.jsonl
  --num-prompts 32
  --measurement-repetitions 3
  --traffic-pattern "${PATTERN}"
  --concurrency 4
  --requests-per-tenant 8
  --speculation-length 4
  --verbose
)

"${CMD[@]}"
