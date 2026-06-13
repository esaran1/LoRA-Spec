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

CMD=(
  python scripts/benchmark_serving.py
  --target-model meta-llama/Meta-Llama-3-8B-Instruct
  --draft-model meta-llama/Llama-3.2-1B-Instruct
  --adapter-path AdnanRiaz107/CodeLLAMA3-8BI-APPS
  --dataset tatsu-lab/alpaca
  --num-prompts 128
  --traffic-pattern "${PATTERN}"
  --concurrency 4
  --requests-per-tenant 8
  --speculation-length 4
  --verbose
)

if [[ -n "${VLLM_SERVER_URL:-}" ]]; then
  CMD+=(--server-url "${VLLM_SERVER_URL}")
fi

"${CMD[@]}"
