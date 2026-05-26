#!/bin/bash
#SBATCH --job-name=lora-spec-70b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
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

RANKS=(4 8 16 32 64)
RANK="${RANKS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${ADAPTER_PATHS_FILE:-}" ]]; then
  echo "Set ADAPTER_PATHS_FILE to a text file containing one compatible 70B adapter path per line."
  exit 1
fi

mapfile -t ADAPTER_PATHS < "${ADAPTER_PATHS_FILE}"
ADAPTER_PATH="${ADAPTER_PATHS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${ADAPTER_PATH:-}" ]]; then
  echo "Missing adapter path for array index ${SLURM_ARRAY_TASK_ID} in ${ADAPTER_PATHS_FILE}"
  exit 1
fi

python scripts/validate_hypothesis.py \
  --target-model meta-llama/Meta-Llama-3-70B-Instruct \
  --draft-model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-degree 4 \
  --adapter-path "${ADAPTER_PATH}" \
  --adapter-rank "${RANK}" \
  --adapter-domain chat \
  --adapter-epochs 3 \
  --dataset tatsu-lab/alpaca \
  --num-prompts 32 \
  --speculation-length 4 \
  --gpu-memory-utilization 0.85 \
  --verbose
