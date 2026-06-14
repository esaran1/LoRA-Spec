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

python scripts/measure_logit_shift_rank.py \
  --target-model meta-llama/Meta-Llama-3-70B-Instruct \
  --adapter-path "${ADAPTER_PATH}" \
  --adapter-rank "${RANK}" \
  --adapter-domain chat \
  --device-map auto \
  --rank-estimation-mode projected \
  --projection-dim 512 \
  --projection-dimensions 128,256,512 \
  --projection-repetitions 5 \
  --batch-size 1 \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --verbose
