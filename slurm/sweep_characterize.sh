#!/bin/bash
#SBATCH --job-name=lora-spec-char
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

MODEL_KEYS=("llama3_8b_1b" "llama3_8b_1b" "qwen25_7b_05b")
ADAPTER_KEYS=("medical_rank16" "chat_high_rank" "code_rank32")

MODEL_KEY="${MODEL_KEYS[$SLURM_ARRAY_TASK_ID]}"
ADAPTER_KEY="${ADAPTER_KEYS[$SLURM_ARRAY_TASK_ID]}"

python scripts/characterize.py \
  --models-config configs/models.yaml \
  --adapters-config configs/adapters.yaml \
  --override selected_model="${MODEL_KEY}" \
  --override selected_adapter="${ADAPTER_KEY}" \
  --num-prompts 32 \
  --speculation-length 4 \
  --verbose
