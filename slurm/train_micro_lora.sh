#!/bin/bash
#SBATCH --job-name=lora-spec-distill
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A.out

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "hostname=$(hostname)"
echo "git_hash=$(git rev-parse HEAD)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "python=$(which python)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "cuda=$(python -c 'import torch; print(torch.version.cuda or \"unknown\")')"
python --version
nvidia-smi

python scripts/train_micro_lora.py \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --target-model meta-llama/Meta-Llama-3-8B \
  --target-adapter-path bootscoder/Llama-3-Medical-8B-SFT-LoRA \
  --prompts-file data/prompts/pilot_v1/calibration.jsonl \
  --num-prompts 12 \
  --num-validation-prompts 4 \
  --draft-lora-rank 4 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --epochs 2 \
  --max-length 512 \
  --verbose
