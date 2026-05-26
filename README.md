# LoRA-Spec

LoRA-Spec studies how LoRA adaptation degrades speculative decoding acceptance rates in multi-tenant LLM serving. The central hypothesis is that speculative decoding works because the draft model is aligned to the base target distribution, but a LoRA adapter shifts the target distribution enough that the draft model's proposals get rejected more often and throughput falls.

## Research Scope

The repository is structured around six phases:

1. Phase 1 hypothesis validation on a single target and draft pair with a public LoRA adapter.
2. Phase 2 characterization across model pairs, LoRA ranks, domains, and fine-tuning intensity.
3. Phase 3 predictive modeling from adapter properties.
4. Phase 4 analytical correction methods.
5. Phase 5 micro-LoRA distillation on the draft model.
6. Phase 6 multi-tenant serving benchmarks under uniform, skewed, and bursty traffic.

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

## Repository Layout

- `src/lora_spec/`: core library code.
- `scripts/`: experiment entry points for each phase.
- `configs/`: model, adapter, and serving benchmark definitions.
- `notebooks/01_hypothesis_check.ipynb`: Colab-ready Phase 1 notebook.
- `slurm/`: cluster launch templates for 8B, 70B, distillation, and serving runs.
- `tests/`: CPU-only synthetic tests.
- `research_log/log.md`: experiment log template.

## Reproducing Phase 1

```bash
python scripts/validate_hypothesis.py \
  --target-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Meta-Llama-3-1B-Instruct \
  --adapter-path bootscoder/Llama-3-Medical-8B-SFT-LoRA \
  --adapter-rank 16 \
  --adapter-domain medical \
  --adapter-epochs 3 \
  --dataset tatsu-lab/alpaca \
  --num-prompts 32 \
  --speculation-length 4 \
  --gpu-memory-utilization 0.85 \
  --verbose
```

Results are written to `results/phase1/` with timestamp, config hash, git hash, and full config.

## Multi-Tenant Serving Benchmark

The serving benchmark supports:

- `uniform` request distribution across tenants.
- `skewed_80_20` demand concentration where 20% of tenants receive 80% of requests.
- `bursty` arrivals for concurrency spikes.

Example:

```bash
python scripts/benchmark_serving.py \
  --target-model meta-llama/Meta-Llama-3-8B-Instruct \
  --draft-model meta-llama/Meta-Llama-3-1B-Instruct \
  --adapter-path bootscoder/Llama-3-Medical-8B-SFT-LoRA \
  --adapter-path grimjim/Llama-3-Instruct-Nephilim-v3-LoRA-8B \
  --traffic-pattern skewed_80_20 \
  --concurrency 4 \
  --requests-per-tenant 8 \
  --num-prompts 128 \
  --verbose
```

## Colab Notebook

Open [notebooks/01_hypothesis_check.ipynb](/Users/Evan/nvidiaresearch/LoRA-Spec/notebooks/01_hypothesis_check.ipynb). The notebook installs dependencies, authenticates to Hugging Face, downloads the gated Llama 3 base models plus public LoRA adapters, runs the hypothesis check, and plots the acceptance and throughput comparison inline.
