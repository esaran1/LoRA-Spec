This is LoRA-Spec, a research project investigating how LoRA adaptation degrades speculative decoding acceptance rates in multi-tenant LLM serving. Nobody has published on this interaction. The draft model in speculative decoding is aligned to the base target model's distribution, but when a LoRA adapter shifts the target's distribution (`W' = W + BA`), the draft model's guesses get rejected more often, silently degrading throughput.

Four planned contributions:
1. Empirical characterization of acceptance rate degradation across LoRA ranks `4/8/16/32/64`, domains `code/medical/chat/math`, fine-tuning intensities `1/3/10` epochs, and model pairs `Llama 3 8B/1B`, `Llama 3 70B/8B`, `Qwen 2.5 7B/0.5B`, and `Qwen 2.5 72B/7B`.
2. A predictive model estimating degradation from adapter properties such as Frobenius norm of `BA`, spectral norm, and calibration KL divergence without running inference.
3. Analytical logit correction methods including distribution offset vectors, low-rank SVD correction, and first-order Jacobian approximation that recover alignment at zero training cost.
4. Micro-LoRA draft adapters distilled via KL divergence from the adapted target for full recovery.

Infrastructure:
- University GPU cluster with A100 and H100 multi-GPU nodes for 70B experiments.
- Google Colab Pro A100 for prototyping and Phase 1 validation.
- Laptop RTX 3060 6 GB for code, analysis, and CPU-safe development only.

Frameworks and runtime assumptions:
- vLLM for speculative decoding, especially `vllm/spec_decode/rejection_sampler.py` and `vllm/lora/`.
- Hugging Face Transformers and PEFT for model and adapter loading.
- PyTorch for training and analytical methods.
- Nsight for profiling when needed.

Code standards:
- Python 3.10+.
- Type hints on every function signature and return.
- Pydantic for configs.
- Dataclasses for runtime result containers.
- Every script accepts `--config` YAML plus CLI overrides.
- Every experiment writes JSON results with timestamp, config hash, git hash, and full config.
- Use the Python `logging` module only, never `print()`.
- No TODOs, placeholders, or skeleton functions.
