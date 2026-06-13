This is LoRA-Spec, a theory-first research project on speculative decoding under LoRA adaptation in multi-tenant LLM serving.

Central thesis:
- A LoRA adapter is a low-rank weight perturbation, `W' = W + BA`.
- The induced shift in next-token logits, `delta_z(x) = z_adapted(x) - z_base(x)`, is hypothesized to be approximately low-rank and geometrically structured across contexts.
- This reframes the problem from "train a better draft model" into "characterize and correct a structured logit-space perturbation".

Paper spine:
1. The adapter-induced logit shift may be approximately low-rank; this is a contingent empirical claim, not a consequence of LoRA rank alone. Any hard rank bound must state its linearization and Jacobian assumptions explicitly.
2. A closed-form ridge reduced-rank regression operator minimizes its regularized calibration objective within the chosen output subspace. Its unregularized residual separates into the truncated-SVD tail and coefficient-regression residual.
3. There is a regime boundary where analytical correction fails because the shift becomes too nonlinear or too large, at which point training-based recovery is required.

Contingent empirical claims to test:
- Whether the measured logit shift matrix is actually low-rank in practice across families, ranks, and domains.
- Whether adapter rank predicts logit-shift effective rank after controlling for magnitude, domain, and calibration-set size.
- Whether speculative-decoding degradation is concentrated in early drafted positions or spreads across depth.
- Whether there is a sharp phase transition as adapter magnitude increases.
- Whether different adapters share a common dominant logit-shift subspace that supports transferable correction.

Research program:
1. Measure the logit-shift matrix `Delta` on calibration contexts and analyze its singular spectrum.
2. Relate effective rank, stable rank, participation ratio, and first-order linearization quality to adapter properties.
3. Build theory-grounded correction operators from `Delta`, then test whether approximation quality predicts acceptance-rate recovery.
4. Treat micro-LoRA draft distillation as the fallback upper bound when the analytical regime breaks down.

Experimental scope:
- Model pairs: `Llama 3 8B/1B`, `Llama 3 70B/8B`, `Qwen 2.5 7B/0.5B`, `Qwen 2.5 72B/7B`, plus a third family for architecture robustness.
- LoRA ranks: `4/8/16/32/64`.
- Domains: `code/medical/chat/math`.
- Fine-tuning intensities: `1/3/10` epochs.
- Adapter magnitude sweeps for phase-transition analysis.
- Baselines: autoregressive decoding, vanilla speculative decoding, analytical correction, static distillation, and stronger draft-side baselines such as EAGLE-family or DistillSpec-style methods when feasible.

Infrastructure:
- University GPU cluster with A100 and H100 nodes, including multi-GPU runs for `70B`-class targets.
- Google Colab Pro A100 for rapid prototyping and early-stage validation.
- Laptop RTX 3060 6 GB for code, analysis, plotting, and CPU-safe tests only.

Frameworks and runtime assumptions:
- vLLM for speculative decoding and LoRA serving. In the pinned vLLM release the sampler is `vllm/model_executor/layers/rejection_sampler.py`; newer releases may move it. LoRA support lives under `vllm/lora/`.
- Hugging Face Transformers and PEFT for model, tokenizer, and adapter handling.
- PyTorch for theory evaluation, correction operators, and distillation.
- Nsight when profiling inference overhead matters.

Research-grade implementation requirements:
- Python `3.10+`.
- Type hints on all public signatures and return values.
- Pydantic for configs and dataclasses for runtime results.
- Every script accepts `--config` YAML plus CLI overrides.
- Every experiment writes deterministic JSON artifacts with timestamp, config hash, git hash, seed, GPU/runtime metadata, and full config.
- Logging uses the Python `logging` module only, never `print()`.
- No TODOs, placeholders, or skeleton functions.
- Prefer simple, explicit code over framework-heavy abstraction, but keep the measurement path rigorous and reproducible.
- Label exact quantities, statistical estimates, projected/sketched estimates, and mathematical lower bounds distinctly in artifacts and figures.
- Analyze logits in a row-mean-centered gauge unless raw logits are explicitly required; per-context scalar logit offsets do not change the softmax distribution.
- Calibrate adapter geometry from `adapted target - base target`. Draft-model logits or hidden states may be correction features, but must never be substituted into the definition of adapter-induced shift.
- Treat the greedy speculative simulator as an offline diagnostic only. Claims about vLLM rejection-sampling acceptance must come from the instrumented vLLM path.
