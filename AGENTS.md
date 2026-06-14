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
- vLLM `0.15.1` for combined V1 draft-model speculative decoding and LoRA serving. The instrumented sampler is `vllm/v1/sample/rejection_sampler.py`; the engine core must remain in-process so acceptance decisions are observable.
- Hugging Face Transformers and PEFT for model, tokenizer, and adapter handling.
- PyTorch for theory evaluation, correction operators, and distillation.
- Nsight when profiling inference overhead matters.

Research-grade implementation requirements:
- Python `3.10+`.
- Type hints on all public signatures and return values.
- Pydantic for configs and dataclasses for runtime results.
- Every script accepts `--config` YAML plus CLI overrides.
- Every experiment writes deterministic JSON artifacts with timestamp, config hash, git hash, seed, GPU/runtime metadata, and full config.
- Calibration and evaluation prompts must come from a versioned frozen split with byte-level SHA-256 hashes. Result configs must include prompt and manifest provenance, and evaluation prompts must never be used to fit a correction.
- Geometry and correction measurements use deterministic base-target continuation trajectories. Measure logits beginning at the final prompt token, which predicts the first continuation token; do not substitute teacher-forced prompt-interior positions for speculative proposal contexts.
- Resolve every Hugging Face model and adapter reference to an immutable commit SHA before loading it. Include requested and resolved revisions in the hashed experiment config; hash local artifacts by content.
- Any full-vocabulary comparison requires exact tokenizer equivalence: token-to-ID vocabulary, added vocabulary, special-token IDs, and probe encodings must match.
- Phase 1 systems comparisons use paired repeated measurements with randomized condition order, preserve each replicate, and base go/no-go decisions on confidence intervals rather than one timing run.
- Random-projection rank analyses use multiple independent sketches and report the sketch-level estimates and ranges. Never present a projected spectrum as exact.
- Projected-rank claims must include a projection-dimension sensitivity sweep; repeated sketches quantify random-sketch variance but not projection-dimension bias.
- Treat `configs/adapters.yaml` as a pilot design until `scripts/validate_experiment_design.py --strict` passes. Magnitude-scaled copies of one adapter are controls, not independent adapter sources, and cannot identify rank, domain, or model-family effects.
- Paper-ready factorial cells require the configured number of independently trained adapter sources, normally at least three, and adapter target models must match the selected model pair.
- The instrumented vLLM acceptance path is single-process only. Do not report tensor-parallel acceptance until worker-side decision aggregation is implemented and validated; multi-GPU templates currently support theory/model-analysis experiments instead.
- Do not downgrade the serving runtime below vLLM `0.15.1`: older tagged releases either reject LoRA on the speculative worker or do not support an ordinary draft model in the LoRA-capable V1 engine.
- Exact spectral claims use float64 covariance accumulation or direct float64 SVD. Never infer effective rank from a float32 Gram matrix because its squared condition number can erase the spectral tail.
- Colab correction experiments stage model residency: collect base/adapted labels first, release or offload the base, then load the draft for correction fitting and held-out evaluation.
- Serving and distillation experiments must use frozen registered prompt splits. External vLLM server runs must separately record and verify the server-side model and adapter revisions.
- Micro-LoRA checkpoint selection uses a deterministic, disjoint train/validation partition of the frozen calibration split. The evaluation split is never used for optimization, early stopping, or checkpoint selection.
- External vLLM adapters must be preloaded with `--lora-modules` and selected by their registered `model` name. Any serving request failure invalidates the benchmark run.
- Predictive-model validation must group all magnitude variants and repeated measurements from the same adapter source into the same fold. Report source-held-out and model-family-held-out results; row-wise LOOCV is diagnostic only.
- First-order Jacobian analysis must keep LoRA perturbations factored and materialize bounded tangent groups. Do not construct all dense `BA` matrices simultaneously.
- First-order residual claims support plain LoRA only unless DoRA magnitudes, trainable biases, and modules-to-save are explicitly represented. Unsupported PEFT variants must fail closed.
- Logging uses the Python `logging` module only, never `print()`.
- No TODOs, placeholders, or skeleton functions.
- Prefer simple, explicit code over framework-heavy abstraction, but keep the measurement path rigorous and reproducible.
- Label exact quantities, statistical estimates, projected/sketched estimates, and mathematical lower bounds distinctly in artifacts and figures.
- Serving latency percentiles use a named estimator, and publication-facing concurrency claims require the external HTTP load path rather than the serialized in-process smoke-test path.
- CI must pass lint, formatting, CPU tests, prompt-split verification, shell syntax checks, compilation, and package construction before experiment code is merged.
- Treat token positions from one prompt as correlated. Held-out uncertainty must resample prompts or independent runs, never individual token rows.
- Effective-rank claims must report the finite-sample rank ceiling and prompt-count sensitivity; a ceiling-saturated result is not evidence of intrinsic low rank.
- Treat segmented magnitude breakpoints as exploratory unless replicated across independent adapters with uncertainty and held-out confirmation.
- Analyze logits in a row-mean-centered gauge unless raw logits are explicitly required; per-context scalar logit offsets do not change the softmax distribution.
- Calibrate adapter geometry from `adapted target - base target`. Draft-model logits or hidden states may be correction features, but must never be substituted into the definition of adapter-induced shift.
- Treat the greedy speculative simulator as an offline diagnostic only. Claims about vLLM rejection-sampling acceptance must come from the instrumented vLLM path.
