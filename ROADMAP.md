# ShadowInfer Roadmap

This document consolidates the deep system analysis and optimization directions for ShadowInfer. It is a living guide for making ShadowInfer a high-quality, production-ready, and influential open-source project.

---

## 1. Project Vision

ShadowInfer aims to become a leading open-source inference optimization framework for Diffusion LLMs on edge and device GPUs. The vision is built on three pillars:

1. **Research Innovation**: Novel algorithms for KV cache compression, step-aware scheduling, and low-bit FFN computation.
2. **Production Readiness**: Reliable, observable, and scalable systems that can be deployed in real-world serving environments.
3. **Community & Reproducibility**: Clear documentation, reproducible benchmarks, and an active open-source community.

---

## 2. Current State Summary

### Strengths

- Clean multi-agent architecture (Orchestrator + Profiler / ShadowKV / Q-drift / FFN Optimizer).
- 630 passing tests (1 skipped) with 81.68% coverage.
- Comprehensive documentation (architecture, technical spec, engineering, ADRs, bilingual README, tutorials).
- Built-in observability, benchmarking, and production-safety components.
- CUDA kernel scaffold (INT8/INT4 quantization, sparse GEMM, fused quantized attention).
- Real model backends: `SimpleDiffusionLLM` / PyTorch and HuggingFace `AutoModelForCausalLM`.
- Neural-network-based learned multi-agent scheduler (`LearnedScheduler`).

### Critical Gaps Already Addressed

- [x] Added `pyproject.toml` for installable packaging.
- [x] Added `.gitignore` and cleaned committed cache/artifact directories.
- [x] Fixed code formatting and lint issues (black, isort, flake8).
- [x] Synchronized `requirements.txt` with actual dependencies.
- [x] Added `ModelBackend` abstraction for real model integration.
- [x] Added `SimpleDiffusionLLM` and `PyTorchModelBackend` for end-to-end real inference.
- [x] Integrated `ProductionSafetyNet` into the Orchestrator runtime loop.
- [x] Implemented token-level KV cache reuse in ShadowKV.
- [x] Implemented byte-packed mixed-precision KV cache storage.
- [x] Added Streamlit-based interactive web profiler.
- [x] Rewrote README to be project-focused and community-oriented.
- [x] Added development contribution section in README.md.
- [x] Added HuggingFace backend adapter (`HuggingFaceModelBackend`).
- [x] Added INT4 CUDA kernels and fused quantized attention kernel scaffold.
- [x] Added `LearnedScheduler` neural-network-based multi-agent scheduler.
- [x] Added bilingual content (now consolidated into README.md).

### Remaining High-Priority Gaps

- [~] Full CUDA kernel implementation (INT4, fused quantized attention) — scaffold complete; compilation/tuning pending GPU environment.
- [x] Kernel auto-tuning cache — persistent, shape-aware cache + CPU/CUDA dispatcher (CPU-testable scaffold).
- [x] Backend plugin system — `ModelBackend` registry with `pyproject.toml` entry-point discovery.
- [x] vLLM / SGLang backend adapter scaffolding — `VLLMModelBackend` / `SGLangModelBackend` with optional imports and graceful degradation.
- [x] Declarative Policy DSL — YAML/JSON rules for Q-drift, ShadowKV, FFN, Profiler, and Orchestrator arbitration (`configs/policy_default.yaml`).
- [ ] Real Diffusion LLM model adapter (Fast-dLLM-v2 or equivalent).
- [ ] Production serving integration with vLLM / SGLang adapter (full single-step diffusion forward).
- [x] **CPU optimization-path vectorization** (v3.4.0): ShadowKV decision / packing /
      compression hot paths are fully vectorized (compress chain 11.43s -> 0.575s,
      19.9x on the 6-step CPU benchmark); optimized-vs-baseline ratio improved from
      7.29x to 1.61x. Element-wise reference implementations retained as golden
      tests and exception fallbacks.
- [x] **FFN hot path + per-layer state** (v3.6.0, AR004): precision-plan
      fingerprint cached at pack time (O(1)), per-layer quantization statistics,
      vectorized unpack / mixed-precision quantization (reference implementations
      kept as golden), per-layer state isolation (FFN-P1), and down-projection
      column-mask fix. CPU same-batch microbench: on_step 1.95 ms -> 0.71 ms;
      fingerprint 151.4 us -> 0.3 us; unpack/quantize ~20x. 7B end-to-end benefit
      pending the server protocol in `specs/archive/AR004-ffn-hotpath/`.
- [x] **Decision-signal correctness + IRAP** (v3.5.0, AR003): explicit attention
      contract (single normalization, KV-P2), received-attention importance axis
      with sink protection (KV-P3), cross-step EMA with constant memory (KV-P9),
      pre-pack budget admission + pressure-driven global eviction (KV-P4/P8/P10),
      delta-A drift export to Q-drift. Benefit claims pending GPU experiment plan
      (`specs/changes/AR003-irap-signal-integrity/experiment-plan.md`).
- [x] **Pipeline constant-overhead elimination** (v3.7.0, AR005): memoized kernel
      backend resolution (success and failure) and batched KV INT4 byte packing;
      byte-identical to the reference packer. CPU same-process A/B: loader 0.022x,
      INT4 pack 0.016x.
- [ ] **AOG overhead gating** (planned): device-aware adaptive overhead gating
      with K-step decision cadence and pre-flight arbitration (ORC-P1/ORC-P3).
- [ ] **Macroblock rate control** (planned): SKIP/DELTA/REFRESH reuse modes with
      rate-distortion control and per-precision free-list slab allocator.
- [x] Continuous benchmarking and performance regression tracking.
- [x] English documentation and broader internationalization — all content consolidated into the Chinese README.md.

---

## 3. Optimization Directions

### 3.1 Architecture & Design

- [x] **Policy DSL**: Implemented declarative optimization policy language (`shadowinfer/core/policy.py`) with YAML/JSON rules replacing hard-coded dispatch thresholds in Orchestrator arbitration.
- [~] **Event Sourcing**: Step-level snapshot persistence implemented (JSON in `snapshot_dir`); full replay/offline training pipeline pending.
- [x] **Async Agent Execution**: `AsyncTaskExecutor` for background Profiler aggregation, A/B statistics, dashboard rendering, etc. (`shadowinfer/core/async_executor.py`).
- **Plugin System**: Support third-party agents via `pyproject.toml` entry points.

### 3.2 Algorithms & Models

- **Real Model Adapters**: Implement adapters for Fast-dLLM-v2 and other Diffusion LLMs through the `ModelBackend` interface.
  - Status: `PyTorchModelBackend` + `SimpleDiffusionLLM` reference implementation; `HuggingFaceModelBackend` for generic causal LMs.
- **Calibration & Auto-Tuning**: Use Bayesian optimization to learn Q-drift dispatch matrices from real validation data.
- [x] **Uncertainty-Aware Step Budget**: `UncertaintyAwareEarlyStopper` integrated into Orchestrator; stops denoising early when latent/output changes fall below a configurable threshold.
- **Multi-Objective Pareto Optimization**: Trade off latency, memory, and accuracy via surrogate models and NSGA-II/MOBO.

### 3.3 CUDA Kernels & Hardware Acceleration

- [x] **INT4 CUDA Kernels**: Implemented `quantize_per_channel_int4`, `dequantize_per_channel_int4`, pack/unpack, and fused quantized attention kernels (CPU fallbacks active; compile/tuning pending GPU environment).
- [ ] **Triton Kernels**: Add Triton implementations for FFN mixed-precision and sparse paths to improve maintainability.
- [x] **Kernel Auto-Tuning**: Implemented persistent per-GPU auto-tuning cache (`KernelAutoTuner`) and CPU/CUDA dispatcher; GPU benchmark loop will be enabled once CUDA compilation is available.
- **FFN Kernel Fusion**: Fuse `quantize → matmul → activation → dequantize → residual` into single kernels.

### 3.4 ShadowKV

- [x] Token-level reuse decisions.
- [x] Packed KV cache storage (real byte-level compression).
- [x] Importance-aware eviction for long-context scenarios (production-wired in AR003:
      pre-pack admission + memory-pressure driven cross-layer eviction).
- [x] KV cache prefetching based on Q-drift sensitivity predictions.
- [x] Formalize ShadowKV as Importance / Precision / Reuse planes.
- [x] IRAP received-attention importance: column-wise received attention, EMA
      propagation, rank normalization, sink hard protection, delta-A drift signal.
- [ ] Wire `KVDecisionPlane` singleton as the single dispatch path (KV-P7 remainder, AR005).

### 3.5 FFN Optimization

- [x] **Real Sparse FFN Kernel**: Integrate packed INT8/INT4 weights and the existing `sparse_gemm` kernel.
- [ ] **Real Sparse FFN Kernel**: Use CSR/CSC formats and a production-grade sparse kernel.
- **Packed Weight Storage**: Store weights as packed INT8/INT4 with on-the-fly dequantization.
- [x] **Vectorized Packing** (v3.8.0, AR006): grouped batched FFN pack
      (fp32/fp16/int8/int4, pack_dim 0/1), bit-identical to the per-channel
      reference; CPU same-process A/B 0.072-0.082x at 256/1024 channels.
- [x] **Vectorized Unpack/Dequantization** (v3.6.0, AR004): grouped batched unpack and mixed-precision quantization (element-wise reference kept as golden); precision-plan fingerprint cached at pack time. CPU same-process A/B: unpack 0.044-0.070x, quantize 0.053-0.077x at 256/1024 channels.
- **FFN-Compiler**: Lightweight compiler to generate fused kernels from FFN subgraphs.
- [x] **Dynamic Channel Pruning**: Zero out low-contribution channels at runtime.

### 3.6 Production & Serving

- [x] `ProductionSafetyNet` integrated into Orchestrator.
- [x] Dependency-light local HTTP server (`shadowinfer serve`) with `/generate`, `/health` and `/metrics` endpoints.
- [x] Token-bucket rate limiting and concurrency control in the serving wrapper.
- [x] A/B testing integration into the request path with configurable weights.
- [x] Hot config reload wired into serving runtime decisions.
- [x] vLLM / SGLang backend adapter scaffolding (`VLLMModelBackend` / `SGLangModelBackend`); full single-step diffusion forward pending engine internals.

### 3.7 Observability

- [x] **Standard Backends**: Migrate metrics to `prometheus_client` and tracing to OpenTelemetry.
- [x] **Live Dashboard**: Build a Streamlit interactive profiler.
- [x] **Structured Logging**: Adopt `structlog` with log rotation and dynamic levels.

### 3.8 Testing & Quality

- **CUDA Kernel Tests**: Add GPU correctness tests for all CUDA kernels.
- [x] **Real Model Smoke Tests**: Run end-to-end tests with `SimpleDiffusionLLM` in CI.
- [ ] **Real Model Smoke Tests**: Run end-to-end tests with a publicly available Diffusion LLM in CI.
- [x] **Performance Regression Tests**: Track latency/memory/accuracy across PRs via `RegressionTracker` and `shadowinfer benchmark`.
- [x] **Reproducible Experiment Harness** (v3.9.0, AR007): one-command real-backend
      baseline/optimized runs with fixed seed, per-repeat KV reset, config/environment
      digests, median/min/max summaries, isolated `run_i/` artifacts, and s1/s2/s3
      scale profiles; mock backends are rejected.
- **Property-Based Testing**: Use Hypothesis for quantization and metrics fuzzing.

### 3.9 Distributed Inference

- Implement real pipeline parallelism and tensor parallelism using `torch.distributed`.
- Add distributed KV cache synchronization and load balancing.

### 3.10 Community & Ecosystem

- Bilingual documentation (English + Chinese).
- Academic paper / arXiv report on ShadowKV and Q-drift.
- Reproducibility artifacts (Docker, scripts, checkpoints).
- Benchmark leaderboard with automated result tracking.

---

## 4. Innovation Proposals

These directions can differentiate ShadowInfer on GitHub and in the research community:

1. **ShadowInfer Runtime**: A dedicated `shadowinfer serve` CLI for Diffusion LLM serving.
2. [x] **Learned Multi-Agent Scheduler**: Implemented `LearnedScheduler` with an MLP policy trained on `StepExperience` rewards; used to override Q-drift dispatch decisions.
3. **ShadowKV 2.0**: Token-level compressed sparse KV cache with learned importance.
4. **FFN-Compiler**: Automatic operator fusion and code generation for FFN subgraphs.
5. **Speculative Decoding for Diffusion LLM**: Cross-step draft-then-verify acceleration.
6. **Edge-to-Cloud Adaptive Offloading**: Offload layers or KV cache when device memory is exhausted.
7. **Model-Aware Quantization Search (MQS)**: Auto-search optimal per-layer quantization strategies.
8. **Reproducible Benchmark Leaderboard**: Automated benchmarking across multiple GPUs.
9. **Interactive Web Profiler**: Web-based visualization of profiling results.
10. **Academic Paper + Artifact**: Publish core contributions and provide reproducibility artifacts.

---

## 5. Phased Roadmap

### Phase 0: Foundation (Completed)

- [x] Git-ready project structure (`.gitignore`, `pyproject.toml`, README.md with contribution guidelines).
- [x] Code quality tooling (black, isort, flake8, mypy, pytest-cov).
- [x] Clean working tree (removed cache/artifact directories).
- [x] README rewritten for open-source community.
- [x] `ModelBackend` abstraction.
- [x] `ProductionSafetyNet` integration.
- [x] Token-level ShadowKV reuse.

### Phase 1: Runnability (0–4 weeks)

- [x] `pip install -e .` fully functional.
- [x] First real Diffusion LLM adapter (`SimpleDiffusionLLM` + `PyTorchModelBackend`).
- [x] CLI commands: `shadowinfer profiler` / `shadowinfer optimize` / `shadowinfer compare` / `shadowinfer dashboard`.
- [x] Lock file for reproducible dependencies (`requirements-lock.txt`).
- [x] Core docs consolidated into README.md.

### Phase 2: Algorithm Hardwareization (1–2 months)

- [x] INT4 CUDA kernels (scaffold).
- [x] Fused quantized attention kernel (scaffold).
- [x] FFN sparse path calling custom kernel (`sparse_gemm_ffn` integration).
- [x] Packed KV cache storage.
- [x] Kernel auto-tuning cache.

### Phase 3: Production Closure (2–3 months)

- [x] Rate limiting and concurrency control.
- [x] Prometheus / OpenTelemetry integration.
- [x] A/B testing in request path.
- [x] vLLM / SGLang adapter scaffolding.
- [x] Live web dashboard (`shadowinfer dashboard`).

### Phase 4: Influence & Ecosystem (3–6 months)

- [ ] arXiv / paper publication.
- [ ] Reproducibility artifact.
- [ ] Benchmark leaderboard.
- [x] Learned scheduler.
- [ ] Edge-to-cloud offloading.
- [ ] Active community (issues, discussions, contributors).

---

## 6. Success Metrics

| Metric | Current | 3-Month Target | 6-Month Target |
|--------|---------|----------------|----------------|
| GitHub Stars | N/A | 200+ | 1000+ |
| pip Installable | ✅ | ✅ | ✅ |
| Real Model Support | Partial (`SimpleDiffusionLLM` + HuggingFace adapter) | 1 public Diffusion LLM | 3+ |
| CUDA Kernel Completion | ~35% | 70% | 90%+ |
| Test Coverage | ~81.68% | >82% | >85% |
| Docs Languages | Chinese + English intro | Bilingual core | Bilingual full |
| Contributors | 1 | 5+ | 20+ |
| Academic Output | 0 | 1 arXiv | 1 paper + artifact |

---

## 7. How to Contribute

See the [开发贡献](README.md#开发贡献) section in README.md for development setup, code style, and PR workflow.

Priority contribution areas:

- CUDA kernel implementation and testing.
- Real Diffusion LLM adapters.
- Documentation and tutorials.
- Benchmarking on edge GPUs.
- Observability backend integrations.

---

## 8. References

- 所有历史文档已合并到 `README.md`，包括系统架构、技术规范、工程化设计、Agent 协作规范与版本历史。
