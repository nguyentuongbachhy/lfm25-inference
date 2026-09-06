# LFM2.5-1.2B Inference Runtime — Final Validated Release Report

## Status

**FINAL VALIDATED MILESTONE — FROZEN PRODUCTION BASELINE**

This report establishes the final, empirical production release baseline for `LFM2.5-1.2B-Instruct` on the NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell GeForce SM120 / `compute_120`).

Following a comprehensive empirical audit and exact-defaults ABBA benchmarking across multiple languages, all runtime defaults have been locked to conservative, mathematically validated configurations. Features that failed exact bitwise parity or did not meet the strict $\ge 1.0\%$ end-to-end performance improvement threshold have been demoted to opt-in experimental flags.

---

## Final Production Configuration Matrix

| Component / Feature | Production Default | Rationale & Gate Decision |
|---|:---:|---|
| **Selective FP8 E4M3 Decode** | **ON** (`selected-policy.json`) | **ACCEPT**: +0.226% NLL delta (within $\le 1\%$ gate), 1.221x TPOT speedup at short context. |
| **Tensor Core Contiguous FlashAttention** | **ON** | **ACCEPT**: Up to 2.513x TTFT speedup at 8K prefill, bitwise identical outputs. |
| **Segmented / Ragged FlashAttention** | **ON** | **ACCEPT**: 1.703x E2E speedup on batched prefill ($B=2 \times 2048$), zero waste on padding. |
| **Scratchless Atomic Greedy Argmax** | **ON** ($B \le 16$) | **ACCEPT**: 2.0–2.54x kernel speedup, 100% top-1 agreement against legacy CPU/staged argmax. |
| **Paged KV Cache (PS16 + Split-K)** | **ON** | **ACCEPT**: Zero reallocation, near-optimal memory bandwidth utilization. |
| **Bounded CUDA Graphs** | **ON** (`LFM25_CUDA_GRAPHS=true`) | **ACCEPT**: Eliminates host kernel launch jitter, 9.3x reduction in CPU dispatch overhead. |
| **Radix Tree Multi-Turn Prefix Caching** | **ON** | **ACCEPT**: **6.20x TTFT speedup** on multi-turn conversations via automated KV block reuse. |
| **Continuous GPU Batching Serving Engine** | **ON** | **ACCEPT**: Scales from 138 tok/s ($C=1$) to **817+ tok/s** ($C=8$) on laptop GPU envelope. |
| **OpenAI & Ollama Drop-in Endpoints** | **ON** | **ACCEPT**: Full SSE streaming, NDJSON streaming, and JSON Schema structured outputs verified. |
| **Speculative Decoding (`speculative_draft`)** | **OFF** (`default: 0`) | **REJECT AS DEFAULT**: Fails golden reference bitwise parity (80% mismatch due to hybrid SSM/conv rollback). Opt-in via `--speculative-draft <N>`. |
| **Fused Residual RMSNorm $\to$ FP8** | **OFF** (`default: false`) | **REJECT AS DEFAULT**: 0.55% standalone E2E gain (fails $\ge 1.0\%$ threshold). Opt-in via `LFM25_FUSED_RMS_FP8=1`. |
| **Fused SwiGLU $\to$ FP8** | **OFF** (`default: false`) | **REJECT AS DEFAULT**: 0.05%–0.13% incremental gain over baseline (pure noise, fails $\ge 1.0\%$). Opt-in via `LFM25_FUSED_SWIGLU_FP8=1`. |
| **Metadata Scratchpad** | **OFF** (`default: false`) | **REJECT**: Causes a -0.17% regression in decode throughput. |
| **NVFP4 Quantization** | **REJECTED** | **REJECT**: Fails hidden-state propagation quality gates (see `docs/research/nvfp4_rejection.md`). |

---

## Exact Defaults ABBA Benchmark & Empirical Gating Decisions

To eliminate compiler, order, and thermal bias, an exact Latin-square order-balanced ABBA benchmark was conducted across 5 diverse prompt domains (English, Vietnamese, French, German, and Rust programming) measuring 100 trials on the target RTX 5060 Laptop GPU.

### Benchmark Configurations

- **Config A (Conservative Validated Baseline)**: `speculative_draft = 0`, `fused_rms_fp8 = false`, `fused_swiglu_fp8 = false`
- **Config B (Prior Unvalidated Default)**: `speculative_draft = 3`, `fused_rms_fp8 = true`, `fused_swiglu_fp8 = true`
- **Config C (Ablation: SwiGLU Off)**: `speculative_draft = 3`, `fused_rms_fp8 = true`, `fused_swiglu_fp8 = false`
- **Config D (Deterministic Fused RMS Only)**: `speculative_draft = 0`, `fused_rms_fp8 = true`, `fused_swiglu_fp8 = false`
- **Config E (Deterministic Both Fused)**: `speculative_draft = 0`, `fused_rms_fp8 = true`, `fused_swiglu_fp8 = true`

### Empirical Measurements Summary

| Configuration | Mean TPOT (ms) | p50 TPOT (ms) | p95 TPOT (ms) | Mean TTFT (ms) | Paired Speedup | Bitwise Parity vs Config A | Gating Decision |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| **Config A (Conservative Validated)** | 6.057 | 6.060 | 6.105 | 35.89 | **1.0000x** (ref) | **100.00%** (ref) | **ACCEPT (Production Baseline)** |
| **Config B (Prior Default: Spec3 + Both Fused)** | 5.681 | 5.805 | 6.082 | 35.55 | 1.0662x | **20.00% (FAIL)** | **REJECT AS DEFAULT** |
| **Config C (Spec3 + Fused RMS Only)** | 5.684 | 5.821 | 6.098 | 35.31 | 1.0656x | **20.00% (FAIL)** | **REJECT AS DEFAULT** |
| **Config D (Spec0 + Fused RMS Only)** | 6.023 | 6.018 | 6.073 | 36.45 | 1.0055x (+0.55%) | **100.00% (PASS)** | **REJECT AS DEFAULT** (<1.0% gain) |
| **Config E (Spec0 + Both Fused)** | 6.015 | 6.006 | 6.095 | 35.86 | 1.0069x (+0.69%) | **100.00% (PASS)** | **REJECT AS DEFAULT** (<1.0% gain) |

### Key Architectural Findings

1. **Speculative Decoding Parity Failure**:
   Under `speculative_draft = 3` (Configs B and C), 4 out of 5 evaluation prompts produced different token sequences compared to the deterministic golden reference (Config A). In LFM's hybrid Conv1d/SSM architecture, rolling back recurrent states upon token rejection introduces minor state drift that alters downstream autoregressive trajectories. Therefore, **`speculative_draft` must be `0` by default** in production.
2. **Fused SwiGLU Marginal Benefit**:
   Comparing Config B vs Config C reveals an incremental speedup of only **0.05%** ($5.681\text{ ms}$ vs $5.684\text{ ms}$). Similarly, under deterministic decode (Config E vs Config D), SwiGLU yields only **0.13%** ($6.015\text{ ms}$ vs $6.023\text{ ms}$). Both are well within random measurement variance and fail the required $\ge 1.0\%$ threshold. Default is frozen to `false`.
3. **Fused RMSNorm Marginal Benefit**:
   Standalone Fused RMSNorm-FP8 (Config D vs Config A) yields **+0.55%** throughput improvement. While maintaining 100.0% bitwise parity, it falls short of the mandatory 1.0% production gate. Default is frozen to `false`.

---

## Serving & Concurrency Evaluation

The production continuous serving engine was evaluated using `scripts/run_serving_evaluation.sh 8088`. All endpoints passed validation with zero errors.

### Continuous Batching Concurrency Scaling

| Concurrency ($C$) | Total Tokens | Wall Clock (s) | Aggregate TPS | Mean TTFT (ms) | Scaling Efficiency |
|:---:|---:|---:|---:|---:|:---:|
| **1** | 43 | 0.31 | **137.8 tok/s** | 17.92 ms | 1.00x |
| **2** | 82 | 0.34 | **242.6 tok/s** | 22.58 ms | 1.76x |
| **4** | 169 | 0.37 | **452.1 tok/s** | 29.49 ms | 3.28x |
| **8** | 330 | 0.40 | **817.2 tok/s** | 40.12 ms | **5.93x** |

### Radix Tree Prefix Caching Speedup

- **Turn 1 (Cold Prefill, ~600 tokens)**: TTFT = **181.36 ms**, Total Time = 311.23 ms.
- **Turn 2 (Prefix Cache Hit, ~630 tokens)**: TTFT = **29.27 ms**, Total Time = 136.94 ms.
- **Prefix Caching Speedup**: **6.20x TTFT acceleration** ($181.36\text{ ms} \to 29.27\text{ ms}$).

### Endpoint Compatibility & Protocol Matrix

- **OpenAI Compatibility**:
  - `GET /v1/models`: Correctly lists `LFM2.5-1.2B-Instruct` with `max_model_len: 32768`.
  - `POST /v1/chat/completions` (Non-streaming): Accurate token usage accounting and formatting.
  - `POST /v1/chat/completions` (Streaming SSE): Compliant `data: {...}\n\ndata: [DONE]\n\n` protocol.
  - `POST /v1/completions`: Text completion compatibility.
  - Structured Outputs: Strict JSON Schema (`response_format: {"type": "json_schema"}`) with automated markdown fence stripping and deterministic JSON compliance.
- **Ollama Compatibility**:
  - `GET /api/tags`: Drop-in discovery for Ollama CLI and OpenWebUI.
  - `GET /api/version`: Returns compatibility probe `{"version": "0.1.0"}`.
  - `POST /api/show`: Returns model architecture, template, and context length.
  - `POST /api/chat` (NDJSON): Streaming newline-delimited JSON chunks with duration timing statistics.
  - `POST /api/generate`: Native raw generation endpoint.

---

## Code Quality & Release Hygiene

All release gates pass with strict zero-tolerance standards:

1. **Rust Formatting**: `LLM_CUDA_ARCH=compute_120 cargo fmt --check` $\to$ **PASS** (zero diffs).
2. **Type Checking**: `LLM_CUDA_ARCH=compute_120 cargo check --all-features` $\to$ **PASS** in 0.90s.
3. **Linter Gate**: `LLM_CUDA_ARCH=compute_120 cargo clippy --all-targets --all-features -- -D warnings` $\to$ **PASS** with 0 warnings.
4. **Test Suite**: `LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1` $\to$ **120 passed; 0 failed; 24 ignored** in 67.51s.
5. **No Regressions / Safety**: Zero unsafe code additions without bounds verification, zero CPU polling loops, strict channel decoupling in HTTP server.

---

## Release Conclusion

The codebase at this milestone represents a robust, highly optimized, and empirically validated inference runtime. All experimental paths have been cleanly decoupled and made opt-in, guaranteeing deterministic, bitwise-reproducible, high-performance execution by default.
