# Segmented & Ragged Tensor Core FlashAttention — Final Report

## Executive Summary

- **Direction**: Blackwell WMMA $16 \times 16 \times 16$ Tensor Core FlashAttention for Segmented & Ragged Multi-Sequence Prefill.
- **Target Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell SM120, `compute_120`, 8 GB VRAM, CUDA 12.8).
- **Status**: **PROMOTED & VERIFIED** (All primitive, numerical, and full-model E2E gates passed).
- **Outcome**:
  - Primitive speedup: **3.44x to 4.62x** across ragged batch configurations.
  - End-to-end full-model prompt speedup: **1.17x to 1.70x** ($B=2 \times 2048$ saves **~248.5 ms** per batch).
  - Numerical quality: Logit cosine $\ge 0.9997$, NRMSE $\le 0.022$, 100% top-1 argmax agreement, 0 NaN/Inf.
  - Continuous serving decode: 100% TPOT SLO pass, $B=16$ throughput sustained at 2108.9 tok/s.

---

## 1. Problem & Bottleneck Analysis

When multiple prefill requests arrive concurrently (e.g. concurrent user requests $B \ge 2$), `BatchModelCache` received ragged inputs with `segment_offsets.len() > 2`.
Previously, the runtime fell back to `hybrid_ragged_attention_lfm2_bf16`, which:
1. Executed a scalar-loop warp-shuffle attention kernel with $O(N^2)$ scalar math per head.
2. Completely bypassed the Blackwell SM120 Tensor Cores.
3. Created severe pipeline stalls and high latency (e.g. 602.2 ms for $2 \times 2048$ tokens).

---

## 2. Kernel Design & Architecture

We implemented `segmented_prefill_gqa_lfm2_bf16_flash` in `kernels/attention.cu`:
- **Tensor Core Tiling**: $16 \times 16 \times 16$ Blackwell WMMA instructions (`nvcuda::wmma`).
- **2D Execution Grid**: `dim3(max_q_tiles * NUM_QUERY_HEADS, num_segments)`.
- **Dynamic Segment Handling**: Each block unpacks its segment start offset $S_{\text{start}}$ and length $L_s$ from `segment_offsets_dev`. Blocks exceeding $L_s$ exit immediately with zero overhead.
- **Segment-Bounded Causal Masking**: Causal masking $r_k \le r_q$ and boundaries $k_{\text{start}} + t_k < L_s$ are evaluated entirely within the segment's coordinate frame, guaranteeing perfect isolation between batched sequences.
- **Online Softmax & Shared Memory Double Buffering**: Numerically stable streaming max/sum subtraction with shared memory tiles for Q ($16 \times 64$), K ($16 \times 64$), and V ($16 \times 64$).

---

## 3. Experimental Verification & Benchmarks

### 3.1 Primitive ABBA Benchmark Results (`bench_segmented_prefill_flash_bf16`)

| Shape ($B \times L$) | Total Tokens ($N$) | Legacy Mean (µs) | Flash Mean (µs) | Speedup Mean | Speedup p50 | Speedup p95 | Cosine | NRMSE | Non-finite |
|---|---|---|---|---|---|---|---|---|---|
| $2 \times 512$ | 1024 | 3,296.2 | 957.5 | **3.4426x** | 3.4426x | 3.4475x | 0.999997 | 0.002646 | 0 |
| $4 \times 512$ | 2048 | 6,377.1 | 1,680.9 | **3.7939x** | 3.7925x | 3.8052x | 0.999997 | 0.002646 | 0 |
| $2 \times 2048$ | 4096 | 51,314.9 | 11,109.5 | **4.6190x** | 4.6190x | 4.6288x | 0.999994 | 0.003540 | 0 |

*Primitive Gate (speedup $\ge 1.25\times$, NRMSE $\le 0.05$, cosine $\ge 0.9999$, 0 NaN/Inf): **PASS** across all shapes.*

---

### 3.2 Full-Model Production ABBA Benchmark (`bench_ragged_flash_production_abba`)

Conducted on real LFM2.5-1.2B checkpoint with selected FP8 precision policy active:

| Shape ($B \times L$) | Total Tokens ($N$) | Legacy Mean (ms) | Flash Mean (ms) | Speedup Mean | Speedup p50 | Speedup p95 | Logit Cosine | Logit NRMSE | Top-1 Match |
|---|---|---|---|---|---|---|---|---|---|
| $2 \times 512$ | 1024 | 97.15 ms | 82.84 ms | **1.1727x** | 1.1730x | 1.1864x | $\ge 0.999756$ | $\le 0.022457$ | 100% |
| $4 \times 512$ | 2048 | 182.86 ms | 154.81 ms | **1.1812x** | 1.1818x | 1.1899x | $\ge 0.999756$ | $\le 0.022457$ | 100% |
| $2 \times 2048$ | 4096 | 602.21 ms | 353.65 ms | **1.7029x** | 1.7043x | 1.7127x | $\ge 0.999903$ | $\le 0.014018$ | 100% |

*End-to-End Gate (speedup $\ge 1.10\times$, p95 speedup $\ge 1.05\times$, NRMSE $\le 0.05$, cosine $\ge 0.999$, top-1 match = true, 0 NaN/Inf): **PASS** across all shapes.*

---

### 3.3 Serving Benchmark Validation (`docs/serving/ragged-flash-serving-ps16.json`)

- **Serving Decode Throughput**:
  - $B=1$ ($ctx=16$): 130.6 tok/s (7.66 ms TPOT, 100% SLO pass)
  - $B=4$ ($ctx=16$): 542.6 tok/s (7.37 ms TPOT, 100% SLO pass)
  - $B=8$ ($ctx=16$): 1084.0 tok/s (7.38 ms TPOT, 100% SLO pass)
  - $B=16$ ($ctx=16$): 2108.9 tok/s (7.59 ms TPOT, 100% SLO pass)
- **Ragged Prefill Validation**:
  - $B=4 \times 32$: `final_logits_cosine_min = 0.999951`, `final_logits_nrmse_max = 0.010191`, `top1_agreement = true`.

---

## 4. Conclusion & Promotion

Segmented Blackwell WMMA Tensor Core FlashAttention has been verified and fully promoted into `src/model/lfm2_base.rs` for multi-sequence prefill batches.
