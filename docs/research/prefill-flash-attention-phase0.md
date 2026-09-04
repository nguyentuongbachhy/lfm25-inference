# Tiled FlashAttention contiguous prefill attention — Phase 0

## Baseline

main SHA: 8377b91a061d0b45d1b6ce8f0f496907828b20c8
branch: agent/prefill-flash-attention

Previous investigations established that:
- Q4 query reuse (halving repeated K/V tile loads) yielded only ~1.045x mean speedup at 2K and 8K.
- Fast-exp scalar replacement (`__expf`) yielded only 1.0071x at 2K and 1.0397x at 8K.
- Both proved that contiguous prefill is neither global K/V bandwidth-bound nor transcendental math-bound.

## Hypothesis

Contiguous prefill attention is bound by warp execution serialization and scalar CUDA core compute:
1. The production Q2 kernel evaluates Q-K dot products scalar-wise across 32 threads, incurring 5 `shfl.sync.down` and 1 `shfl.sync.idx` broadcast per visited key.
2. For an 8K prompt, this executes ~24,500 warp shuffles per query token, stalling the warp scheduler on warp communication latency.
3. The current kernel operates at only ~767 GFLOP/s on Blackwell SM120, which is <1% of the GPU's hardware Tensor Core capacity.

A FlashAttention-style tiled formulation using hardware Tensor Cores (`nvcuda::wmma` $16 \times 16 \times 16$) or block-tiled GEMM computes entire tiles of Q-K and P-V matrix products without warp shuffle serialization, unlocking hardware matrix-multiply throughput.

## Evidence

- Single-layer prefill attention latency at $N=8192$ is ~357 ms. Across 16 model layers, attention alone takes ~5.7 s, explaining why full-model E4M3 speedup collapses from 1.23x at prompt 40 to 1.05x at prompt 8202.
- SASS/PTX analysis confirms 6 warp shuffles and scalar FP32 operations per key inside the inner loop.

## Amdahl analysis

- At $N \approx 8202$, attention represents $>75\%$ of the prefill execution time ($f \ge 0.75$).
- With an expected local speedup $S_{local} \ge 1.50\times$:
  $$S_{total} \ge \frac{1}{(1 - 0.75) + 0.75 / 1.50} = \frac{1}{0.25 + 0.50} = 1.33\times$$
- At $N \approx 2048$, attention represents $\approx 40\%$ of prefill time ($f \approx 0.40$):
  $$S_{total} \ge \frac{1}{(1 - 0.40) + 0.40 / 1.25} = \frac{1}{0.60 + 0.32} = 1.087\times$$

## Implementation

What changes:
- Research-only CUDA module: `kernels/attention_prefill_flash.cu`.
- Tiled QK score computation and PV accumulation using Tensor Cores (`nvcuda::wmma`) or tiled warp matrix blocks.
- Block sizes: $Q$-tile = 16 tokens, $K$-tile = 16 tokens, $D = 64$.
- Online softmax with tiled row-max and row-sum tracking.

What does NOT change:
- GQA configuration: 32 Q heads, 8 KV heads, head dimension 64, GQA ratio 4:1.
- Attention scale: $\frac{1}{\sqrt{64}} = 0.125$.
- Causal masking: token $i$ only attends to keys $j \le i$.
- BF16 input and output layout.
- Production kernel dispatch remains completely unchanged.

## Primitive benchmark

- Reference: Production Q2 contiguous prefill kernel (`prefill_gqa_lfm2_bf16`).
- Candidate: Research FlashAttention tiled prefill kernel (`prefill_gqa_lfm2_bf16_flash`).
- Benchmark method: Same-process balanced AB/BA GPU timing (`benchmark_gpu_paired`).
- Token counts: $N = 512, 2048, 8192$.
- Dtype: BF16.

## Numerical gate

- Compare complete BF16 output against the production Q2 kernel.
- Fast-exp tolerance: $| \text{candidate} - \text{reference} | \le 0.035 + 0.025 \times | \text{reference} |$.
- Cosine similarity $\ge 0.999$.
- NRMSE $\le 0.05$.
- Zero non-finite outputs (no NaN, no Inf).

## Performance gate

- $N=512$: mean regression $\le 5\%$.
- $N=2048$: mean speedup $\ge 1.15\times$, no material p95 regression.
- $N=8192$: mean speedup $\ge 1.25\times$, no material p95 regression.

## Model-quality gate (Phase 1)

Only after primitive gate passes:
- Teacher-forced relative NLL increase $\le 1\%$.
- Hidden cosine $\ge 0.99$.
- Hidden NRMSE $\le 0.10$.
- Zero non-finite activations or logits.

## E2E gate (Phase 1)

- Primary prompt lengths: ~516, ~2056, ~8202 tokens.
- Demonstrable prompt TTFT speedup at 2K and 8K.

## Stop condition

If the candidate cannot satisfy the numerical gate or fails the $1.15\times$ (2K) / $1.25\times$ (8K) mean speedup gate, stop and reject. Do not proceed to model integration without meeting both gates.

## Iteration budget

Maximum 2 materially different implementations:
1. WMMA Tensor Core tiled FlashAttention.
2. Vectorized warp-tiled attention without WMMA if fragment transpose overhead prevents numerical parity.

## Phase 0 result

RTX 5060 Laptop GPU, SM120, same-process balanced AB/BA timing:

| Tokens | Precise mean | Fast mean | Mean speedup | Precise p50 | Fast p50 | Precise p95 | Fast p95 | Cosine | NRMSE | Tolerance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 1536.531 us | 428.538 us | 3.5858x | 1540.320 us | 427.840 us | 1546.560 us | 435.392 us | 0.99999654 | 0.00264553 | pass |
| 2048 | 21375.840 us | 5526.720 us | 3.8681x | 21473.761 us | 5555.872 us | 21489.441 us | 5581.024 us | 0.99999418 | 0.00354003 | pass |
| 8192 | 357400.571 us | 84594.830 us | 4.2257x | 358118.378 us | 84592.705 us | 361581.116 us | 88242.371 us | 0.99999229 | 0.00422397 | pass |

Numerical tolerance:
- Fast-exp elementwise tolerance passes across all lengths (`within_tolerance=true`).
- 0 non-finite values across all lengths.
- Cosine similarity $> 0.99999$ across all lengths (exceeds $\ge 0.999$ gate).
- NRMSE $< 0.0045$ across all lengths (exceeds $\le 0.05$ gate).

Performance evaluation:
- $N=512$: measured 3.5858x mean speedup (gate $\le 5\%$ regression: PASS).
- $N=2048$: required $\ge 1.15\times$, measured 3.8681x (PASS).
- $N=8192$: required $\ge 1.25\times$, measured 4.2257x (PASS).
- p95 speedups: 3.6443x at 512, 3.9889x at 2048, 4.2941x at 8192 (no p95 regression: PASS).

## Phase 0 Verdict

**PASS ALL PRIMITIVE AND NUMERICAL GATES.**

Single-layer prefill attention latency at $N=8192$ dropped from 357.4 ms to 84.6 ms ($4.23\times$ speedup).
Across 16 layers, this represents an estimated savings of 4.36 seconds per 8K prefill.
The direction is approved to proceed to Phase 1 (model integration, quality validation, and prompt E2E benchmarking).

