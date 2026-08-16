# Optimization status

Baseline work was measured on 2026-08-14 and the selective FP8 milestone on
2026-08-15 with an NVIDIA GeForce RTX 5060 Laptop GPU. Laptop clocks were not
locked, so FP8 speedups use the longer interleaved A/B protocol below.

## Implemented

- CUDA-event phase metrics plus CPU request/tokenization/detokenization timing.
- Engine-level KV page size; `/v1/completions` rejects request-level overrides.
- Uninitialized fully-overwritten outputs and a bounded BF16 temporary pool.
- Typed model weights with no hot-path string formatting or hash lookup.
- Zero-copy last-hidden-row input to the LM-head cuBLASLt matmul.
- Fused residual add + RMSNorm.
- Gate/Up weights packed at load and evaluated with one cuBLASLt matmul plus a
  packed SiLU-multiply kernel.
- XQA-like paged GQA decode: one block per KV head, four Q heads reuse K/V page
  tiles from shared memory.
- GPU block-table lookup for non-contiguous physical KV pages.
- Contiguous tiled prefill attention. Prefill still consumes fresh contiguous
  K/V and writes K/V to the paged cache only for future decode.
- Opt-in coarse/detailed decode profiler with warm-up and step limits. Profiling
  is disabled by default and records no CUDA events in the production path.
- Offline checkpoint-aware FP8 calibration, real-input GEMM error analysis,
  hidden/logit propagation, automatic selective-policy search, and independent
  held-out validation.
- Opt-in decode-only selective FP8. The chosen 16-site policy uses persistent
  E4M3 weights, pooled E4M3 activation buffers and pre-resolved typed dispatch.
  Prefill, KV, attention math, normalization, RoPE and residuals remain BF16.

## Selective FP8 result

The full record is in `docs/fp8_report.md`. On the independent WikiText-2 test
split (32 sequences, 5,664 next-token observations), the selected policy had
relative NLL delta +0.2154%, perplexity delta +0.4335, mean KL 0.01120,
top-1 agreement 92.96%, final RMSNorm NRMSE 0.08633/cosine 0.99627 and zero
non-finite logits.

The promotion workload used 128 fixed decode steps, PS16, two warm-up pairs and
20 same-process interleaved/order-balanced BF16--FP8 pairs. Mean TPOT changed
from 7.675 to 6.286 ms at 40 actual prompt tokens, a paired mean speedup of
1.221x. Every measured decode arm had zero BF16 and FP8 pool misses after
warm-up. Because prefill remains BF16, TTFT was intentionally flat: 8.811
versus 8.848 ms.

Speedup decreases with context as paged attention/KV traffic becomes a larger
part of TPOT:

| Actual context | BF16 mean TPOT | FP8 mean TPOT | Paired mean speedup |
|---:|---:|---:|---:|
| 40 | 7.675 ms | 6.286 ms | 1.221x |
| 138 | 8.015 ms | 6.638 ms | 1.207x |
| 516 | 8.373 ms | 7.016 ms | 1.193x |
| 2,056 | 10.902 ms | 9.419 ms | 1.157x |
| 8,202 | 22.023 ms | 20.282 ms | 1.086x |

The verdict is `PROMOTE` for the measured short-context, batch-1, decode-only
scope. BF16 remains the default and golden fallback; the policy is an explicit
engine startup choice because calibration and speedup are checkpoint- and
GPU-specific.

## Historical BF16 E2E sample

Workload: 13 prompt tokens, 32 greedy completion tokens, page size 16, five
serial warm requests after one cold request.

- Correct output and usage: 13 + 32 = 45 tokens.
- Warm TTFT median: about 10.3 ms.
- Warm TPOT median: about 9.0 ms.
- Warm total latency range: about 275-307 ms.
- BF16 pool after warm-up: 5792 hits and 0 misses per request.

This is a short-context, batch-1 measurement. It is not a throughput or
long-context result.

## Historical BF16 decode profile

Workload: batch 1, 30 prompt tokens, 128 greedy completion tokens. Eight decode
steps were skipped as warm-up and the next 100 steps were aggregated. The
checkpoint is 2,340,697,936 bytes, so the coarse envelope corresponds to about
267 GB/s of effective checkpoint-weight bandwidth. This is an approximation:
not every checkpoint byte has identical runtime traffic.

Coarse mode:

| Component | Mean per token | Envelope share |
|---|---:|---:|
| MLP total | 5.014 ms | 57.10% |
| Conv total | 1.157 ms | 13.18% |
| LM head | 0.961 ms | 10.95% |
| Attention total | 0.772 ms | 8.79% |
| Residual/RMSNorm | 0.351 ms | 4.00% |
| Sampling | 0.197 ms | 2.25% |
| Other CUDA | 0.329 ms | 3.75% |

The coarse GPU envelope was 8.781 ms/token; measured regions accounted for
8.452 ms/token. `other_cuda` also contains unclassified GPU work, so 0.329 ms
is an upper bound—not an expected CUDA Graph speedup.

Detailed mode:

| Component | Mean per token |
|---|---:|
| MLP Gate/Up GEMM | 3.415 ms |
| MLP Down GEMM | 1.801 ms |
| MLP SiLU | 0.075 ms |
| Conv input projection | 1.018 ms |
| Conv output projection | 0.442 ms |
| ShortConv body | 0.049 ms |
| Attention QKV projections | 0.396 ms |
| Attention postprocess | 0.171 ms |
| XQA | 0.181 ms |
| Attention output projection | 0.195 ms |
| LM head | 1.007 ms |

Detailed instrumentation increased the measured envelope to 9.595 ms/token,
so use it to rank components, not as the uninstrumented TPOT baseline. GEMMs
accounted for about 8.27 ms/token in this run. The current evidence therefore
does not justify ShortConv fusion or more short-context XQA tuning. Large TPOT
improvements require lower-precision weights, weight reuse through batching,
or multi-token/speculative execution; BF16 remains the reference path.

## Attention microbenchmarks

Paged XQA decode, batch 1:

| Context | PS16 mean | PS32 mean |
|---:|---:|---:|
| 16 | 9.65 us | 13.10 us |
| 32 | 11.85 us | 13.49 us |
| 128 | 29.44 us | 37.83 us |
| 512 | 105.84 us | 104.85 us |
| 2048 | 412.28 us | 393.83 us |

Tiled contiguous prefill attention, one layer:

| Tokens | Mean |
|---:|---:|
| 16 | 15.39 us |
| 128 | 107.42 us |
| 512 | 1.67 ms |
| 1024 | 5.47 ms |

Run the benchmark commands in `docs/command.md` on the deployment GPU before
selecting PS16 versus PS32. Fragmentation and scheduler-level concurrency are
not represented by the kernel-only table.

## Metric interpretation

`first_token_gpu_wait_and_sampling_ms` and
`gpu_wait_and_sampling_total_ms` deliberately include the wait for queued GPU
forward work. The D2H token transfer is the synchronization boundary, so these
fields must not be interpreted as pure sampler compute time.
