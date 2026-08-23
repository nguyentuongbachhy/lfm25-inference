# LFM2.5-1.2B Inference Runtime — Final Validated Release Report

## Status

**FINAL VALIDATED MILESTONE — READY TO FREEZE**

This report closes the current optimization campaign for `LFM2.5-1.2B-Instruct` on the measured NVIDIA GeForce RTX 5060 Laptop GPU / SM120 target.

The runtime keeps BF16 as the golden reference and fallback, BF16 prefill, selective tensor-wide E4M3 decode, the validated serving/runtime optimizations already merged into `main`, and the scratchless atomic greedy argmax production dispatch for the validated batch domain.

No rejected NVFP4 production code is present in `main`.

## Target environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Architecture | Blackwell GeForce SM120 / `compute_120` |
| VRAM | 8 GB class |
| CUDA compiler | 12.8.93 |
| Checkpoint | `LFM2.5-1.2B-Instruct` |
| Primary storage/output dtype | BF16 |
| Decode low precision | selective E4M3 |
| KV dtype | BF16 |
| Initial/prefill path | BF16 |

## Final production precision policy

The frozen decode policy is `policy_frontier_16` with 16 E4M3 sites:

```text
Gate/Up: layers 2, 3, 5, 7, 8, 9, 11, 15
Down:    layers 6, 8, 9, 10, 12, 14, 15
LM head
```

Conv and attention projections remain BF16. BF16 weights remain resident for prefill, reference execution and fallback. The persistent E4M3 copies occupy approximately 496 MiB according to the implementation-derived accounting in `docs/fp8_report.md`.

## Final independent quality rerun

The final quality rerun used the untouched WikiText-2 test split with SHA-256:

```text
d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0
```

Execution scope:

```text
policy: policy_frontier_16
sequences: 32
source tokens: 5,696
teacher-forced next-token observations: 5,664
history: identical BF16/candidate teacher-forced tokens
```

Final measured metrics:

| Metric | BF16 / E4M3 result |
|---|---:|
| Mean NLL | 3.930884 / 3.939768 |
| Absolute NLL delta | +0.008884 |
| Relative NLL delta | +0.2260% |
| Perplexity | 50.9520 / 51.4067 |
| Perplexity delta | +0.4547 |
| Mean KL | 0.011059 |
| KL p50 | 0.007947 |
| KL p95 | 0.030445 |
| KL p99 | 0.064480 |
| Mean logit cosine | 0.997995 |
| Top-1 agreement | 92.97% |
| Mean top-5 overlap | 93.73% |
| Mean top-10 overlap | 94.01% |
| Final RMSNorm NRMSE | 0.087295 |
| Final RMSNorm cosine | 0.996187 |
| Final RMSNorm RMS ratio | 0.999184 |
| Non-finite logits | 0 |
| Non-finite hidden values | 0 |
| Quality gate | **PASS** |

The hard gate is:

```text
relative NLL delta <= 1%
no non-finite values
final hidden cosine >= 0.99
final hidden NRMSE <= 0.10
```

The final rerun therefore independently confirms the frozen selective E4M3 policy remains inside the established quality envelope.

The greedy generation diagnostic remains secondary evidence rather than the hard gate. Across eight fixed temperature-0 prompts, exact sequence agreement is 62.5% while output-length agreement is 100%. Autoregressive divergence is expected to amplify small ranking changes, so teacher-forced NLL/KL and hidden-state propagation remain the primary promotion criteria.

## Reproducibility against the previous independent validation

The final rerun is materially consistent with the earlier frozen-policy test result recorded in `docs/fp8_report.md`:

```text
Previous relative NLL delta: +0.2154%
Final rerun relative NLL:     +0.2260%

Previous mean KL:             0.01120
Final rerun mean KL:          0.01106

Previous logit cosine:        0.998010
Final rerun logit cosine:     0.997995

Previous top-1 agreement:     92.96%
Final rerun top-1 agreement:  92.97%

Previous final hidden NRMSE:  0.08633
Final rerun hidden NRMSE:     0.08729
```

These differences do not change any gate decision and provide a useful final reproducibility check on the current `main` runtime.

## Final E4M3 decode performance

The validated same-process, order-balanced BF16/E4M3 benchmark uses PS16, 128 fixed decode steps, two warm-up pairs and 20 measured pairs per context.

| Context | BF16 TPOT mean | E4M3 TPOT mean | Paired mean speedup |
|---:|---:|---:|---:|
| 40 | 7.675 ms | 6.286 ms | 1.221x |
| 138 | 8.015 ms | 6.638 ms | 1.207x |
| 516 | 8.373 ms | 7.016 ms | 1.193x |
| 2,056 | 10.902 ms | 9.419 ms | 1.157x |
| 8,202 | 22.023 ms | 20.282 ms | 1.086x |

Short-context gain is concentrated in selected weight GEMMs. As context grows, paged attention/KV traffic becomes a larger fraction of decode time, so the E4M3 speedup falls according to Amdahl's law.

The post-FP8 detailed profile recorded approximately:

```text
Decode GPU envelope: 7.060 ms
MLP Gate/Up:          2.376 ms
MLP Down:             1.370 ms
MLP total:            3.828 ms
Conv total:           1.135 ms
Attention total:      0.815 ms
LM head:              0.379 ms
```

These numbers are profiling evidence for bottleneck ranking; the interleaved E2E benchmark above is the performance promotion evidence.

## Runtime v2 atomic argmax

The final production runtime also includes the validated scratchless atomic greedy argmax path.

Production dispatch is deliberately bounded:

```text
B <= 16 and vocab <= 65536 -> atomic argmax
otherwise                  -> legacy argmax
```

Full-model ABBA measurements on the selected E4M3 policy showed consistent positive mean results in the production B<=16 domain, with complete sampled-trace top-1 agreement. Representative B1 results include approximately 1.0126x speedup at context 128, 1.0121x at context 512, and 1.0089x at context 2048. B32 was not a stable win and B64 regressed, so neither is routed to the atomic path.

See `docs/benchmarks/validated-runtime-v2.md` for the complete matrix and exact legacy tie-semantics contract.

## Final promoted runtime components

The current validated runtime includes the production paths accumulated through the campaign, including:

- BF16 reference/fallback execution;
- paged KV cache with PS16 production policy;
- paged GQA/XQA-like decode path;
- tiled contiguous prefill attention;
- persistent decode/runtime buffers;
- fused residual/RMSNorm paths;
- packed Gate/Up execution;
- selective persistent E4M3 decode weights;
- bounded FP8 activation scratch/pooling;
- fused SwiGLU-to-E4M3 path for selected down projections;
- measured Split-K attention dispatch/workspace where validated;
- zero-copy last-hidden-row LM-head path;
- scratchless atomic greedy argmax for the validated production domain.

Production changes were promoted only after numerical, model-level and E2E validation.

## Rejected directions

Rejected research is intentionally not part of the production runtime.

### W8A8

Strong local integer-kernel speedups did not preserve model decision quality. Rejected.

### W8A16

Some local/full-decode performance wins were observed, but sampled-token agreement failed. Rejected.

### Custom tiny-M BF16 GEMM

The custom kernels did not exceed cuBLASLt sufficiently to justify production complexity. The vendor path was already close to the available memory-bandwidth roofline for the measured shapes. Rejected.

### MXFP8 block-32

Numerically promising in a small capability test but materially slower in the measured GEMM/E2E research path. Rejected for this target/runtime.

### NVFP4 / SM120

NVFP4 showed strong primitive performance, including roughly 1.5x-class cold GEMM improvement over the E4M3 comparison for key tiny-M shapes, but model-quality propagation became the limiting factor.

Research progression:

```text
Phase 0/0.5  primitive performance          PASS
Phase 1      persistent W4A4 path           PASS
Phase 1b     synthetic numerical sanity     PASS as sanity only
Phase 2A     real-checkpoint local screen   FILTER
Phase 2B     single-site propagation        narrowed to Gate/Up 8 + 9
Phase 2C     disjoint test confirmation     REJECT
```

On the disjoint Phase-2C confirmation, Gate/Up layer 8, layer 9 and the combined 8+9 policy all failed the hidden-state propagation gate. No site survived the bounded confirmation frontier, so no in-process or production NVFP4 backend was justified.

The complete rejection record is `docs/research/nvfp4_rejection.md`. The research branch may remain archived for evidence, but it is not merged into the production runtime.

## Final decision

**Freeze the current runtime as the final validated milestone of this optimization campaign.**

There is no remaining quality or correctness blocker identified by the completed gates. Additional work such as lower-precision KV cache, further attention strategies, CUDA Graphs, continuous batching, prefix KV reuse or speculative decoding should be treated as a new optimization campaign starting from this frozen baseline, not as unfinished work required to validate this release.

The release claim is deliberately scoped:

> This repository contains a validated custom LFM2.5-1.2B inference runtime for the measured RTX 5060 Laptop / SM120 environment, with BF16 as the golden path and a selectively optimized E4M3 decode policy whose model-quality and latency trade-off has been independently measured.

It does not claim that all kernels are globally optimal, that the precision policy transfers to another checkpoint/GPU without revalidation, or that no future optimization can improve the runtime.

## Recommended freeze point

After this report is merged, the repository can be frozen/tagged as the final version of the current campaign. Future optimization work should branch from this validated point and use the same promotion/rejection discipline.
