# Fused FP8 LM head + greedy argmax — Phase 0

## Baseline

This direction starts from current `main` after the bounded CUDA Graph long-context extension was promoted.

Closed directions remain closed. In particular, this experiment does not reopen the general custom tiny-M BF16 GEMM direction or cuBLASLt autotuning.

## Motivation

The production B1 decode path currently performs the final serving boundary as:

```text
final normalized BF16 hidden [1,2048]
    -> E4M3 activation quantization
    -> cuBLASLt E4M3 LM-head projection [1,65536]
    -> BF16 logits [65536]
    -> scratchless atomic greedy argmax
    -> token id
```

A specialized fused serving kernel could in principle avoid materializing the full 65,536-element BF16 logits row and avoid the separate argmax pass. This is a fixed-shape serving problem rather than a general GEMM replacement.

Before implementing custom FP8 Tensor Core math, measure how much latency remains outside the vendor LM-head projection after the already-promoted atomic argmax work.

## Phase 0 benchmark

Use the exact production shape and dtypes with deterministic nonzero synthetic E4M3 values:

- M=1
- K=2048
- N=65536
- tensor-wide E4M3 input and weight
- BF16 logits
- production scratchless atomic argmax

The decision measurement is a balanced paired AB/BA comparison between:

Reference:

```text
quantize -> cuBLASLt LM head -> production argmax
```

Optimistic lower bound:

```text
prequantized input -> same cuBLASLt LM head only
```

Both paths use separate output buffers but the same stream, weight, shape, dtype and cuBLASLt plan. The AB/BA order reduces laptop boost-clock, power and thermal bias.

Quantization-only and argmax-only timing remain diagnostic outputs. They are not used to derive the continuation decision independently.

## Invalid initial sequential run

The first sequential Phase-0 run reported:

- quantize mean 11.832 us;
- isolated LM-head mean 393.195 us;
- argmax mean 28.898 us;
- complete boundary mean 382.337 us.

The complete boundary cannot be faster than the identical LM-head projection it contains when measured under comparable GPU state. The negative `removable_us=-10.859` and `fusion_ceiling=0.9724x` therefore identify measurement-order / power-state drift, not a real architectural result.

That run is invalid for promotion or rejection.

## Valid paired result

RTX 5060 Laptop GPU, balanced same-process AB/BA:

| Metric | Result |
|---|---:|
| complete boundary mean | 408.086 us |
| complete boundary p95 | 411.725 us |
| LM-head-only mean | 394.454 us |
| LM-head-only p95 | 398.547 us |
| optimistic fusion ceiling | 1.0346x |
| paired mean ratio | 1.0347x |
| paired p50 ratio | 1.0344x |
| paired p95 ratio | 1.0483x |
| removable mean latency | 13.632 us |
| quantize mean, diagnostic | 9.001 us |
| argmax mean, diagnostic | 33.182 us |
| complete boundary / 6 ms step | 6.80% |
| deterministic token | 0 |

The paired result is internally consistent. The complete boundary is slower than the LM-head-only lower bound, but only by 13.632 us on mean.

## Decision

**REJECT** the fused FP8 LM-head + greedy argmax direction.

The predefined continuation gate required both:

- optimistic `fusion_ceiling >= 1.08x`;
- removable mean latency >= 30 us.

Measured values are only `1.0346x` and `13.632 us`. Both fail by a large margin.

Although the complete boundary is about 6.8% of a 6 ms B1 step, approximately 96.7% of that boundary is already the vendor FP8 LM-head projection itself. Even a hypothetical fusion that makes activation quantization, logits materialization and greedy reduction free while preserving cuBLASLt projection time has only about a 3.35% local ceiling. The practical gain would be smaller because a custom fused projection would also need to match cuBLASLt throughput.

Do not implement a custom Tensor Core LM-head GEMV/argmax kernel from this direction. Do not reopen it unless the LM-head implementation or serving semantics change materially.

## Stop condition

Phase 0 ends this direction. No production code is changed and this branch is not merged.
