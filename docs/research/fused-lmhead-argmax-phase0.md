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

The values and scale do not affect kernel geometry or memory traffic, so model loading is intentionally excluded from this primitive feasibility screen.

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

Verify the complete path returns a stable token.

## Invalid initial sequential run

The first sequential Phase-0 run reported:

- quantize mean 11.832 us;
- isolated LM-head mean 393.195 us;
- argmax mean 28.898 us;
- complete boundary mean 382.337 us.

The complete boundary cannot be faster than the identical LM-head projection it contains when measured under comparable GPU state. The negative `removable_us=-10.859` and `fusion_ceiling=0.9724x` therefore identify measurement-order / power-state drift, not a real architectural result.

That run is invalid for promotion or rejection. The branch now requires paired boundary-versus-LM-head timing.

## Feasibility metric

Define from the paired means:

```text
fusion_ceiling = paired_boundary_mean / paired_lm_head_mean
removable_us   = paired_boundary_mean - paired_lm_head_mean
```

This is an optimistic ceiling that assumes a fused kernel can make quantization, logits materialization and argmax free while matching the current cuBLASLt projection time. Real fused performance will be lower.

## Phase 0 continuation gate

Continue to a custom fused FP8 LM-head/argmax prototype only if all are true:

- the benchmark returns a deterministic token;
- paired `fusion_ceiling >= 1.08x`;
- paired non-GEMM removable mean latency is at least 30 us;
- the paired complete boundary is at least 5% of the current B1 decode step, so the E2E ceiling is material.

If the paired optimistic ceiling is below 1.08x, reject the fused-LM-head direction before custom Tensor Core implementation.

## Later gate if Phase 0 passes

A custom fused kernel must compare against the complete current boundary, not against argmax alone.

It must preserve the exact greedy token for the deterministic primitive test and then pass the existing teacher-forced model-quality gate before production consideration.

E2E promotion requires at least 1.01x B1 TPOT improvement at C128 and C2048 with no material p95 or batch regression.
