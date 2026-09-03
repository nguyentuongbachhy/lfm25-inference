# Packed QKV decision — REJECT

## Decision

**REJECT for production integration.**

The packed-QKV direction produced a large local projection-group speedup, but the
benefit did not survive the full selected-E4M3 model at the primary B1 serving
shapes. The predefined stop condition therefore applies. No third packed-QKV
implementation is allowed on this branch.

This result does not invalidate packed GEMM as a primitive. It shows that, in the
current LFM2.5 decode runtime, replacing the three Q/K/V projections with one
packed projection plus the bounded downstream fusion does not improve the
end-to-end decode path enough to justify additional weights, scratch state, and
dispatch complexity.

## Primitive result

Attempt A compared three BF16 projections against one packed 3072-wide BF16
projection plus an unpack kernel.

| M | Direct mean | Packed mean | Mean speedup | NRMSE | Cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 48.074 us | 28.671 us | 1.6848x | 0.00000000 | 1.00000000 |
| 8 | 46.923 us | 21.947 us | 2.1576x | 0.00414138 | 0.99999142 |
| 16 | 50.617 us | 23.797 us | 2.1290x | 0.00414292 | 0.99999142 |

The M1 primitive gate of 1.10x passed by a wide margin.

## Attempt B

Attempt B removed the unpack launch from the normal paged-attention path. The
packed postprocess consumed `[Q|K|V]` directly, emitted rotated Q, and wrote K/V
directly to the paged cache.

No attention math, Split-K policy, KV precision, or model precision policy was
changed.

## Full-model ABBA result

Selected E4M3 policy, PS16, same-process D/P/P/D ordering:

| Shape | Direct mean | Packed mean | Mean speedup | Direct p95 | Packed p95 | p95 speedup | Top-1 match |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1/C128 | 6.021443 ms | 6.020888 ms | 1.0001x | 6.313312 ms | 6.739456 ms | 0.9368x | 95.8333% |
| B16/C128 | 6.173871 ms | 6.000840 ms | 1.0288x | 6.456672 ms | 6.130912 ms | 1.0531x | 93.4896% |
| B1/C2048 | 6.117581 ms | 6.125365 ms | 0.9987x | 6.171232 ms | 6.233152 ms | 0.9901x | 91.6667% |
| B8/C2048 | 8.618496 ms | 8.538956 ms | 1.0093x | 8.681408 ms | 8.646912 ms | 1.0040x | 95.8333% |
| B1/C8192 | 8.247029 ms | 8.013643 ms | 1.0291x | 8.865664 ms | 8.099968 ms | 1.0945x | 100.0000% |

First observed greedy divergences:

- B1/C128: step 9, slot 0, direct 1595, packed 779;
- B16/C128: step 0, slot 3, direct 14817, packed 1169;
- B1/C2048: step 8, slot 0, direct 1317, packed 779;
- B8/C2048: step 1, slot 0, direct 1587, packed 15434;
- B1/C8192: none.

## Gate evaluation

Predefined E2E continuation gates:

- B1/C128 mean speedup >= 1.02x: **FAIL** (`1.0001x`);
- B1/C128 p95 regression no worse than 1%: **FAIL** (`0.9368x`, about 6.7% slower p95);
- B1/C2048 mean speedup >= 1.01x: **FAIL** (`0.9987x`);
- larger-batch points: mixed, with B16/C128 positive and B8/C2048 near neutral;
- B1/C8192: positive, but this does not rescue the primary short/moderate-context gates.

The direction fails on performance alone. A separate model-quality evaluation is
therefore not required for the production decision. The observed greedy-trace
differences remain useful diagnostic evidence but are not the rejection reason.

## Why the primitive win disappears

The previous profile measured QKV projection at only about 306.954 us of a roughly
7.017 ms decode step. Even an infinitely fast QKV region had an Amdahl upper bound
of approximately 1.046x for the whole step.

The packed primitive reduces launch and local projection cost, but the complete
model remains dominated by MLP, convolution, attention, output projections,
normalization, sampling, and other runtime work. Additional packed-weight memory,
cuBLASLt algorithm changes, and downstream scheduling also reduce the amount of
the local win that reaches end-to-end latency.

The B1/C128 p95 regression is especially important for serving: mean neutrality is
not sufficient when tail latency becomes materially worse.

## Final status

- Attempt A primitive: **PASS**.
- Attempt B full-model: **FAIL**.
- Direction status: **REJECT**.
- Production integration: **NO**.
- Merge to main: **NO**.
- Reopen another packed-QKV local variant: **NO** under the current research policy.

The branch remains as a research record. Future work should start from the latest
validated production baseline and choose a different bottleneck/direction rather
than adding another packed-QKV variant.
