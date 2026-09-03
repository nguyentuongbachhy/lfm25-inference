# CUDA Graph decision — PARTIAL / bounded promotion

## Decision

**PARTIAL: promote only measured-safe B1 graph regions.**

CUDA Graph replay gives real end-to-end decode gains in selected B1 regimes, but it is not a universal policy. The production-dispatch gate confirms that the Split-K=4 region around C512 is not safe for graph replay because tail latency regresses, while the unsplit and Split-K=8 regions retain useful gains.

Production policy must therefore keep the existing direct decode path as the fallback and enable graph replay only for measured-safe topology/context regions.

## Production-dispatch results

Selected E4M3 policy, PS16, actual `try_forward_ragged_decode -> bounded graph cache` path:

| Shape | Direct mean | Graph mean | Mean speedup | Direct p95 | Graph p95 | p95 speedup | Submit speedup | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1/C128 | 6.087884 ms | 5.732306 ms | 1.0620x | 6.220928 ms | 5.784608 ms | 1.0754x | 11.5848x | exact |
| B1/C512 | 5.956612 ms | 5.902548 ms | 1.0092x | 5.979840 ms | 6.176832 ms | 0.9681x | 12.3612x | exact |
| B1/C1024 | 6.017053 ms | 5.859777 ms | 1.0268x | 6.076352 ms | 5.937632 ms | 1.0234x | 13.5270x | exact |
| B1/C2048 | 6.146545 ms | 5.908258 ms | 1.0403x | 6.194592 ms | 5.990080 ms | 1.0341x | 13.5554x | exact |

All measured graph paths preserve exact sampled-token traces.

## Gate evaluation

### Unsplit region

B1/C128 passes both mean and p95 gates:

- mean speedup: **1.0620x**;
- p95 speedup: **1.0754x**;
- host submission speedup: **11.5848x**;
- exact token agreement: **PASS**.

Status: **PROMOTE as a bounded graph bucket**.

### Split-K=4 region

B1/C512 does not justify graph replay:

- mean speedup: only **1.0092x**;
- p95 speedup: **0.9681x**, approximately 3.2% slower;
- host submission is much cheaper, but that does not compensate for the GPU/tail-latency regression.

Status: **REJECT graph replay for Split-K=4; use direct decode**.

### Split-K=8 region

B1/C1024 and B1/C2048 both pass:

- C1024 mean: **1.0268x**, p95: **1.0234x**;
- C2048 mean: **1.0403x**, p95: **1.0341x**;
- submission overhead falls by more than 13x;
- exact token agreement remains true.

The earlier research sweep also measured B1/C4096 at 1.1688x mean and 1.2600x p95, while B1/C8192 regressed to about 0.9647x. This supports a bounded moderate-long-context Split-K=8 policy rather than an unlimited long-context graph policy.

Status: **PROMOTE bounded Split-K=8 graph replay; keep very-long-context direct fallback**.

## Final production policy

Required policy after this decision:

1. CUDA Graphs remain opt-in until merged into the validated production baseline.
2. PS16 only.
3. B1 only.
4. Unsplit graph bucket: enabled in the measured-safe short/moderate region.
5. Split-K=4 graph bucket: **disabled**; direct decode is mandatory.
6. Split-K=8 graph bucket: enabled only in the measured-safe moderate-long region; very long context remains direct.
7. B>1 remains direct.
8. Any unsupported topology remains direct.
9. Existing selected-E4M3 model-quality policy is unchanged.

## Why the result matters

The primitive graph probe showed a 1.2464x GPU speedup and more than 33x lower host submission cost. The full-model results show that only part of this launch-overhead win survives after GPU scheduling and memory behavior are included.

This confirms two points:

- CUDA launch/submission overhead is material in B1 decode;
- topology/context-dependent GPU execution can still make graph replay neutral or negative, so graph dispatch must be performance-gated rather than enabled universally.

## Direction status

- Primitive compatibility probe: **PASS**.
- Full-model research capture: **PASS in bounded B1 regions**.
- Actual production-dispatch gate: **PASS for Unsplit and Split-K=8; FAIL for Split-K=4**.
- Numerical correctness: **PASS; exact token traces at all measured production-dispatch points**.
- Universal CUDA Graph policy: **REJECT**.
- Bounded CUDA Graph policy: **PROMOTE**.
- Split-K=4 graph cache: **DO NOT PROMOTE**.

The next optimization direction must start from the eventual validated baseline that includes only these bounded graph wins. It must not reopen packed QKV, FP8 KV, custom tiny-M BF16 GEMM, or other previously rejected directions.