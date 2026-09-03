# CUDA Graph Phase 2 — bounded-dispatch result

## Broad full-model sweep

All rows use the selected weight-E4M3 policy, PS16 KV cache, deterministic forced-token history, fresh prefill per pass, and complete-pass ABBA ordering. Every measured row preserved exact sampled-token agreement between direct execution and CUDA Graph replay.

| Shape | Direct mean | Graph mean | Mean speedup | Direct p95 | Graph p95 | P95 speedup | Submit speedup | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1/C128 | 6.115702 ms | 5.699269 ms | 1.0731x | 6.250752 ms | 5.773824 ms | 1.0826x | 14.7186x | PASS |
| B16/C128 | 6.099422 ms | 6.012530 ms | 1.0145x | 6.161056 ms | 6.097184 ms | 1.0105x | 17.7921x | NEUTRAL |
| B1/C2048 | 6.159550 ms | 5.901027 ms | 1.0438x | 6.419392 ms | 6.011104 ms | 1.0679x | 11.2184x | PASS |
| B8/C2048 | 8.611586 ms | 8.530084 ms | 1.0096x | 8.655424 ms | 8.739424 ms | 0.9904x | 13.4237x | REJECT FOR GRAPH |
| B1/C8192 | 8.135903 ms | 8.433921 ms | 0.9647x | 8.139040 ms | 8.437984 ms | 0.9646x | 16.0845x | REJECT FOR GRAPH |

`top1_agreement=true` for every row.

## Decision

A universal CUDA Graph decode policy is rejected.

CUDA Graph replay remains a viable bounded optimization for single-request decode at moderate context. B1/C128 and B1/C2048 exceed the predefined full-model gates, while high-batch regimes are neutral and very-long-context B1 regresses materially.

Host submission improves by roughly 10x to 18x across all measured shapes, but submission reduction alone is not sufficient for promotion. At large GPU workloads the graph can still lose on GPU execution time, as shown by B1/C8192.

## Required bounded sweep

Before production graph-cache integration, measure the remaining B1 topology and boundary points only:

- B1/C512 — PS16 Split-K=4 bucket;
- B1/C1024 — start of PS16 Split-K=8 bucket;
- B1/C4096 — determines whether the B1 upper bound can extend beyond C2048.

The already-measured B1/C128, B1/C2048, and B1/C8192 points do not need to be repeated.

The benchmark harness on `agent/cuda-graphs` now contains only these three unknown points.

## Production rule after the bounded sweep

The production candidate must be opt-in and fail closed to the existing direct path.

Promote only context ranges with both:

1. exact sampled-token agreement; and
2. at least 1.02x mean full-model speedup with no material p95 regression.

Do not enable graphs for B>1 from the current evidence. Do not enable graphs at B1/C8192. Keep graph entries separated by attention topology because PS16 B1 dispatch changes from unsplit to Split-K=4 at C512 and to Split-K=8 at C1024.

If C4096 is neutral or negative, cap the graph policy at the highest validated positive B1 context. If C512 or C1024 fails, exclude that topology bucket rather than forcing one continuous context interval.
