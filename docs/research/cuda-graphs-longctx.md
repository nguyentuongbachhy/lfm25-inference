# CUDA Graph long-context crossover

## Baseline

This branch starts from current `main` at `a25eacb37a41f32c2d2080bc0be47604dedd24a8`.

The existing production graph policy is already validated for B1/PS16 in selected short and moderate context ranges. This experiment does not change Split-K policy or reopen rejected Split-K tuning.

## Observation

Prior full-model ABBA results showed:

- B1/C4096: graph mean speedup 1.1688x, p95 speedup 1.2600x, top1 exact;
- B1/C8192: graph mean speedup 0.9647x, p95 speedup 0.9646x, top1 exact.

Production currently routes `C >= 4096` to direct execution, so the positive C4096 point is not used.

## Coarse long-context sweep

RTX 5060 Laptop GPU, selected E4M3 policy, PS16, B1, Split-K=8, full-model ABBA:

| Context | Direct mean | Graph mean | Mean speedup | Direct p95 | Graph p95 | P95 speedup | Top1 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 6.582971 ms | 6.123284 ms | 1.0751x | 6.800672 ms | 6.188832 ms | 1.0989x | exact |
| 5120 | 7.825203 ms | 7.923881 ms | 0.9875x | 7.929152 ms | 7.971968 ms | 0.9946x | exact |
| 6144 | 7.971439 ms | 8.031264 ms | 0.9926x | 7.970592 ms | 8.066656 ms | 0.9881x | exact |
| 7168 | 8.046827 ms | 8.236858 ms | 0.9769x | 8.039264 ms | 8.367296 ms | 0.9608x | exact |
| 8192 | 8.082230 ms | 8.205627 ms | 0.9850x | 8.084128 ms | 8.422272 ms | 0.9599x | exact |

The crossover therefore lies between C4096 and C5120. It is not safe to set the production cutoff to 5120 from this evidence.

## Refined benchmark

Keep the same topology and measure the remaining interval at 256-token spacing:

- 4096
- 4352
- 4608
- 4864
- 5120

A point is eligible for graph dispatch only if:

- `top1_agreement=true`;
- mean speedup >= 1.02x;
- p95 does not regress materially.

Choose the largest contiguous positive range starting at C4096. Do not interpolate across a measured failing point.

## Stop condition

After the refined sweep, set the production maximum to the last measured-safe boundary and keep all larger contexts on direct execution.

If only C4096 passes, use a conservative cutoff covering only the validated C4096 neighborhood rather than extrapolating toward C5120.
