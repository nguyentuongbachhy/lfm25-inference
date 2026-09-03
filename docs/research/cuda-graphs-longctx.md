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

The crossover therefore lies between C4096 and C5120.

## Refined sweep

RTX 5060 Laptop GPU, same selected E4M3 policy, PS16, B1, Split-K=8:

| Context | Direct mean | Graph mean | Mean speedup | Direct p95 | Graph p95 | P95 speedup | Submit speedup | Top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 7.230642 ms | 6.281808 ms | 1.1510x | 7.752608 ms | 6.555392 ms | 1.1826x | 15.9708x | exact |
| 4352 | 7.743422 ms | 7.805860 ms | 0.9920x | 7.782400 ms | 7.904320 ms | 0.9846x | 14.8562x | exact |
| 4608 | 7.770530 ms | 7.908967 ms | 0.9825x | 7.827040 ms | 7.936320 ms | 0.9862x | 12.3281x | exact |
| 4864 | 7.758693 ms | 7.880345 ms | 0.9846x | 7.773344 ms | 7.950752 ms | 0.9777x | 12.3029x | exact |
| 5120 | 7.784657 ms | 7.892247 ms | 0.9864x | 7.803360 ms | 7.973440 ms | 0.9787x | 14.1465x | exact |

## Decision

The long-context extension is **PARTIAL**.

C4096 reproduces as a strong graph win and preserves exact sampled-token output. C4352 is already below parity, and every larger measured point regresses. Therefore the graph/direct crossover occurs inside the interval `(4096, 4352)`.

Do not set the production maximum to 4352 based only on these samples. The C4096 benchmark itself validates graph replay only through its measured 20-step decode trajectory. Starting from a 4096-token prefill, those steps advance cache context through approximately 4097..4116 tokens.

The conservative production extension is therefore:

- retain all previously validated graph ranges below 4096;
- permit the long-context Split-K=8 graph only through `context_tokens < 4117`;
- use direct execution at `context_tokens >= 4117`;
- keep the rejected Split-K=4 C512..C1023 region on direct execution.

This deliberately avoids interpolating across the unmeasured 4117..4351 interval.

## Final production gate

After changing the production maximum from 4096 to 4117, rerun the production-dispatch ABBA benchmark at:

- B1/C4096: must use CUDA Graph and preserve the >=1.02x mean gate with no p95 regression;
- B1/C4352: must remain on direct fallback and preserve exact output.

If B1/C4096 fails under the real production dispatcher, keep the old 4096 cutoff and reject the extension.

## Stop condition

No further long-context graph boundary search is planned after the production-dispatch confirmation. Contexts at and above C4352 remain direct unless a materially different graph implementation is researched in a future direction.
