# CUDA Graph long-context crossover

## Baseline

This branch starts from current `main` at `a25eacb37a41f32c2d2080bc0be47604dedd24a8`.

The existing production graph policy is already validated for B1/PS16 in selected short and moderate context ranges. This experiment does not change Split-K policy or reopen rejected Split-K tuning.

## Observation

Prior full-model ABBA results showed:

- B1/C4096: graph mean speedup 1.1688x, p95 speedup 1.2600x, top1 exact;
- B1/C8192: graph mean speedup 0.9647x, p95 speedup 0.9646x, top1 exact.

Production currently routes `C >= 4096` to direct execution, so the positive C4096 point is not used.

## Hypothesis

The graph/direct crossover lies between C4096 and C8192 while the attention topology remains unchanged (PS16, B1, Split-K=8). Extending graph dispatch only through the measured-positive long-context range can improve B1 latency without changing numerical behavior or Split-K selection.

## Benchmark

Use the existing full-model ABBA harness with exact sampled-token agreement and the selected E4M3 policy.

Test contexts:

- 4096
- 5120
- 6144
- 7168
- 8192

All are B1, PS16 and expected to remain Split-K=8 for the complete measured decode window.

## Gate

A context point is eligible for graph dispatch only if:

- `top1_agreement=true`;
- mean speedup >= 1.02x;
- p95 does not regress materially.

Choose the largest contiguous positive range starting at C4096. Do not interpolate across a measured failing point.

## Stop condition

If C4096 fails to reproduce at >=1.02x, reject the extension and keep the existing production boundary.

If later points regress, cap the production maximum at the last measured-safe boundary. C8192 remains direct unless it independently passes, which prior evidence says is unlikely.
