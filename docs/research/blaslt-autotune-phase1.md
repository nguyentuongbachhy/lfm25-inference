# cuBLASLt decode algorithm autotune — final decision

## Verdict: REJECT

This direction is closed. Do not change production cuBLASLt plan selection from the current first legal heuristic based on this experiment.

## Baseline

This branch starts from current `main` after bounded CUDA Graph promotion and formatting cleanup.

Closed directions remain closed. This experiment did not replace cuBLASLt, change model precision, or reopen custom tiny-M GEMM, packed-QKV, FP8-KV, or RMSNorm->FP8 fusion.

## Hypothesis

The current BF16 and FP8 plan constructors ask cuBLASLt for one heuristic algorithm and cache it. The experiment tested whether another legal heuristic could materially improve the fixed M=1 LFM2 decode GEMMs on RTX 5060 Laptop GPU / SM120.

## First sweep

A sequential sweep showed large timing drift when the same candidate was measured at different points in the run. Those results were not used for promotion. The benchmark was replaced with same-process paired reference/candidate measurement.

## Paired confirmation

All paired candidates produced exact output.

| Dtype | M | N | K | Candidate | Reference mean | Candidate mean | Mean speedup | Reference p95 | Candidate p95 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 | 1 | 2048 | 2048 | 4 | 19.458 us | 20.847 us | 0.9618x | 20.069 us | 26.635 us | reject |
| FP8 | 1 | 16384 | 2048 | 1 | 39.198 us | 39.229 us | 0.9993x | 39.397 us | 40.091 us | reject |
| BF16 | 1 | 6144 | 2048 | 5 | 53.479 us | 51.914 us | 1.0305x | 55.784 us | 55.192 us | below gate |
| FP8 | 1 | 2048 | 8192 | 3 | 29.673 us | 31.532 us | 0.9396x | 38.994 us | 32.424 us | reject |

The precommitted continuation gate required at least one dominant M=1 shape to achieve >=1.05x mean speedup with no p95 regression.

The best exact candidate was BF16 Conv input candidate 5 at only 1.0305x mean. FP8 Down candidate 3 was slower in mean latency despite a better p95.

## Decision

Phase 1 fails. Production plan selection remains unchanged.

No full-model integration is justified because no primitive candidate reaches the predefined 1.05x threshold. A roughly 3% local gain on the BF16 Conv input GEMM is too small to justify keyed production overrides and would have negligible whole-model impact.

## Stop condition

Satisfied: no dominant M=1 GEMM reaches 1.05x under paired measurement.

Do not iterate further on cuBLASLt heuristic autotuning unless the GPU, CUDA/cuBLASLt version, model shapes, or measurement policy changes materially.
