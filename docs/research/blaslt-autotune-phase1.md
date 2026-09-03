# cuBLASLt decode algorithm autotune — Phase 1

## Baseline

This branch starts from current `main` after bounded CUDA Graph promotion and formatting cleanup.

Closed directions remain closed. This experiment does not replace cuBLASLt, change model precision, or reopen custom tiny-M GEMM, packed-QKV, FP8-KV, or RMSNorm->FP8 fusion.

## Observation

The current BF16 and FP8 plan constructors call cuBLASLt heuristic selection once and cache the returned algorithm. They do not benchmark multiple legal algorithms on the target GPU.

CUDA 12.8 `cublasLtMatmulAlgoGetHeuristic` can return multiple candidate algorithms in estimated-time order. Heuristic rank is not a measured latency guarantee for the RTX 5060 Laptop GPU.

## Hypothesis

For the fixed LFM2 decode GEMM shapes, especially M=1, a non-first legal cuBLASLt heuristic may run materially faster on SM120. Selecting a measured-best algorithm at initialization can reduce the dominant MLP/linear region without changing math or introducing a custom GEMM.

## Phase 1 candidate

Test-only autotuner:

1. construct the same descriptors/layouts/preferences as production;
2. request up to 16 legal heuristic algorithms;
3. reject candidates with failed status or workspace above the existing 32 MiB workspace;
4. run warm-up launches;
5. benchmark each candidate with CUDA events;
6. compare against the current cached first-heuristic algorithm;
7. verify BF16 output bytes match the current algorithm for deterministic inputs.

No production plan-selection change is allowed in Phase 1.

## Shapes

Primary M=1 decode shapes:

- BF16 hidden projection: M=1, N=2048, K=2048;
- BF16 Conv input: M=1, N=6144, K=2048;
- FP8 Gate/Up: M=1, N=16384, K=2048;
- FP8 Down: M=1, N=2048, K=8192;
- FP8 LM head: M=1, N=65536, K=2048.

Secondary batch diagnostics may use M=8 and M=16 after the M=1 screen.

## Numerical gate

For the same inputs, weights, scales and alpha/beta:

- candidate output must be bit-identical to the current cached algorithm when cuBLASLt produces deterministic output for the tested shape;
- otherwise NRMSE must be zero within BF16 representation and no non-finite output is allowed;
- model precision policy remains unchanged.

Any algorithm that changes numerical implementation in a way that violates the existing model quality gate is not eligible for production.

## Primitive performance gate

Continue only if at least one dominant M=1 shape improves by >=1.05x mean latency versus the current first heuristic, with no p95 regression.

A smaller local gain is not worth production autotune complexity.

## End-to-end gate

If Phase 1 passes, install measured winners only for exact matching keys and run real-checkpoint ABBA.

Promotion requires:

- B1/C128 mean TPOT >=1.01x;
- B1/C2048 mean TPOT >=1.01x;
- no material p95, batched, or C8192 regression;
- existing NLL/hidden quality gate unchanged;
- bounded CUDA Graph policy remains valid.

## Stop condition

If no dominant M=1 GEMM reaches 1.05x, reject the direction without changing production plan selection.

If local winners exist but whole-model B1 gain is below 1.01x, reject production autotuning.

Maximum iteration budget: one candidate-enumeration implementation plus one bounded production-selection implementation if Phase 1 passes.
