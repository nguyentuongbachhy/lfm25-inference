# Residual RMSNorm -> FP8 fusion

## Final decision — REJECT

The primitive fusion is valid and fast, but the full-model production gate fails the precommitted B1/C2048 requirement. Do not merge this branch and do not add another implementation unless new profiling identifies a specific avoidable bottleneck.

## Baseline

This branch is based on current `main` at `4bf64bb1ea993b8deed3697a6b406719210ce371`, which contains the bounded CUDA Graph promotion plus Rust formatting cleanup.

Rejected directions are not reopened. This experiment does not replace cuBLASLt and does not change the rejected custom tiny-M GEMM or packed-QKV directions.

## Hypothesis

At selected E4M3 Gate/Up sites, decode executes:

1. fused residual add + RMSNorm and writes normalized BF16;
2. a separate BF16 -> E4M3 activation-quantization kernel;
3. the existing FP8 cuBLASLt Gate/Up GEMM.

The residual/RMSNorm kernel already has the normalized value in registers before storing BF16. A fused kernel preserves the same BF16 rounding boundary, converts that rounded value directly to E4M3, and avoids the normalized BF16 global-memory round trip plus one kernel launch.

The same boundary applies to the final RMSNorm -> FP8 LM-head input.

## Candidate

Reference:

`residual + update -> residual_rms_norm_bf16 -> normalized BF16 -> quantize_bf16_e4m3`

Candidate:

`residual + update -> residual_rms_norm_bf16_to_e4m3 -> {residual BF16, normalized E4M3}`

The expensive GEMM remains cuBLASLt.

## Numerical gate

For identical inputs and quantization scale:

- BF16 residual output must be bit-identical;
- E4M3 output bytes must be bit-identical;
- no non-finite or precision-policy change is allowed.

Non-ignored regression test:

`residual_rms_norm_fp8_matches_two_kernel_reference_exactly`

## Phase 1 primitive result — PASS

RTX 5060 Laptop GPU, SM120, same-process paired AB/BA:

| Shape | Reference mean | Fused mean | Mean speedup | Reference p95 | Fused p95 | Exact |
|---|---:|---:|---:|---:|---:|---|
| M1 | 24.850 us | 14.437 us | 1.7748x | 30.081 us | 17.080 us | residual + FP8 |
| M8 | 23.743 us | 13.712 us | 1.7467x | 29.196 us | 16.512 us | residual + FP8 |
| M16 | 15.713 us | 8.321 us | 1.9151x | 23.309 us | 12.423 us | residual + FP8 |

The precommitted M1 gate was >=1.20x mean with no p95 regression. M1 achieves 1.7748x and exact output, so Phase 1 passes decisively.

## Phase 2 production result — REJECT

Real-checkpoint selected-E4M3 production decode, CUDA Graph dispatch disabled to isolate this direction:

| Shape | Direct mean | Fused mean | Mean speedup | Direct p95 | Fused p95 | P95 speedup | Top1 |
|---|---:|---:|---:|---:|---:|---:|---|
| B1/C128 | 6.050889 ms | 5.973100 ms | 1.0130x | 6.232704 ms | 6.159936 ms | 1.0118x | exact |
| B16/C128 | 6.074366 ms | 6.055652 ms | 1.0031x | 6.133536 ms | 6.117088 ms | 1.0027x | exact |
| B1/C2048 | 6.158916 ms | 6.125776 ms | 1.0054x | 6.390400 ms | 6.146336 ms | 1.0397x | exact |
| B8/C2048 | 8.593808 ms | 8.570881 ms | 1.0027x | 8.630976 ms | 8.592512 ms | 1.0045x | exact |
| B1/C8192 | 8.106916 ms | 8.075912 ms | 1.0038x | 8.221120 ms | 8.182560 ms | 1.0047x | exact |

The full-model gate required both B1/C128 and B1/C2048 mean speedup >=1.01x. B1/C128 passes at 1.0130x, but B1/C2048 reaches only 1.0054x. Therefore the production integration is rejected even though every token trace is exact and no measured shape materially regresses.

The result shows that removing nine small residual/RMSNorm-to-FP8 boundaries is not large enough to move whole-model TPOT consistently. The primitive optimization is real, but Amdahl's law dominates at the model level.

## Integration attempted

The rejected candidate was opt-in with `LFM25_RMS_FP8_FUSION=1` and was limited to selected FP8 Gate/Up sites plus the final RMSNorm -> FP8 LM-head boundary. BF16-only sites remained unchanged.

Do not enable or merge this production path.

## Stop condition

Satisfied: B1/C2048 mean improvement is below 1.01x. Close the direction. No second implementation is justified because the benchmark does not expose a specific local bottleneck; the remaining issue is insufficient whole-model weight.
