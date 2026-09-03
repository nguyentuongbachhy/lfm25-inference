# Residual RMSNorm -> FP8 fusion — Phase 1

## Baseline

This branch starts from `main` at `63a87e5557ef12c00551b635ef4faad7bb3fcb2a`, after the bounded CUDA Graph promotion.

Rejected directions are not reopened. In particular, this experiment does not replace cuBLASLt and does not change the rejected custom tiny-M GEMM or packed-QKV directions.

## Hypothesis

At selected E4M3 Gate/Up sites, decode currently executes:

1. fused residual add + RMSNorm and writes normalized BF16;
2. a separate BF16 -> E4M3 activation-quantization kernel;
3. the existing FP8 cuBLASLt Gate/Up GEMM.

The residual/RMSNorm kernel already has the normalized value in registers before storing BF16. A fused kernel can preserve the same BF16 rounding boundary, convert that rounded value directly to E4M3, and avoid the normalized BF16 global-memory round trip plus one kernel launch.

The same boundary can later be used for the final RMSNorm -> FP8 LM-head input if the primitive and full-model gates pass.

## Candidate

Reference:

`residual + update -> residual_rms_norm_bf16 -> normalized BF16 -> quantize_bf16_e4m3`

Candidate:

`residual + update -> residual_rms_norm_bf16_to_e4m3 -> {residual BF16, normalized E4M3}`

The expensive GEMM remains cuBLASLt and is outside the primitive benchmark.

## Numerical gate

For identical inputs and quantization scale:

- BF16 residual output must be bit-identical;
- E4M3 output bytes must be bit-identical;
- no non-finite or precision-policy change is allowed.

Non-ignored regression test:

`residual_rms_norm_fp8_matches_two_kernel_reference_exactly`

## Microbenchmark

Ignored test:

`bench_residual_rms_norm_fp8_fusion`

Shapes:

- M=1: primary decode gate;
- M=8 and M=16: serving diagnostics;
- hidden width = 2048.

The benchmark uses same-process paired AB/BA ordering.

## Primitive performance gate

Continue to production integration only if:

- M1 mean speedup >= 1.20x;
- M1 p95 does not regress;
- exact BF16 residual and FP8 outputs pass.

M8/M16 regressions prevent universal dispatch but do not automatically reject a B1-only candidate.

## Expected end-to-end benefit

This fusion affects only boundaries where the next linear consumes E4M3 activations. It does not accelerate the Gate/Up GEMM itself.

Therefore the expected whole-model gain is small. The direction is worth continuing only if the local two-kernel boundary is materially faster. A later full-model gate will require at least about 1% B1 TPOT improvement with no material p95 regression.

## Full-model gate if Phase 1 passes

Use selected-weight E4M3 and same-process ABBA against the current CUDA-Graph-capable baseline.

Primary shapes:

- B1/C128;
- B1/C1024;
- B1/C2048.

Secondary serving shapes:

- B16/C128;
- B8/C2048;
- B1/C8192.

Promotion requires the existing model-quality gate unchanged and no regression in graph/direct dispatch policy.

## Stop condition

If M1 primitive speedup is below 1.20x, reject this direction without modifying `DecodeExecutor`.

If the primitive passes but full-model B1 improvement is below 1.01x or p95 regresses materially, reject the production integration. One materially different fusion implementation is allowed only if profiling identifies a specific avoidable bottleneck.

## Commands

```bash
LLM_CUDA_ARCH=compute_120 cargo fmt --check
LLM_CUDA_ARCH=compute_120 cargo check --all-features
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_residual_rms_norm_fp8_fusion -- \
  --ignored --nocapture --test-threads=1
```
