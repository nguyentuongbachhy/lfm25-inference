# Residual RMSNorm -> FP8 fusion

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

## Phase 2 production integration

The candidate is opt-in with `LFM25_RMS_FP8_FUSION=1`.

Integration is intentionally narrow:

- selected FP8 Gate/Up sites use fused `residual + RMSNorm -> E4M3` and call the existing prequantized FP8 cuBLASLt GEMM;
- BF16 Gate/Up sites keep the existing residual/RMSNorm and linear path;
- the final layer fuses final RMSNorm directly into the selected FP8 LM-head activation;
- intermediate next-layer operator norms remain BF16 because subsequent attention/conv operators consume the normalized BF16 tensor;
- CUDA Graph dispatch is disabled inside the Phase 2 ABBA harness so this direction is measured in isolation.

Because the fused kernel preserves the reference BF16 rounding point before E4M3 conversion, the full-model sampled-token trace is required to remain exact.

## Expected end-to-end benefit

The selected policy contains eight FP8 Gate/Up sites plus the FP8 LM head. With roughly 10 us local saving per fused M1 boundary, the ideal aggregate saving is around 90 us per decode step before scheduling effects.

Against an approximately 6 ms decode step, a realistic expected whole-model gain is about 1-2%, not the 1.77x primitive speedup.

## Phase 2 full-model gate

Ignored test:

`bench_rms_fp8_fusion_full_model_abba`

Same-process order: Direct -> Fused -> Fused -> Direct.

Shapes:

- B1/C128: primary short-context gate;
- B16/C128: batched short-context regression gate;
- B1/C2048: primary long-context gate;
- B8/C2048: batched long-context regression gate;
- B1/C8192: very-long-context regression gate.

Promotion requires:

- `top1_agreement=true` at every shape;
- B1/C128 mean speedup >=1.01x;
- B1/C2048 mean speedup >=1.01x;
- no material p95 regression at primary B1 shapes;
- no material batched or C8192 regression;
- existing checkpoint NLL/hidden quality gate remains unchanged.

If the isolated Phase 2 gate passes, run one final combined `LFM25_CUDA_GRAPHS=1 + LFM25_RMS_FP8_FUSION=1` serving check before merging.

## Stop condition

If full-model B1 improvement is below 1.01x or p95 regresses materially, reject the production integration. One materially different fusion implementation is allowed only if profiling identifies a specific avoidable bottleneck.

## Commands

```bash
LLM_CUDA_ARCH=compute_120 cargo fmt --check
LLM_CUDA_ARCH=compute_120 cargo check --all-features
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_rms_fp8_fusion_full_model_abba -- \
  --ignored --nocapture --test-threads=1
```
