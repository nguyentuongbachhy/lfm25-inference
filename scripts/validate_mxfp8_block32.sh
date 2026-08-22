#!/usr/bin/env bash
set -euo pipefail

export LLM_CUDA_ARCH="${LLM_CUDA_ARCH:-compute_120}"

MXFP8_RUST_FILES=(
  src/cuda/tests/mod.rs
  src/cuda/tests/mxfp8.rs
)

echo "[mxfp8-block32] CUDA arch: ${LLM_CUDA_ARCH}"
echo "[mxfp8-block32] formatting research delta"
rustfmt --edition 2024 --check "${MXFP8_RUST_FILES[@]}"

echo "[mxfp8-block32] compile all features"
cargo check --all-features

echo "[mxfp8-block32] clippy all targets/features"
cargo clippy --all-targets --all-features -- -D warnings

echo "[mxfp8-block32] small capability/correctness gate"
cargo test --release \
  mxfp8_block32_dynamic_scales_match_bf16_small_shape \
  -- --test-threads=1 --nocapture

echo "[mxfp8-block32] BF16 vs tensorwide FP8 vs MXFP8 shape benchmark"
cargo test --release \
  bench_mxfp8_block32_outlier_shapes \
  -- --ignored --test-threads=1 --nocapture

echo "[mxfp8-block32] isolated research gates passed"
