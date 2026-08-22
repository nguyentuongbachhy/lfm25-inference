#!/usr/bin/env bash
set -euo pipefail

export LLM_CUDA_ARCH="${LLM_CUDA_ARCH:-compute_120}"

RUST_FILES=(
  src/cuda/tests/mod.rs
  src/cuda/tests/mxfp8_algo.rs
)

echo "[mxfp8-algos] CUDA arch: ${LLM_CUDA_ARCH}"
echo "[mxfp8-algos] formatting research delta"
rustfmt --edition 2024 --check "${RUST_FILES[@]}"

echo "[mxfp8-algos] compile all features"
cargo check --all-features

echo "[mxfp8-algos] clippy all targets/features"
cargo clippy --all-targets --all-features -- -D warnings

echo "[mxfp8-algos] cuBLASLt heuristic candidate sweep"
cargo test --release \
  bench_mxfp8_block32_algorithm_sweep \
  -- --ignored --test-threads=1 --nocapture
