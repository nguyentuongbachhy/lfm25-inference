#!/usr/bin/env bash
set -euo pipefail

export LLM_CUDA_ARCH="${LLM_CUDA_ARCH:-compute_120}"

# The v1/main base still carries unrelated rustfmt debt. Runtime-v2 therefore
# checks formatting only for Rust files touched by this release delta so the
# validation branch does not absorb cosmetic changes outside its scope.
RUNTIME_V2_RUST_FILES=(
  src/cuda/kernels/sampling.rs
  src/model/argmax_production_tests.rs
  src/model/mod.rs
  src/ops/mod.rs
  src/ops/sampling.rs
  src/ops/sampling_dispatch.rs
)

echo "[runtime-v2] CUDA arch: ${LLM_CUDA_ARCH}"
echo "[runtime-v2] formatting release delta"
rustfmt --edition 2024 --check "${RUNTIME_V2_RUST_FILES[@]}"

echo "[runtime-v2] compile all features"
cargo check --all-features

echo "[runtime-v2] clippy all targets/features"
cargo clippy --all-targets --all-features -- -D warnings

echo "[runtime-v2] atomic argmax correctness"
cargo test --release \
  atomic_argmax \
  -- --test-threads=1 --nocapture

echo "[runtime-v2] production dispatch policy"
cargo test --release \
  production_policy_ \
  -- --test-threads=1 --nocapture

echo "[runtime-v2] full-model ABBA sampled-trace gate"
cargo test --release \
  bench_production_atomic_argmax_abba \
  -- --ignored --test-threads=1 --nocapture

echo "[runtime-v2] all validation gates passed"
