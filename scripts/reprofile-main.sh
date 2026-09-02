#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

MODEL="${MODEL:-models/LFM2.5-1.2B-Instruct}"
FP8_POLICY="${FP8_POLICY:-docs/benchmarks/fp8/selected-policy.json}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-docs/serving/fp8-splitk-hardware-ps16.cost-model.json}"
CUDA_ARCH="${LLM_CUDA_ARCH:-compute_120}"
PAIRS="${BENCHMARK_PAIRS:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-128}"
PROFILE_WARMUP="${PROFILE_WARMUP:-8}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-docs/benchmarks/reprofile/${STAMP}}"

mkdir -p "$OUT"

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'missing required file: %s\n' "$1" >&2
        exit 1
    fi
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

require_command git
require_command cargo
require_command nvidia-smi
require_command nvcc
require_file "$FP8_POLICY"
require_file "$HARDWARE_PROFILE"

CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_SHA="$(git rev-parse HEAD)"

{
    printf 'timestamp_utc=%s\n' "$STAMP"
    printf 'branch=%s\n' "$CURRENT_BRANCH"
    printf 'sha=%s\n' "$CURRENT_SHA"
    printf 'model=%s\n' "$MODEL"
    printf 'fp8_policy=%s\n' "$FP8_POLICY"
    printf 'hardware_profile=%s\n' "$HARDWARE_PROFILE"
    printf 'cuda_arch=%s\n' "$CUDA_ARCH"
    printf 'benchmark_pairs=%s\n' "$PAIRS"
    printf 'profile_warmup_steps=%s\n' "$PROFILE_WARMUP"
    printf 'profile_steps=%s\n' "$PROFILE_STEPS"
} > "$OUT/manifest.txt"

git status --short > "$OUT/git-status.txt"
nvidia-smi -q > "$OUT/nvidia-smi.txt"
nvcc --version > "$OUT/nvcc.txt"

export LLM_CUDA_ARCH="$CUDA_ARCH"

cargo fmt --check 2>&1 | tee "$OUT/cargo-fmt.log"
cargo check --all-features 2>&1 | tee "$OUT/cargo-check.log"
cargo test --release -- --test-threads=1 2>&1 | tee "$OUT/cargo-test.log"

cargo run --release -- \
    --model "$MODEL" \
    --benchmark-hardware "$OUT/hardware.json" \
    --page-size 16 2>&1 | tee "$OUT/benchmark-hardware.log"

cargo run --release -- \
    --model "$MODEL" \
    --benchmark-fp8 "$FP8_POLICY" \
    --benchmark-output "$OUT/fp8-abba.json" \
    --benchmark-pairs "$PAIRS" \
    --page-size 16 2>&1 | tee "$OUT/benchmark-fp8.log"

cargo run --release -- \
    --model "$MODEL" \
    --benchmark-batched-fp8 "$FP8_POLICY" \
    --benchmark-output "$OUT/fp8-batched-abba.json" \
    --benchmark-pairs "$PAIRS" \
    --page-size 16 2>&1 | tee "$OUT/benchmark-batched-fp8.log"

cargo run --release -- \
    --model "$MODEL" \
    --benchmark-serving "$OUT/continuous-decode.json" \
    --fp8-policy "$FP8_POLICY" \
    --page-size 16 2>&1 | tee "$OUT/benchmark-serving.log"

cargo run --release -- \
    --model "$MODEL" \
    --prompt "Profile the current production decode path." \
    --max-new-tokens "$((PROFILE_WARMUP + PROFILE_STEPS + 8))" \
    --temperature 0 \
    --fp8-policy "$FP8_POLICY" \
    --hardware-profile "$HARDWARE_PROFILE" \
    --page-size 16 \
    --profile-decode detailed \
    --profile-warmup-steps "$PROFILE_WARMUP" \
    --profile-steps "$PROFILE_STEPS" \
    --profile-output "$OUT/decode-detailed.json" 2>&1 | tee "$OUT/decode-detailed.log"

printf '\nRe-profile evidence written to %s\n' "$OUT"
printf 'Return the complete directory or at least these JSON files:\n'
printf '  %s\n' "$OUT/hardware.json"
printf '  %s\n' "$OUT/fp8-abba.json"
printf '  %s\n' "$OUT/fp8-batched-abba.json"
printf '  %s\n' "$OUT/continuous-decode.json"
printf '  %s\n' "$OUT/decode-detailed.json"
