#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_WORK_DIR:-${ROOT}/target/nvfp4-sm120}"
CUTLASS_DIR="${WORK_DIR}/cutlass"
BIN="${WORK_DIR}/nvfp4-phase1-accuracy-tn8"
LOG="${WORK_DIR}/nvfp4-phase1-accuracy.log"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
SOURCE="${ROOT}/research/nvfp4/nvfp4_phase1_accuracy.cu"

mkdir -p "${WORK_DIR}"

for tool in git nvcc; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[nvfp4-accuracy] missing required tool: ${tool}" >&2
    exit 1
  }
done

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${CUTLASS_REF}" \
    https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
else
  git -C "${CUTLASS_DIR}" fetch --depth 1 origin "${CUTLASS_REF}"
  git -C "${CUTLASS_DIR}" checkout --detach FETCH_HEAD
fi

git -C "${CUTLASS_DIR}" reset --hard HEAD >/dev/null

python3 - "${CUTLASS_DIR}/include/cutlass/subbyte_reference.h" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "__nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED)"
new = "__nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)"
if old in text:
    text = text.replace(old, new)
elif new not in text:
    raise SystemExit("[nvfp4-accuracy] unexpected CUTLASS atomic-load source")
path.write_text(text)
PY

echo "[nvfp4-accuracy] compiling tileN=8"
nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_120a \
  --expt-relaxed-constexpr \
  -diag-suppress=20012 \
  -DNVFP4_TILE_N=8 \
  -I "${CUTLASS_DIR}/include" \
  -I "${CUTLASS_DIR}/tools/util/include" \
  "${SOURCE}" \
  -o "${BIN}"

git -C "${CUTLASS_DIR}" reset --hard HEAD >/dev/null

: > "${LOG}"

run_shape() {
  local site="$1"
  local n="$2"
  local k="$3"
  local weight_seed="$4"
  local input_seed="$5"
  "${BIN}" \
    --site="${site}" \
    --m=1 \
    --n="${n}" \
    --k="${k}" \
    --iterations=1 \
    --weight-seed="${weight_seed}" \
    --input-seed="${input_seed}" | tee -a "${LOG}"
}

for seeds in "0x9abc 0x1234" "0x51a7 0xc0de" "0xdead 0xbeef"; do
  read -r weight_seed input_seed <<<"${seeds}"
  run_shape mlp_down 2048 8192 "${weight_seed}" "${input_seed}"
  run_shape mlp_gate_up 16384 2048 "${weight_seed}" "${input_seed}"
  run_shape lm_head 65536 2048 "${weight_seed}" "${input_seed}"
done

echo "[nvfp4-accuracy] log: ${LOG}"
