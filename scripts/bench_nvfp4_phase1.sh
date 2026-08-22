#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_WORK_DIR:-${ROOT}/target/nvfp4-sm120}"
CUTLASS_DIR="${WORK_DIR}/cutlass"
BIN="${WORK_DIR}/nvfp4-phase1-tn8"
LOG="${WORK_DIR}/nvfp4-phase1.log"
BASELINE_LOG="${WORK_DIR}/rust-baseline-phase1.log"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
ITERATIONS="${NVFP4_ITERATIONS:-100}"
SOURCE="${ROOT}/research/nvfp4/nvfp4_phase1.cu"

mkdir -p "${WORK_DIR}"

for tool in git nvcc cargo python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[nvfp4-phase1] missing required tool: ${tool}" >&2
    exit 1
  }
done

export LLM_CUDA_ARCH="${LLM_CUDA_ARCH:-compute_120}"

echo "[nvfp4-phase1] CUTLASS ref: ${CUTLASS_REF}"
echo "[nvfp4-phase1] nvcc: $(nvcc --version | tail -n 1)"
echo "[nvfp4-phase1] scope: cached weights, dynamic activation, M=1"

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
    raise SystemExit("[nvfp4-phase1] unexpected CUTLASS atomic-load source")
path.write_text(text)
PY

echo "[nvfp4-phase1] compiling tileN=8"
nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_120a \
  --expt-relaxed-constexpr \
  -DNVFP4_TILE_N=8 \
  -include functional \
  -I "${CUTLASS_DIR}/include" \
  -I "${CUTLASS_DIR}/tools/util/include" \
  "${SOURCE}" \
  -o "${BIN}"

git -C "${CUTLASS_DIR}" reset --hard HEAD >/dev/null

echo "[nvfp4-phase1] validating current Rust baseline"
cargo fmt --check
cargo check --all-features
cargo clippy --all-targets --all-features -- -D warnings

echo "[nvfp4-phase1] collecting M=1 E4M3 baseline"
{
  cargo test --release \
    bench_lfm_narrow_precision_gemms \
    -- --ignored --test-threads=1 --nocapture
} | tee "${BASELINE_LOG}" >/dev/null

grep -E '^(mlp_gate_up|mlp_down|lm_head),1,(fp8_e4m3|activation_quantize|fp8_quantize_gemm),' \
  "${BASELINE_LOG}" || true

: > "${LOG}"

run_shape() {
  local site="$1"
  local n="$2"
  local k="$3"
  "${BIN}" \
    --site="${site}" \
    --m=1 \
    --n="${n}" \
    --k="${k}" \
    --iterations="${ITERATIONS}" | tee -a "${LOG}"
}

run_shape mlp_down 2048 8192
run_shape mlp_gate_up 16384 2048
run_shape lm_head 65536 2048

echo "[nvfp4-phase1] comparison against current E4M3 quantize+GEMM"
python3 - "${BASELINE_LOG}" "${LOG}" <<'PY'
from pathlib import Path
import re
import sys

baseline_text = Path(sys.argv[1]).read_text()
phase1_text = Path(sys.argv[2]).read_text()

baseline = {}
for line in baseline_text.splitlines():
    parts = line.split(",")
    if len(parts) < 4:
        continue
    if parts[0] not in {"mlp_down", "mlp_gate_up", "lm_head"} or parts[1] != "1":
        continue
    if parts[2] == "fp8_quantize_gemm":
        baseline[parts[0]] = float(parts[3])

pattern = re.compile(
    r"^nvfp4_phase1 site=(\S+) M=1 N=(\d+) K=(\d+) tileN=(\d+) "
    r"quant_hot_us=([0-9.]+) gemm_hot_us=([0-9.]+) "
    r"e2e_hot_us=([0-9.]+) e2e_cold_us=([0-9.]+)$"
)

for line in phase1_text.splitlines():
    match = pattern.match(line)
    if not match:
        continue
    site, n, k, tile_n, quant, gemm, hot, cold = match.groups()
    if site not in baseline:
        continue
    e4 = baseline[site]
    cold = float(cold)
    hot = float(hot)
    print(
        f"nvfp4_phase1_compare site={site} M=1 "
        f"e4m3_quant_gemm_us={e4:.3f} "
        f"nvfp4_hot_e2e_us={hot:.3f} "
        f"nvfp4_cold_e2e_us={cold:.3f} "
        f"cold_speedup_vs_e4m3={e4 / cold:.4f}x"
    )
PY

echo "[nvfp4-phase1] logs:"
echo "  ${BASELINE_LOG}"
echo "  ${LOG}"
