#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_WORK_DIR:-${ROOT}/target/nvfp4-sm120}"
CUTLASS_DIR="${WORK_DIR}/cutlass"
CUTLASS_BUILD_DIR="${WORK_DIR}/cutlass-build"
BIN_DIR="${WORK_DIR}/bin"
BASELINE_LOG="${WORK_DIR}/rust-baseline.log"
NVFP4_LOG="${WORK_DIR}/nvfp4-cutlass.log"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
ITERATIONS="${NVFP4_ITERATIONS:-100}"
BUILD_JOBS="${NVFP4_BUILD_JOBS:-$(nproc)}"
EXAMPLE_REL="examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu"
EXAMPLE_TARGET="79a_blackwell_geforce_nvfp4_bf16_gemm"
EXAMPLE_BIN="${CUTLASS_BUILD_DIR}/examples/79_blackwell_geforce_gemm/${EXAMPLE_TARGET}"

mkdir -p "${WORK_DIR}" "${BIN_DIR}"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[nvfp4-sm120] missing required tool: $1" >&2
    exit 1
  fi
}

for tool in git cmake nvcc cargo python3; do
  require_tool "${tool}"
done

export LLM_CUDA_ARCH="${LLM_CUDA_ARCH:-compute_120}"

echo "[nvfp4-sm120] CUDA arch: ${LLM_CUDA_ARCH}"
echo "[nvfp4-sm120] CUTLASS ref: ${CUTLASS_REF}"
echo "[nvfp4-sm120] nvcc: $(nvcc --version | tail -n 1)"

python3 - <<'PY'
import re
import subprocess

text = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\s+(\d+)\.(\d+)", text)
if not match:
    raise SystemExit("[nvfp4-sm120] unable to parse nvcc version")
version = tuple(map(int, match.groups()))
if version < (12, 8):
    raise SystemExit(
        f"[nvfp4-sm120] CUDA 12.8+ is required for SM120 NVFP4; "
        f"found {version[0]}.{version[1]}"
    )
PY

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
  echo "[nvfp4-sm120] cloning NVIDIA CUTLASS ${CUTLASS_REF}"
  git clone --depth 1 --branch "${CUTLASS_REF}" https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
else
  echo "[nvfp4-sm120] reusing CUTLASS checkout"
  git -C "${CUTLASS_DIR}" fetch --depth 1 origin "${CUTLASS_REF}"
  git -C "${CUTLASS_DIR}" checkout --detach FETCH_HEAD
fi

SOURCE="${CUTLASS_DIR}/${EXAMPLE_REL}"
if [[ ! -f "${SOURCE}" ]]; then
  echo "[nvfp4-sm120] CUTLASS NVFP4 example not found: ${SOURCE}" >&2
  exit 1
fi

git -C "${CUTLASS_DIR}" checkout -- "${EXAMPLE_REL}"

if [[ ! -f "${CUTLASS_BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "[nvfp4-sm120] configuring CUTLASS for SM120a"
  CUDACXX="$(command -v nvcc)" cmake \
    -S "${CUTLASS_DIR}" \
    -B "${CUTLASS_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUTLASS_NVCC_ARCHS=120a \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCUTLASS_ENABLE_CUBLAS=OFF \
    -DCUTLASS_ENABLE_CUDNN=OFF
fi

patch_variant() {
  local tile_n="$1"
  git -C "${CUTLASS_DIR}" checkout -- "${EXAMPLE_REL}"
  python3 - "${SOURCE}" "${tile_n}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
tile_n = sys.argv[2]
text = path.read_text()
old_tile = "using ThreadBlockShape    = Shape<_128,_128,_128>;"
new_tile = f"using ThreadBlockShape    = Shape<_128,_{tile_n},_128>;"
if old_tile not in text:
    raise SystemExit(
        "[nvfp4-sm120] CUTLASS example tile declaration changed; refusing to patch"
    )
text = text.replace(old_tile, new_tile, 1)
text = text.replace(
    "using         LayoutCTag  = cutlass::layout::RowMajor;",
    "using         LayoutCTag  = cutlass::layout::ColumnMajor;",
    1,
)
text = text.replace(
    "using         LayoutDTag  = cutlass::layout::RowMajor;",
    "using         LayoutDTag  = cutlass::layout::ColumnMajor;",
    1,
)
path.write_text(text)
PY
}

build_variant() {
  local tile_n="$1"
  local dst="${BIN_DIR}/nvfp4_tn${tile_n}"
  echo "[nvfp4-sm120] building CUTLASS NVFP4 tileN=${tile_n}"
  patch_variant "${tile_n}"
  if cmake --build "${CUTLASS_BUILD_DIR}" --target "${EXAMPLE_TARGET}" -j "${BUILD_JOBS}"; then
    cp "${EXAMPLE_BIN}" "${dst}"
    chmod +x "${dst}"
    echo "[nvfp4-sm120] built ${dst}"
  else
    echo "[nvfp4-sm120] tileN=${tile_n} failed to compile; skipping" >&2
    rm -f "${dst}"
  fi
}

for tile_n in 8 16 32 64 128; do
  build_variant "${tile_n}"
done

git -C "${CUTLASS_DIR}" checkout -- "${EXAMPLE_REL}"

if ! compgen -G "${BIN_DIR}/nvfp4_tn*" >/dev/null; then
  echo "[nvfp4-sm120] no NVFP4 CUTLASS variants compiled" >&2
  exit 1
fi

echo "[nvfp4-sm120] validating Rust baseline"
cargo check --all-features
cargo clippy --all-targets --all-features -- -D warnings

echo "[nvfp4-sm120] benchmarking current BF16/tensorwide-E4M3 baseline"
{
  cargo test --release \
    bench_lfm_narrow_precision_gemms \
    -- --ignored --test-threads=1 --nocapture
} | tee "${BASELINE_LOG}"

echo "[nvfp4-sm120] relevant production baselines"
grep -E '^(mlp_gate_up|mlp_down|lm_head),(1|2|8|16|32|64),(bf16|fp8_e4m3|activation_quantize|fp8_quantize_gemm),' "${BASELINE_LOG}" || true

: > "${NVFP4_LOG}"

run_nvfp4() {
  local site="$1"
  local original_m="$2"
  local original_n="$3"
  local k="$4"
  local tile_n="$5"
  local bin="${BIN_DIR}/nvfp4_tn${tile_n}"

  [[ -x "${bin}" ]] || return 0

  # Compute Y^T = W * X^T so the tiny runtime M maps to CUTLASS N.
  # Column-major D makes the physical output layout equivalent to row-major Y.
  local cutlass_m="${original_n}"
  local cutlass_n="${original_m}"
  local output

  if ! output="$("${bin}" \
      --m="${cutlass_m}" \
      --n="${cutlass_n}" \
      --k="${k}" \
      --alpha=1 \
      --beta=0 \
      --iterations="${ITERATIONS}" 2>&1)"; then
    echo "nvfp4_cutlass site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} status=run_failed" | tee -a "${NVFP4_LOG}"
    printf '%s\n' "${output}" >> "${NVFP4_LOG}"
    return 0
  fi

  if ! grep -q 'Disposition: Passed' <<<"${output}"; then
    echo "nvfp4_cutlass site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} status=verification_failed" | tee -a "${NVFP4_LOG}"
    printf '%s\n' "${output}" >> "${NVFP4_LOG}"
    return 0
  fi

  local runtime_ms
  local gflops
  runtime_ms="$(awk '/Avg runtime:/ {print $3}' <<<"${output}" | tail -n 1)"
  gflops="$(awk '/GFLOPS:/ {print $2}' <<<"${output}" | tail -n 1)"
  if [[ -z "${runtime_ms}" ]]; then
    echo "nvfp4_cutlass site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} status=parse_failed" | tee -a "${NVFP4_LOG}"
    printf '%s\n' "${output}" >> "${NVFP4_LOG}"
    return 0
  fi

  local runtime_us
  runtime_us="$(python3 - "${runtime_ms}" <<'PY'
import sys
print(f"{float(sys.argv[1]) * 1000.0:.3f}")
PY
)"

  echo "nvfp4_cutlass site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} mean_us=${runtime_us} gflops=${gflops} verification=pass" | tee -a "${NVFP4_LOG}"
}

for original_m in 1 2 8 16 32 64; do
  for tile_n in 8 16 32 64 128; do
    run_nvfp4 "mlp_down" "${original_m}" 2048 8192 "${tile_n}"
    run_nvfp4 "mlp_gate_up" "${original_m}" 16384 2048 "${tile_n}"
  done
done

for original_m in 1 2 8 16; do
  for tile_n in 8 16 32 64 128; do
    run_nvfp4 "lm_head" "${original_m}" 65536 2048 "${tile_n}"
  done
done

echo "[nvfp4-sm120] best verified NVFP4 variant per shape"
python3 - "${NVFP4_LOG}" <<'PY'
from pathlib import Path
import re
import sys

rows = []
pattern = re.compile(
    r"^nvfp4_cutlass site=(\S+) M=(\d+) N=(\d+) K=(\d+) tileN=(\d+) "
    r"mean_us=([0-9.]+) gflops=(\S+) verification=pass$"
)
for line in Path(sys.argv[1]).read_text().splitlines():
    match = pattern.match(line)
    if not match:
        continue
    site, m, n, k, tile_n, mean_us, gflops = match.groups()
    rows.append((site, int(m), int(n), int(k), int(tile_n), float(mean_us), gflops))

best = {}
for row in rows:
    key = row[:4]
    if key not in best or row[5] < best[key][5]:
        best[key] = row

for key in sorted(best, key=lambda item: (item[0], item[1])):
    site, m, n, k, tile_n, mean_us, gflops = best[key]
    print(
        f"nvfp4_best site={site} M={m} N={n} K={k} "
        f"tileN={tile_n} mean_us={mean_us:.3f} gflops={gflops}"
    )
PY

echo "[nvfp4-sm120] logs:"
echo "  ${BASELINE_LOG}"
echo "  ${NVFP4_LOG}"
