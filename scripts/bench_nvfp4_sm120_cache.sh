#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_WORK_DIR:-${ROOT}/target/nvfp4-sm120}"
CUTLASS_DIR="${WORK_DIR}/cutlass"
BUILD_DIR="${WORK_DIR}/cutlass-build-cache"
BIN_DIR="${WORK_DIR}/bin-cache"
LOG="${WORK_DIR}/nvfp4-cache.log"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
ITERATIONS="${NVFP4_ITERATIONS:-100}"
FLUSH_MB="${NVFP4_L2_FLUSH_MB:-128}"
BUILD_JOBS="${NVFP4_BUILD_JOBS:-$(nproc)}"
EXAMPLE_REL="examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu"
EXAMPLE_TARGET="79a_blackwell_geforce_nvfp4_bf16_gemm"
EXAMPLE_BIN="${BUILD_DIR}/examples/79_blackwell_geforce_gemm/${EXAMPLE_TARGET}"
SOURCE="${CUTLASS_DIR}/${EXAMPLE_REL}"
SUBBYTE_REL="include/cutlass/subbyte_reference.h"
SUBBYTE="${CUTLASS_DIR}/${SUBBYTE_REL}"

mkdir -p "${WORK_DIR}" "${BIN_DIR}"

for tool in git cmake nvcc python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[nvfp4-cache] missing required tool: ${tool}" >&2
    exit 1
  }
done

echo "[nvfp4-cache] CUTLASS ref: ${CUTLASS_REF}"
echo "[nvfp4-cache] nvcc: $(nvcc --version | tail -n 1)"
echo "[nvfp4-cache] L2 flush: ${FLUSH_MB} MiB"

python3 - <<'PY'
import re
import subprocess
text = subprocess.check_output(["nvcc", "--version"], text=True)
m = re.search(r"release\s+(\d+)\.(\d+)", text)
if not m:
    raise SystemExit("[nvfp4-cache] unable to parse nvcc version")
version = tuple(map(int, m.groups()))
if version < (12, 8):
    raise SystemExit(f"[nvfp4-cache] CUDA 12.8+ required, got {version[0]}.{version[1]}")
PY

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
  echo "[nvfp4-cache] cloning NVIDIA CUTLASS ${CUTLASS_REF}"
  git clone --depth 1 --branch "${CUTLASS_REF}" https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
else
  git -C "${CUTLASS_DIR}" fetch --depth 1 origin "${CUTLASS_REF}"
  git -C "${CUTLASS_DIR}" checkout --detach FETCH_HEAD
fi

git -C "${CUTLASS_DIR}" checkout -- "${EXAMPLE_REL}" "${SUBBYTE_REL}"

# CUDA 12.8.93 requires the atomic builtin scope explicitly in this CUTLASS helper.
python3 - "${SUBBYTE}" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
old = "__nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED)"
new = "__nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)"
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("[nvfp4-cache] unexpected CUTLASS atomic-load source")
p.write_text(text)
PY

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "[nvfp4-cache] configuring CUTLASS for SM120a"
  CUDACXX="$(command -v nvcc)" cmake \
    -S "${CUTLASS_DIR}" \
    -B "${BUILD_DIR}" \
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

replacements = [
    (
        "using ThreadBlockShape    = Shape<_128,_128,_128>;",
        f"using ThreadBlockShape    = Shape<_128,_{tile_n},_128>;",
    ),
    (
        "using         LayoutCTag  = cutlass::layout::RowMajor;",
        "using         LayoutCTag  = cutlass::layout::ColumnMajor;",
    ),
    (
        "using         LayoutDTag  = cutlass::layout::RowMajor;",
        "using         LayoutDTag  = cutlass::layout::ColumnMajor;",
    ),
    (
        "  int iterations;\n  int m, n, k;",
        "  int iterations;\n  int flush_mb;\n  int m, n, k;",
    ),
    (
        "    alpha(1.f), beta(0.f),\n    iterations(10)",
        "    alpha(1.f), beta(0.f),\n    iterations(10), flush_mb(0)",
    ),
    (
        '    cmd.get_cmd_line_argument("iterations", iterations);',
        '    cmd.get_cmd_line_argument("iterations", iterations);\n    cmd.get_cmd_line_argument("flush-mb", flush_mb, 0);',
    ),
    (
        '      << "  --iterations=<int>          Number of profiling iterations to perform.\\n\\n";',
        '      << "  --iterations=<int>          Number of profiling iterations to perform.\\n"\n'
        '      << "  --flush-mb=<int>            Evict L2 before each timed GEMM using this many MiB.\\n\\n";',
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f"[nvfp4-cache] patch anchor missing for tileN={tile_n}: {old[:80]}")
    text = text.replace(old, new, 1)

old_loop = '''  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);
'''
new_loop = '''  // Run profiling loop. Initialization is outside timing, matching a cached production plan.
  if (options.iterations > 0)
  {
    size_t flush_bytes = options.flush_mb > 0
      ? size_t(options.flush_mb) * size_t(1024) * size_t(1024)
      : size_t(1);
    cutlass::device_memory::allocation<uint8_t> l2_flush(flush_bytes);
    double elapsed_ms = 0.0;
    for (int iter = 0; iter < options.iterations; ++iter) {
      if (options.flush_mb > 0) {
        CUDA_CHECK(cudaMemset(l2_flush.get(), iter & 0xff, flush_bytes));
      }
      GpuTimer timer;
      timer.start();
      CUTLASS_CHECK(gemm.run());
      timer.stop();
      elapsed_ms += timer.elapsed_millis();
    }

    // Compute average runtime and GFLOPs. Cache eviction is deliberately outside timing.
    result.avg_runtime_ms = elapsed_ms / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);
'''
if old_loop not in text:
    raise SystemExit(f"[nvfp4-cache] profiling-loop anchor missing for tileN={tile_n}")
text = text.replace(old_loop, new_loop, 1)

path.write_text(text)
PY
}

build_variant() {
  local tile_n="$1"
  local dst="${BIN_DIR}/nvfp4_cache_tn${tile_n}"
  rm -f "${dst}"
  patch_variant "${tile_n}"
  echo "[nvfp4-cache] building tileN=${tile_n}"
  if cmake --build "${BUILD_DIR}" --target "${EXAMPLE_TARGET}" -j "${BUILD_JOBS}"; then
    cp "${EXAMPLE_BIN}" "${dst}"
    chmod +x "${dst}"
  else
    echo "[nvfp4-cache] tileN=${tile_n} failed to compile; skipping" >&2
  fi
}

for tile_n in 8 16 32 64 128; do
  build_variant "${tile_n}"
done

git -C "${CUTLASS_DIR}" checkout -- "${EXAMPLE_REL}" "${SUBBYTE_REL}"

compgen -G "${BIN_DIR}/nvfp4_cache_tn*" >/dev/null || {
  echo "[nvfp4-cache] no variants compiled" >&2
  exit 1
}

: > "${LOG}"

run_case() {
  local cache_mode="$1"
  local flush_mb="$2"
  local site="$3"
  local original_m="$4"
  local original_n="$5"
  local k="$6"
  local tile_n="$7"
  local bin="${BIN_DIR}/nvfp4_cache_tn${tile_n}"
  [[ -x "${bin}" ]] || return 0

  # Y^T = W * X^T. Runtime M maps to CUTLASS N.
  local cutlass_m="${original_n}"
  local cutlass_n="${original_m}"
  local output
  if ! output="$("${bin}" \
      --m="${cutlass_m}" \
      --n="${cutlass_n}" \
      --k="${k}" \
      --alpha=1 \
      --beta=0 \
      --iterations="${ITERATIONS}" \
      --flush-mb="${flush_mb}" 2>&1)"; then
    echo "nvfp4_cache cache=${cache_mode} site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} status=run_failed" | tee -a "${LOG}"
    printf '%s\n' "${output}" >> "${LOG}"
    return 0
  fi

  if ! grep -q 'Disposition: Passed' <<<"${output}"; then
    echo "nvfp4_cache cache=${cache_mode} site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} status=verification_failed" | tee -a "${LOG}"
    return 0
  fi

  local runtime_ms runtime_us gflops
  runtime_ms="$(awk '/Avg runtime:/ {print $3}' <<<"${output}" | tail -n 1)"
  gflops="$(awk '/GFLOPS:/ {print $2}' <<<"${output}" | tail -n 1)"
  runtime_us="$(python3 - "${runtime_ms}" <<'PY'
import sys
print(f"{float(sys.argv[1]) * 1000.0:.3f}")
PY
)"
  echo "nvfp4_cache cache=${cache_mode} site=${site} M=${original_m} N=${original_n} K=${k} tileN=${tile_n} mean_us=${runtime_us} gflops=${gflops} verification=pass" | tee -a "${LOG}"
}

# Phase 0.5 only needs production-relevant shapes. Test hot and explicitly L2-cold.
for original_m in 1 2 8 16 32 64; do
  for tile_n in 8 16 32 64 128; do
    for cache in "hot:0" "cold:${FLUSH_MB}"; do
      mode="${cache%%:*}"
      mb="${cache##*:}"
      run_case "${mode}" "${mb}" "mlp_down" "${original_m}" 2048 8192 "${tile_n}"
      run_case "${mode}" "${mb}" "mlp_gate_up" "${original_m}" 16384 2048 "${tile_n}"
    done
  done
done

for original_m in 1 2 8 16; do
  for tile_n in 8 16 32 64 128; do
    for cache in "hot:0" "cold:${FLUSH_MB}"; do
      mode="${cache%%:*}"
      mb="${cache##*:}"
      run_case "${mode}" "${mb}" "lm_head" "${original_m}" 65536 2048 "${tile_n}"
    done
  done
done

echo "[nvfp4-cache] best variant per cache mode / shape"
python3 - "${LOG}" <<'PY'
from pathlib import Path
import re
import sys

pat = re.compile(
    r"^nvfp4_cache cache=(\S+) site=(\S+) M=(\d+) N=(\d+) K=(\d+) "
    r"tileN=(\d+) mean_us=([0-9.]+) gflops=(\S+) verification=pass$"
)
best = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    m = pat.match(line)
    if not m:
        continue
    cache, site, M, N, K, tile, us, gflops = m.groups()
    row = (cache, site, int(M), int(N), int(K), int(tile), float(us), gflops)
    key = row[:5]
    if key not in best or row[6] < best[key][6]:
        best[key] = row

for key in sorted(best, key=lambda x: (x[1], x[2], x[0])):
    cache, site, M, N, K, tile, us, gflops = best[key]
    print(
        f"nvfp4_cache_best cache={cache} site={site} M={M} N={N} K={K} "
        f"tileN={tile} mean_us={us:.3f} gflops={gflops}"
    )
PY

echo "[nvfp4-cache] log: ${LOG}"
