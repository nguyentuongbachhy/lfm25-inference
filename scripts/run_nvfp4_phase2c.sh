#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_PHASE2C_WORK_DIR:-${ROOT}/target/nvfp4-sm120-phase2c}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(realpath "${WORK_DIR}")"
CUTLASS_DIR="${NVFP4_PHASE2C_CUTLASS_DIR:-${ROOT}/target/nvfp4-sm120-phase2b/cutlass}"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
REPLAY_SOURCE="${ROOT}/research/nvfp4/nvfp4_replay_site.cu"
REPLAY_BIN="${WORK_DIR}/nvfp4-replay-tn8"
REPLAY_DATA="${WORK_DIR}/replay-data"
POLICY="${NVFP4_PHASE2C_E4M3_POLICY:-${ROOT}/docs/benchmarks/fp8/selected-policy.json}"
MODEL="${NVFP4_MODEL:-${ROOT}/models/LFM2.5-1.2B-Instruct}"
SEQUENCES="${NVFP4_PHASE2C_SEQUENCES:-8}"
MAX_TOKENS="${NVFP4_PHASE2C_MAX_TOKENS:-256}"
POSITIONS="${NVFP4_PHASE2C_POSITIONS:-8}"
EXPECTED_GPU_SUBSTRING="${NVFP4_EXPECTED_GPU_SUBSTRING:-RTX 5060 Laptop}"
VALIDATION_SHA256="f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8"
EXPECTED_TEST_SHA256="${NVFP4_PHASE2C_EXPECTED_SHA256:-}"
SUMMARY="${WORK_DIR}/nvfp4-phase2c-summary.txt"

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/run_nvfp4_phase2c.sh DISJOINT_TEST_CORPUS" >&2
  exit 2
fi

TEST_CORPUS="$(realpath "$1")"
MODEL="$(realpath "${MODEL}")"
POLICY="$(realpath "${POLICY}")"
mkdir -p "${REPLAY_DATA}"
rm -f "${SUMMARY}" "${WORK_DIR}"/phase2c-*.json "${WORK_DIR}"/phase2c-*.log

for tool in git nvcc cargo python3 realpath nvidia-smi sha256sum wc tee; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[nvfp4-phase2c] missing required host tool: ${tool}" >&2
    exit 1
  }
done

if [[ "${ROOT}" != "/home/hyy4hc/source/lfm25-inference" && "${NVFP4_ALLOW_ALT_HOST_PATH:-0}" != "1" ]]; then
  echo "[nvfp4-phase2c] refusing to run outside canonical host workspace: ${ROOT}" >&2
  exit 1
fi

BRANCH="$(git -C "${ROOT}" branch --show-current)"
HEAD="$(git -C "${ROOT}" rev-parse HEAD)"
if [[ "${BRANCH}" != "agent/nvfp4-sm120" ]]; then
  echo "[nvfp4-phase2c] checkout agent/nvfp4-sm120 before running; current=${BRANCH}" >&2
  exit 1
fi
if ! git -C "${ROOT}" merge-base --is-ancestor 117c4a66828970344cd757d1e977bd729e891526 HEAD; then
  echo "[nvfp4-phase2c] branch does not contain completed Phase 2A commit 117c4a6" >&2
  exit 1
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! grep -Fq "${EXPECTED_GPU_SUBSTRING}" <<<"${GPU_NAMES}"; then
  echo "[nvfp4-phase2c] expected host GPU containing '${EXPECTED_GPU_SUBSTRING}', got: ${GPU_NAMES:-<none>}" >&2
  exit 1
fi
NVCC_VERSION_OUTPUT="$(nvcc --version)"
if ! grep -Eq 'release[[:space:]]+12\.8([,[:space:]]|$)|cuda_12\.8([._/[:space:]]|$)' <<<"${NVCC_VERSION_OUTPUT}"; then
  echo "[nvfp4-phase2c] expected CUDA 12.8.x; nvcc reported:" >&2
  printf '%s\n' "${NVCC_VERSION_OUTPUT}" >&2
  exit 1
fi

TEST_SHA256="$(sha256sum "${TEST_CORPUS}" | awk '{print $1}')"
TEST_BYTES="$(wc -c < "${TEST_CORPUS}" | tr -d ' ')"
TEST_LINES="$(wc -l < "${TEST_CORPUS}" | tr -d ' ')"
if [[ "${TEST_SHA256}" == "${VALIDATION_SHA256}" ]]; then
  echo "[nvfp4-phase2c] refusing Phase 2A/2B validation split; Phase 2C requires a disjoint test split" >&2
  exit 1
fi
if [[ -n "${EXPECTED_TEST_SHA256}" && "${TEST_SHA256}" != "${EXPECTED_TEST_SHA256}" ]]; then
  echo "[nvfp4-phase2c] test corpus hash mismatch" >&2
  echo "[nvfp4-phase2c] expected=${EXPECTED_TEST_SHA256}" >&2
  echo "[nvfp4-phase2c] actual=${TEST_SHA256}" >&2
  exit 1
fi

python3 - "${POLICY}" <<'PY'
import json
import sys
from pathlib import Path

policy = json.loads(Path(sys.argv[1]).read_text())
enabled = [site["site"] for site in policy["sites"] if site.get("enabled")]
required = {"layers.8.mlp.gate_up", "layers.9.mlp.gate_up"}
if len(enabled) != 16:
    raise SystemExit(f"[nvfp4-phase2c] expected validated 16-site E4M3 policy, got {len(enabled)} enabled sites")
missing = required - set(enabled)
if missing:
    raise SystemExit(f"[nvfp4-phase2c] production E4M3 policy missing replacement sites: {sorted(missing)}")
print(f"[nvfp4-phase2c] production E4M3 baseline sites: {len(enabled)}")
PY

echo "[nvfp4-phase2c] host: ${ROOT}"
echo "[nvfp4-phase2c] branch/head: ${BRANCH} ${HEAD}"
echo "[nvfp4-phase2c] GPU: ${GPU_NAMES//$'\n'/; }"
echo "[nvfp4-phase2c] test corpus: ${TEST_CORPUS}"
echo "[nvfp4-phase2c] test sha256: ${TEST_SHA256} bytes=${TEST_BYTES} lines=${TEST_LINES}"
echo "[nvfp4-phase2c] scope: E4M3 vs hybrid sampled quality only; replay wall time is NOT performance evidence"

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${CUTLASS_DIR}")"
  git clone --depth 1 --branch "${CUTLASS_REF}" https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
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
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit("[nvfp4-phase2c] unexpected CUTLASS atomic-load source")
PY

echo "[nvfp4-phase2c] compiling exact nearest-only replay kernel"
nvcc \
  -std=c++17 -O3 -arch=sm_120a --expt-relaxed-constexpr \
  -diag-suppress=20012 -DNVFP4_TILE_N=8 \
  -I "${CUTLASS_DIR}/include" \
  -I "${CUTLASS_DIR}/tools/util/include" \
  "${REPLAY_SOURCE}" -o "${REPLAY_BIN}"
git -C "${CUTLASS_DIR}" reset --hard HEAD >/dev/null

WORKTREE="$(mktemp -d /tmp/lfm25-nvfp4-phase2c.XXXXXX)"
cleanup() {
  git -C "${ROOT}" worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true
  rm -rf "${WORKTREE}"
}
trap cleanup EXIT

git -C "${ROOT}" worktree add --detach "${WORKTREE}" HEAD >/dev/null
python3 "${ROOT}/research/nvfp4/patch_phase2c_worktree.py" "${WORKTREE}" "${ROOT}"
(
  cd "${WORKTREE}"
  cargo fmt
  LLM_CUDA_ARCH=compute_120 cargo check --features nvfp4-research
)

export NVFP4_REPLAY_BIN="${REPLAY_BIN}"
export NVFP4_REPLAY_WORK_DIR="${REPLAY_DATA}"
export NVFP4_PHASE2C_MODEL="${MODEL}"
export NVFP4_PHASE2C_CORPUS="${TEST_CORPUS}"
export NVFP4_PHASE2C_POLICY="${POLICY}"
export NVFP4_PHASE2C_SEQUENCES="${SEQUENCES}"
export NVFP4_PHASE2C_MAX_TOKENS="${MAX_TOKENS}"
export NVFP4_PHASE2C_POSITIONS="${POSITIONS}"

run_mode() {
  local label="$1"
  local allowlist="$2"
  local output="${WORK_DIR}/phase2c-${label}.json"
  local log="${WORK_DIR}/phase2c-${label}.log"
  echo "[nvfp4-phase2c] mode=${label} allowlist='${allowlist}'"
  export NVFP4_REPLAY_ALLOWLIST="${allowlist}"
  export NVFP4_PHASE2C_OUTPUT="${output}"
  set +e
  (
    cd "${WORKTREE}"
    RUST_BACKTRACE=1 LLM_CUDA_ARCH=compute_120 cargo test --release --features nvfp4-research \
      nvfp4_phase2c_sampled_hybrid_quality \
      -- --ignored --nocapture --test-threads=1
  ) 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[nvfp4-phase2c] mode ${label} failed; log=${log}" >&2
    grep -nE 'Error:|Caused by:|NVFP4 replay failed|CUDA error|CUTLASS error|panicked at|failed to' "${log}" | tail -n 80 >&2 || true
    exit "${status}"
  fi
}

# Empty allowlist means all 16 enabled sites execute the normal production E4M3 path.
run_mode e4m3 ""
run_mode l8 "layers.8.mlp.gate_up"
run_mode l9 "layers.9.mlp.gate_up"
run_mode l8-l9 "layers.8.mlp.gate_up,layers.9.mlp.gate_up"

python3 - "${WORK_DIR}" "${SUMMARY}" "${TEST_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
summary = Path(sys.argv[2])
corpus_sha = sys.argv[3]
labels = ["e4m3", "l8", "l9", "l8-l9"]
reports = {label: json.loads((work / f"phase2c-{label}.json").read_text()) for label in labels}


def extract(report):
    quality = report["quality"]
    metrics = quality["metrics"]
    final = next(point for point in quality["propagation"] if point["point"] == "final_rms_norm")
    return {
        "rel_nll": metrics["relative_nll_delta"],
        "abs_nll": metrics["absolute_nll_delta"],
        "ppl_delta": metrics["perplexity_delta"],
        "kl_mean": metrics["mean_kl_bf16_to_candidate"],
        "kl_p95": metrics["p95_kl_bf16_to_candidate"],
        "kl_p99": metrics["p99_kl_bf16_to_candidate"],
        "logit_cos": metrics["mean_logit_cosine"],
        "top1": metrics["top1_agreement"],
        "top5": metrics["mean_top5_overlap"],
        "top10": metrics["mean_top10_overlap"],
        "nonfinite": metrics["non_finite_logits"],
        "hidden_nrmse": final["nrmse"],
        "hidden_cos": final["cosine"],
        "hidden_nonfinite": final["non_finite_values"],
    }

m = {label: extract(report) for label, report in reports.items()}
base = m["e4m3"]


def screen(label):
    if label == "e4m3":
        return True
    x = m[label]
    return (
        x["nonfinite"] == 0
        and x["hidden_nonfinite"] == 0
        and x["hidden_nrmse"] <= 0.10
        and x["hidden_cos"] >= 0.995
        and x["kl_mean"] <= 0.020
        and x["rel_nll"] <= 0.0075
        and x["rel_nll"] - base["rel_nll"] <= 0.005
        and x["kl_mean"] - base["kl_mean"] <= 0.010
        and x["top1"] >= base["top1"] - 0.05
    )

combo_pass = screen("l8-l9")
single_pass = [label for label in ("l8", "l9") if screen(label)]
if combo_pass:
    decision = "PROCEED_IN_PROCESS: layers.8.mlp.gate_up + layers.9.mlp.gate_up"
elif single_pass:
    chosen = min(
        single_pass,
        key=lambda label: (
            max(0.0, m[label]["rel_nll"] - base["rel_nll"]) * 10.0
            + max(0.0, m[label]["kl_mean"] - base["kl_mean"])
            + m[label]["hidden_nrmse"]
        ),
    )
    site = "layers.8.mlp.gate_up" if chosen == "l8" else "layers.9.mlp.gate_up"
    decision = f"NARROW_AND_PROCEED_IN_PROCESS: {site}"
else:
    decision = "REJECT_NVFP4: focused hybrid confirmation did not survive"

lines = [
    "NVFP4 Phase 2C focused hybrid confirmation",
    f"test_sha256: {corpus_sha}",
    "scope: sampled teacher-forced quality; external replay wall time is not performance evidence",
    "baseline: current validated 16-site production E4M3 policy",
    "",
]
for label in labels:
    x = m[label]
    lines.append(
        f"{label:<7} relNLL={x['rel_nll']:+.6f} KL={x['kl_mean']:.6f} "
        f"KL95={x['kl_p95']:.6f} logit_cos={x['logit_cos']:.6f} "
        f"top1={x['top1']:.4f} hidden={x['hidden_nrmse']:.6f}/{x['hidden_cos']:.6f} "
        f"screen={'BASELINE' if label == 'e4m3' else ('PASS' if screen(label) else 'REJECT')}"
    )
lines.extend([
    "",
    f"incremental l8-l9 vs E4M3: relNLL={m['l8-l9']['rel_nll'] - base['rel_nll']:+.6f} "
    f"KL={m['l8-l9']['kl_mean'] - base['kl_mean']:+.6f} "
    f"top1={m['l8-l9']['top1'] - base['top1']:+.4f}",
    f"DECISION: {decision}",
    "NEXT: only a surviving policy may receive an in-process backend; no production merge yet",
])
summary.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[nvfp4-phase2c] summary: ${SUMMARY}"
echo "[nvfp4-phase2c] reports: ${WORK_DIR}/phase2c-{e4m3,l8,l9,l8-l9}.json"
echo "[nvfp4-phase2c] no production src/ changes were committed or merged"
