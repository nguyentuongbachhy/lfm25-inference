#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_PHASE2B_WORK_DIR:-${ROOT}/target/nvfp4-sm120-phase2b}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(realpath "${WORK_DIR}")"
CUTLASS_DIR="${WORK_DIR}/cutlass"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
REPLAY_SOURCE="${ROOT}/research/nvfp4/nvfp4_replay_site.cu"
REPLAY_BIN="${WORK_DIR}/nvfp4-replay-tn8"
REPLAY_DATA="${WORK_DIR}/replay-data"
POLICY_SOURCE="${NVFP4_PHASE2B_FP8_POLICY:-${ROOT}/docs/benchmarks/fp8/selected-policy.json}"
POLICY_JSON="${WORK_DIR}/nvfp4-phase2b-candidates.json"
REPORT_JSON="${WORK_DIR}/nvfp4-phase2b.json"
SUMMARY_TXT="${WORK_DIR}/nvfp4-phase2b-summary.txt"
MODEL="${NVFP4_MODEL:-${ROOT}/models/LFM2.5-1.2B-Instruct}"
SEQUENCES="${NVFP4_PHASE2B_SEQUENCES:-8}"
MAX_TOKENS="${NVFP4_PHASE2B_MAX_TOKENS:-128}"
EXPECTED_GPU_SUBSTRING="${NVFP4_EXPECTED_GPU_SUBSTRING:-RTX 5060 Laptop}"

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/run_nvfp4_phase2b.sh VALIDATION_CORPUS" >&2
  exit 2
fi

VALIDATION_CORPUS="$(realpath "$1")"
MODEL="$(realpath "${MODEL}")"
POLICY_SOURCE="$(realpath "${POLICY_SOURCE}")"

for tool in git nvcc cargo python3 realpath nvidia-smi; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[nvfp4-phase2b] missing required host tool: ${tool}" >&2
    exit 1
  }
done

if [[ "${ROOT}" != "/home/hyy4hc/source/lfm25-inference" && "${NVFP4_ALLOW_ALT_HOST_PATH:-0}" != "1" ]]; then
  echo "[nvfp4-phase2b] refusing to run outside canonical host workspace: ${ROOT}" >&2
  echo "[nvfp4-phase2b] set NVFP4_ALLOW_ALT_HOST_PATH=1 only for an intentional host checkout" >&2
  exit 1
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! grep -Fq "${EXPECTED_GPU_SUBSTRING}" <<<"${GPU_NAMES}"; then
  echo "[nvfp4-phase2b] refusing benchmark/replay environment without expected host GPU" >&2
  echo "[nvfp4-phase2b] expected substring: ${EXPECTED_GPU_SUBSTRING}" >&2
  echo "[nvfp4-phase2b] detected: ${GPU_NAMES:-<none>}" >&2
  exit 1
fi

NVCC_VERSION_OUTPUT="$(nvcc --version)"
CUDA_VERSION="$(printf '%s\n' "${NVCC_VERSION_OUTPUT}" | tail -n 1)"
if ! grep -Eq 'release[[:space:]]+12\.8([,[:space:]]|$)|cuda_12\.8([._/[:space:]]|$)' <<<"${NVCC_VERSION_OUTPUT}"; then
  echo "[nvfp4-phase2b] expected CUDA 12.8.x; nvcc reported:" >&2
  printf '%s\n' "${NVCC_VERSION_OUTPUT}" >&2
  exit 1
fi

BRANCH="$(git -C "${ROOT}" branch --show-current)"
HEAD="$(git -C "${ROOT}" rev-parse HEAD)"
if [[ "${BRANCH}" != "agent/nvfp4-sm120" ]]; then
  echo "[nvfp4-phase2b] checkout agent/nvfp4-sm120 before running; current=${BRANCH}" >&2
  exit 1
fi

if ! git -C "${ROOT}" merge-base --is-ancestor 117c4a66828970344cd757d1e977bd729e891526 HEAD; then
  echo "[nvfp4-phase2b] branch does not contain completed Phase 2A commit 117c4a6" >&2
  exit 1
fi

mkdir -p "${REPLAY_DATA}"
rm -f "${REPORT_JSON}" "${SUMMARY_TXT}"

echo "[nvfp4-phase2b] host workspace: ${ROOT}"
echo "[nvfp4-phase2b] branch/head: ${BRANCH} ${HEAD}"
echo "[nvfp4-phase2b] GPU: ${GPU_NAMES//$'\n'/; }"
echo "[nvfp4-phase2b] nvcc: ${CUDA_VERSION}"
echo "[nvfp4-phase2b] model: ${MODEL}"
echo "[nvfp4-phase2b] validation corpus: ${VALIDATION_CORPUS}"
echo "[nvfp4-phase2b] scope: sampled quality/propagation only; replay wall time is NOT performance evidence"

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
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
    raise SystemExit("[nvfp4-phase2b] unexpected CUTLASS atomic-load source")
PY

echo "[nvfp4-phase2b] compiling exact nearest-only replay kernel"
nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_120a \
  --expt-relaxed-constexpr \
  -diag-suppress=20012 \
  -DNVFP4_TILE_N=8 \
  -I "${CUTLASS_DIR}/include" \
  -I "${CUTLASS_DIR}/tools/util/include" \
  "${REPLAY_SOURCE}" \
  -o "${REPLAY_BIN}"
git -C "${CUTLASS_DIR}" reset --hard HEAD >/dev/null

# The policy is a carrier for the already-validated FP8 evaluation machinery.
# E4M3 scales are retained only so install_fp8_policy can initialize its normal
# toggle path; selected GEMMs are intercepted by the exact CUTLASS NVFP4 replay.
# High-risk Phase-2A sites down{6,8,10} are excluded entirely.
python3 - "${POLICY_SOURCE}" "${POLICY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
policy = json.loads(source.read_text())

metrics = {
    "layers.2.mlp.gate_up": (0.101763, 0.994847, 35.044),
    "layers.3.mlp.gate_up": (0.113162, 0.993625, 35.044),
    "layers.5.mlp.gate_up": (0.105274, 0.994478, 35.044),
    "layers.7.mlp.gate_up": (0.108159, 0.994144, 35.044),
    "layers.8.mlp.gate_up": (0.109813, 0.993976, 35.044),
    "layers.9.mlp.gate_up": (0.106210, 0.994371, 35.044),
    "layers.11.mlp.gate_up": (0.113672, 0.993531, 35.044),
    "layers.15.mlp.gate_up": (0.100662, 0.994993, 35.044),
    "layers.9.mlp.down": (0.119963, 0.993533, 23.571),
    "layers.12.mlp.down": (0.148553, 0.988921, 23.571),
    "layers.14.mlp.down": (0.137574, 0.990493, 23.571),
    "layers.15.mlp.down": (0.105411, 0.994454, 23.571),
    "lm_head": (0.095655, 0.995421, 147.414),
}

by_site = {site["site"]: site for site in policy["sites"]}
missing = sorted(set(metrics) - set(by_site))
if missing:
    raise SystemExit(f"[nvfp4-phase2b] selected FP8 policy is missing candidate sites: {missing}")

sites = []
for name, (nrmse, cosine, saving) in metrics.items():
    site = dict(by_site[name])
    site["enabled"] = True
    site["local_nrmse"] = nrmse
    site["local_cosine"] = cosine
    site["expected_decode_saving_us"] = saving
    sites.append(site)

out.write_text(json.dumps({
    "schema_version": 1,
    "name": "nvfp4_phase2b_nearest_candidates",
    "source": "nvfp4_phase2a_real_checkpoint_nearest_only",
    "decode_only": True,
    "sites": sites,
}, indent=2) + "\n")
print(f"[nvfp4-phase2b] candidate frontier: {len(sites)} sites")
for site in sites:
    print(
        f"  {site['site']:<30} local_nrmse={site['local_nrmse']:.6f} "
        f"cos={site['local_cosine']:.6f} saving_us={site['expected_decode_saving_us']:.3f}"
    )
PY

WORKTREE="$(mktemp -d /tmp/lfm25-nvfp4-phase2b.XXXXXX)"
cleanup() {
  git -C "${ROOT}" worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true
  rm -rf "${WORKTREE}"
}
trap cleanup EXIT

git -C "${ROOT}" worktree add --detach "${WORKTREE}" HEAD >/dev/null
python3 "${ROOT}/research/nvfp4/patch_phase2b_worktree.py" "${WORKTREE}" "${ROOT}"

# Format/check only the temporary research worktree. No production source is
# committed by this script.
(
  cd "${WORKTREE}"
  cargo fmt
  LLM_CUDA_ARCH=compute_120 cargo check --features nvfp4-research
)

export NVFP4_REPLAY_BIN="${REPLAY_BIN}"
export NVFP4_REPLAY_WORK_DIR="${REPLAY_DATA}"
export NVFP4_PHASE2B_MODEL="${MODEL}"
export NVFP4_PHASE2B_CORPUS="${VALIDATION_CORPUS}"
export NVFP4_PHASE2B_POLICY="${POLICY_JSON}"
export NVFP4_PHASE2B_OUTPUT="${REPORT_JSON}"
export NVFP4_PHASE2B_SEQUENCES="${SEQUENCES}"
export NVFP4_PHASE2B_MAX_TOKENS="${MAX_TOKENS}"

echo "[nvfp4-phase2b] running sampled single-site propagation + bounded policy search"
(
  cd "${WORKTREE}"
  LLM_CUDA_ARCH=compute_120 cargo test --release --features nvfp4-research \
    nvfp4_phase2b_sampled_propagation_and_policy_search \
    -- --ignored --nocapture --test-threads=1
)

python3 - "${REPORT_JSON}" "${SUMMARY_TXT}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
if not report_path.is_file():
    raise SystemExit("[nvfp4-phase2b] missing Phase 2B JSON report")
report = json.loads(report_path.read_text())
sites = report["sensitivity"]["sites"]

# Phase-2B one-site screen. LM-head is special: final hidden must remain exact
# because the changed projection occurs after final_rms_norm.
def screen(site):
    finite = True  # non-finites already fail the underlying evaluator.
    if site["site"] == "lm_head":
        hidden_ok = site["final_hidden_nrmse"] <= 1e-7 and site["final_hidden_cosine"] >= 0.999999
    else:
        hidden_ok = site["final_hidden_nrmse"] <= 0.10 and site["final_hidden_cosine"] >= 0.995
    quality_ok = site["mean_logit_kl"] <= 0.05 and site["relative_nll_delta"] <= 0.005
    return finite and hidden_ok and quality_ok

passed = [s for s in sites if screen(s)]
ranked = sorted(
    passed,
    key=lambda s: s["expected_decode_saving_us"] / (s["sensitivity_score"] + 1e-6),
    reverse=True,
)

lines = []
lines.append("NVFP4 Phase 2B decision summary")
lines.append("backend: exact CUTLASS SM120 external replay; wall time is not performance evidence")
lines.append("scale: nearest UE4M3 only; round-up remains rejected")
lines.append("")
lines.append("single-site results:")
for s in sorted(sites, key=lambda x: x["site"]):
    lines.append(
        f"{s['site']:<30} local={s['local_nrmse']:.6f} "
        f"hidden={s['final_hidden_nrmse']:.6f}/{s['final_hidden_cosine']:.6f} "
        f"KL={s['mean_logit_kl']:.6f} relNLL={s['relative_nll_delta']:.6f} "
        f"screen={'PASS' if screen(s) else 'REJECT'}"
    )
lines.append("")
if ranked:
    lines.append(f"viable single-site frontier: {len(ranked)}")
    for s in ranked:
        utility = s["expected_decode_saving_us"] / (s["sensitivity_score"] + 1e-6)
        lines.append(f"  {s['site']} utility={utility:.3f} expected_saving_us={s['expected_decode_saving_us']:.3f}")
else:
    lines.append("viable single-site frontier: 0")
    lines.append("DECISION: reject NVFP4 before in-process production-backend work")

search = report.get("policy_search")
if search is None:
    lines.append(f"policy search: no surviving cumulative policy ({report.get('policy_search_error')})")
else:
    selected = [s["site"] for s in search["selected_policy"]["sites"] if s["enabled"]]
    lines.append(f"policy search selected {len(selected)} sites: {', '.join(selected)}")

summary_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[nvfp4-phase2b] report: ${REPORT_JSON}"
echo "[nvfp4-phase2b] summary: ${SUMMARY_TXT}"
echo "[nvfp4-phase2b] no production src/ changes were committed or merged"
