#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${NVFP4_WORK_DIR:-${ROOT}/target/nvfp4-sm120}"
CUTLASS_DIR="${WORK_DIR}/cutlass"
BRIDGE_BIN="${WORK_DIR}/nvfp4-checkpoint-tn64"
BRIDGE_SOURCE="${ROOT}/research/nvfp4/nvfp4_checkpoint_site.cu"
FULL_LOG="${WORK_DIR}/nvfp4-checkpoint-full.log"
RESULT_LOG="${WORK_DIR}/nvfp4-checkpoint.log"
CALIBRATION_JSON="${WORK_DIR}/nvfp4-checkpoint-fp8-calibration.json"
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
MODEL="${NVFP4_MODEL:-${ROOT}/models/LFM2.5-1.2B-Instruct}"
MAX_SEQUENCES="${NVFP4_CHECKPOINT_MAX_SEQUENCES:-256}"
MAX_TOKENS="${NVFP4_CHECKPOINT_MAX_TOKENS:-1024}"
EVAL_SEQUENCES="${NVFP4_CHECKPOINT_EVAL_SEQUENCES:-16}"
EVAL_TOKENS="${NVFP4_CHECKPOINT_EVAL_TOKENS:-128}"
SITES="${NVFP4_CHECKPOINT_SITES:-layers.2.mlp.gate_up,layers.3.mlp.gate_up,layers.5.mlp.gate_up,layers.7.mlp.gate_up,layers.8.mlp.gate_up,layers.9.mlp.gate_up,layers.11.mlp.gate_up,layers.15.mlp.gate_up,layers.6.mlp.down,layers.8.mlp.down,layers.9.mlp.down,layers.10.mlp.down,layers.12.mlp.down,layers.14.mlp.down,layers.15.mlp.down,lm_head}"

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/check_nvfp4_checkpoint.sh CALIBRATION_CORPUS VALIDATION_CORPUS" >&2
  exit 2
fi

CALIBRATION_CORPUS="$(realpath "$1")"
VALIDATION_CORPUS="$(realpath "$2")"
MODEL="$(realpath "${MODEL}")"
mkdir -p "${WORK_DIR}"

for tool in git nvcc cargo python3 realpath; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[nvfp4-checkpoint] missing required tool: ${tool}" >&2
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
    raise SystemExit("[nvfp4-checkpoint] unexpected CUTLASS atomic-load source")
path.write_text(text)
PY

echo "[nvfp4-checkpoint] compiling CUTLASS bridge tileN=64"
nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_120a \
  --expt-relaxed-constexpr \
  -diag-suppress=20012 \
  -DNVFP4_TILE_N=64 \
  -I "${CUTLASS_DIR}/include" \
  -I "${CUTLASS_DIR}/tools/util/include" \
  "${BRIDGE_SOURCE}" \
  -o "${BRIDGE_BIN}"
git -C "${CUTLASS_DIR}" reset --hard HEAD >/dev/null

WORKTREE="$(mktemp -d /tmp/lfm25-nvfp4-checkpoint.XXXXXX)"
cleanup() {
  git -C "${ROOT}" worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true
  rm -rf "${WORKTREE}"
}
trap cleanup EXIT

git -C "${ROOT}" worktree add --detach "${WORKTREE}" agent/nvfp4-sm120 >/dev/null

python3 - "${WORKTREE}/src/model/fp8_analysis.rs" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

imports = '''use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

'''
marker = "use anyhow::{Context as _, Result, ensure};\n"
if marker not in text:
    raise SystemExit("[nvfp4-checkpoint] failed to locate fp8_analysis imports")
text = text.replace(marker, imports + marker, 1)

helpers = r'''
fn write_nvfp4_bf16_file(path: &Path, values: &[bf16]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create NVFP4 bridge file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut bytes = Vec::with_capacity(131_072);
    for chunk in values.chunks(65_536) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        writer
            .write_all(&bytes)
            .with_context(|| format!("failed to write NVFP4 bridge file {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush NVFP4 bridge file {}", path.display()))
}

fn nvfp4_site_requested(site: &str) -> bool {
    env::var("NVFP4_CHECKPOINT_SITES")
        .ok()
        .is_some_and(|sites| sites.split(',').any(|candidate| candidate.trim() == site))
}

fn maybe_characterize_nvfp4_checkpoint(
    runtime: &CudaRuntime,
    site: &str,
    weight: &Tensor<bf16>,
    samples: &[bf16],
    rows: usize,
    feature_size: usize,
    reference: &[bf16],
) -> Result<()> {
    let Some(binary) = env::var_os("NVFP4_CHECKPOINT_BIN") else {
        return Ok(());
    };
    if !nvfp4_site_requested(site) {
        return Ok(());
    }

    let n = weight.dims()[0];
    let root = env::var_os("NVFP4_CHECKPOINT_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("lfm25-nvfp4-checkpoint-data"));
    fs::create_dir_all(&root)
        .with_context(|| format!("failed to create NVFP4 bridge directory {}", root.display()))?;
    let safe_site = site
        .chars()
        .map(|character| if character.is_ascii_alphanumeric() { character } else { '_' })
        .collect::<String>();
    let prefix = root.join(format!("{}-{}", safe_site, std::process::id()));
    let weight_path = prefix.with_extension("weight.bf16");
    let input_path = prefix.with_extension("input.bf16");
    let reference_path = prefix.with_extension("reference.bf16");

    let result = (|| -> Result<()> {
        let weight_host = runtime
            .download(weight)
            .with_context(|| format!("failed to read BF16 weight for NVFP4 site {site}"))?;
        write_nvfp4_bf16_file(&weight_path, &weight_host)?;
        drop(weight_host);
        write_nvfp4_bf16_file(&input_path, samples)?;
        write_nvfp4_bf16_file(&reference_path, reference)?;

        let status = Command::new(&binary)
            .arg(format!("--site={site}"))
            .arg(format!("--rows={rows}"))
            .arg(format!("--n={n}"))
            .arg(format!("--k={feature_size}"))
            .arg(format!("--weight={}", weight_path.display()))
            .arg(format!("--input={}", input_path.display()))
            .arg(format!("--reference={}", reference_path.display()))
            .status()
            .with_context(|| format!("failed to launch NVFP4 bridge for {site}"))?;
        ensure!(status.success(), "NVFP4 bridge failed for {site}: {status}");
        Ok(())
    })();

    for path in [&weight_path, &input_path, &reference_path] {
        let _ = fs::remove_file(path);
    }
    result
}

'''
fn_marker = "pub(crate) fn characterize_gemm_site(\n"
if fn_marker not in text:
    raise SystemExit("[nvfp4-checkpoint] failed to locate characterize_gemm_site")
text = text.replace(fn_marker, helpers + fn_marker, 1)

old = '''    let reference_host = runtime
        .download(&reference)
        .with_context(|| format!("failed to read BF16 reference for {site}"))?;
    let activation_candidates = collector
'''
new = '''    let reference_host = runtime
        .download(&reference)
        .with_context(|| format!("failed to read BF16 reference for {site}"))?;
    maybe_characterize_nvfp4_checkpoint(
        runtime,
        &site,
        weight,
        samples,
        rows,
        feature_size,
        &reference_host,
    )?;
    let activation_candidates = collector
'''
if old not in text:
    raise SystemExit("[nvfp4-checkpoint] failed to locate BF16 reference block")
text = text.replace(old, new, 1)
path.write_text(text)
PY

(
  cd "${WORKTREE}"
  cargo fmt
  cargo check --all-features
  cargo clippy --all-targets --all-features -- -D warnings
)

export NVFP4_CHECKPOINT_BIN="${BRIDGE_BIN}"
export NVFP4_CHECKPOINT_SITES="${SITES}"
export NVFP4_CHECKPOINT_TMP="${WORK_DIR}/checkpoint-tmp"
export CARGO_TARGET_DIR="${ROOT}/target/nvfp4-checkpoint-cargo"
mkdir -p "${NVFP4_CHECKPOINT_TMP}"

echo "[nvfp4-checkpoint] model: ${MODEL}"
echo "[nvfp4-checkpoint] calibration corpus: ${CALIBRATION_CORPUS}"
echo "[nvfp4-checkpoint] validation corpus: ${VALIDATION_CORPUS}"
echo "[nvfp4-checkpoint] selected sites: 16"

(
  cd "${WORKTREE}"
  cargo run --release -- \
    --model "${MODEL}" \
    --calibrate-fp8 "${CALIBRATION_CORPUS}" \
    --fp8-eval-corpus "${VALIDATION_CORPUS}" \
    --calibration-output "${CALIBRATION_JSON}" \
    --calibration-max-sequences "${MAX_SEQUENCES}" \
    --calibration-max-tokens "${MAX_TOKENS}" \
    --fp8-eval-sequences "${EVAL_SEQUENCES}" \
    --fp8-eval-max-tokens "${EVAL_TOKENS}"
) 2>&1 | tee "${FULL_LOG}"

grep '^nvfp4_checkpoint ' "${FULL_LOG}" > "${RESULT_LOG}" || true

python3 - "${RESULT_LOG}" "${SITES}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
requested = [site for site in sys.argv[2].split(",") if site]
rows = []
for line in path.read_text().splitlines() if path.exists() else []:
    fields = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    if fields:
        rows.append(fields)

seen = {row.get("site") for row in rows}
missing = [site for site in requested if site not in seen]
if missing:
    raise SystemExit(f"[nvfp4-checkpoint] missing bridge results: {','.join(missing)}")

print("[nvfp4-checkpoint] real-checkpoint local summary")
for row in rows:
    nrmse = float(row["nrmse"])
    cosine = float(row["cosine"])
    non_finite = int(row["non_finite"])
    if non_finite:
        verdict = "reject_nonfinite"
    elif nrmse <= 0.10 and cosine >= 0.995:
        verdict = "strong"
    elif nrmse <= 0.15 and cosine >= 0.98:
        verdict = "candidate"
    else:
        verdict = "high_risk"
    extra = ""
    if row["site"] == "lm_head":
        extra = (
            f" top1={float(row['top1_agreement']):.4f}"
            f" top5={float(row['top5_overlap']):.4f}"
            f" top10={float(row['top10_overlap']):.4f}"
            f" kl={float(row['mean_kl']):.6f}"
        )
    print(
        f"nvfp4_checkpoint_summary site={row['site']} verdict={verdict} "
        f"nrmse={nrmse:.6f} cosine={cosine:.6f}{extra}"
    )
PY

echo "[nvfp4-checkpoint] logs:"
echo "  ${RESULT_LOG}"
echo "  ${FULL_LOG}"
