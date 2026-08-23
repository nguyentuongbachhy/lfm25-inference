use std::{
    collections::HashSet,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

static ACTIVE_SITES: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
static EXPORTED_WEIGHTS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
static INVOCATION_ID: AtomicU64 = AtomicU64::new(0);

fn active_sites() -> &'static Mutex<HashSet<String>> {
    ACTIVE_SITES.get_or_init(|| Mutex::new(HashSet::new()))
}

fn exported_weights() -> &'static Mutex<HashSet<String>> {
    EXPORTED_WEIGHTS.get_or_init(|| Mutex::new(HashSet::new()))
}

pub(crate) fn set_active_sites(sites: Vec<String>) {
    let mut active = match active_sites().lock() {
        Ok(active) => active,
        Err(poisoned) => poisoned.into_inner(),
    };
    active.clear();
    active.extend(sites);
}

pub(crate) fn is_active_site(site: &str) -> bool {
    let active = match active_sites().lock() {
        Ok(active) => active,
        Err(poisoned) => poisoned.into_inner(),
    };
    active.contains(site)
}

fn sanitized_site(site: &str) -> String {
    site.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn write_bf16_file(path: &Path, values: &[bf16]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create NVFP4 replay file {}", path.display()))?;
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(2));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    file.write_all(&bytes)
        .with_context(|| format!("failed to write NVFP4 replay file {}", path.display()))
}

fn read_bf16_file(path: &Path, expected: usize) -> Result<Vec<bf16>> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read NVFP4 replay output {}", path.display()))?;
    ensure!(
        bytes.len() == expected.saturating_mul(2),
        "NVFP4 replay output length mismatch: expected {} bytes, got {}",
        expected.saturating_mul(2),
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

fn replay_work_dir() -> Result<PathBuf> {
    let path = PathBuf::from(
        env::var("NVFP4_REPLAY_WORK_DIR")
            .context("NVFP4_REPLAY_WORK_DIR is required for nvfp4-research")?,
    );
    fs::create_dir_all(&path)
        .with_context(|| format!("failed to create NVFP4 replay dir {}", path.display()))?;
    Ok(path)
}

fn replay_binary() -> Result<PathBuf> {
    let path = PathBuf::from(
        env::var("NVFP4_REPLAY_BIN").context("NVFP4_REPLAY_BIN is required for nvfp4-research")?,
    );
    ensure!(
        path.is_file(),
        "NVFP4 replay binary does not exist: {}",
        path.display()
    );
    Ok(path)
}

fn ensure_weight_exported(
    runtime: &CudaRuntime,
    weight: &Tensor<bf16>,
    site: &str,
    path: &Path,
) -> Result<()> {
    let key = format!("{site}:{}x{}", weight.dims()[0], weight.dims()[1]);
    {
        let exported = match exported_weights().lock() {
            Ok(exported) => exported,
            Err(poisoned) => poisoned.into_inner(),
        };
        if exported.contains(&key) && path.is_file() {
            return Ok(());
        }
    }

    let values = runtime
        .download(weight)
        .with_context(|| format!("failed to read BF16 weight for NVFP4 replay site {site}"))?;
    write_bf16_file(path, &values)?;

    let mut exported = match exported_weights().lock() {
        Ok(exported) => exported,
        Err(poisoned) => poisoned.into_inner(),
    };
    exported.insert(key);
    Ok(())
}

/// Exact, deliberately slow quality-only replay through the validated CUTLASS
/// SM120 NVFP4 kernel. This function must never be used for performance timing.
/// Phase 2B invokes it only at sampled M=1 decode positions.
pub(crate) fn linear_nvfp4_replay(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    site: &str,
) -> Result<Tensor<bf16>> {
    ensure!(input.rank() >= 1, "NVFP4 replay input must have rank >= 1");
    ensure!(weight.rank() == 2, "NVFP4 replay weight must have rank 2");
    let k = input.dims()[input.rank() - 1];
    let n = weight.dims()[0];
    ensure!(weight.dims()[1] == k, "NVFP4 replay K mismatch");
    ensure!(input.numel() == k, "NVFP4 replay is Phase-2B M=1 only");
    ensure!(k.is_multiple_of(128), "NVFP4 replay K must be divisible by 128");
    ensure!(n.is_multiple_of(128), "NVFP4 replay N must be divisible by 128");

    let work_dir = replay_work_dir()?;
    let binary = replay_binary()?;
    let site_slug = sanitized_site(site);
    let weight_path = work_dir.join(format!("{site_slug}-weight-{n}x{k}.bf16"));
    ensure_weight_exported(runtime, weight, site, &weight_path)?;

    let invocation = INVOCATION_ID.fetch_add(1, Ordering::Relaxed);
    let input_path = work_dir.join(format!("{site_slug}-input-{invocation}.bf16"));
    let output_path = work_dir.join(format!("{site_slug}-output-{invocation}.bf16"));
    let input_host = runtime
        .download(input)
        .with_context(|| format!("failed to read NVFP4 replay input for {site}"))?;
    write_bf16_file(&input_path, &input_host)?;

    let result = Command::new(&binary)
        .arg(format!("--site={site}"))
        .arg(format!("--n={n}"))
        .arg(format!("--k={k}"))
        .arg(format!("--weight={}", weight_path.display()))
        .arg(format!("--input={}", input_path.display()))
        .arg(format!("--output={}", output_path.display()))
        .output()
        .with_context(|| format!("failed to launch NVFP4 replay for {site}"))?;

    if !result.status.success() {
        anyhow::bail!(
            "NVFP4 replay failed for {site} (status={}): stdout={} stderr={} weight={} input={} output={}",
            result.status,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
            weight_path.display(),
            input_path.display(),
            output_path.display(),
        );
    }

    let output_host = read_bf16_file(&output_path, n)?;
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
    runtime.upload(&output_host, Shape::new([1, n]))
}
