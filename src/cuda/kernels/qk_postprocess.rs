use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const BLOCK_SIZE: u32 = 256;
const Q_WIDTH: usize = 32 * 64;
const KV_WIDTH: usize = 8 * 64;
const PACKED_QKV_WIDTH: usize = Q_WIDTH + 2 * KV_WIDTH;

pub(crate) struct QkPostprocessLaunch<'a> {
    pub(crate) page_size: usize,
    pub(crate) query: &'a mut CudaSlice<bf16>,
    pub(crate) key: &'a CudaSlice<bf16>,
    pub(crate) value: &'a CudaSlice<bf16>,
    pub(crate) query_norm: &'a CudaSlice<bf16>,
    pub(crate) key_norm: &'a CudaSlice<bf16>,
    pub(crate) inv_freq: &'a CudaSlice<f32>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) key_cache: &'a mut CudaSlice<bf16>,
    pub(crate) value_cache: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
    pub(crate) eps: f32,
}

pub(crate) struct QkPackedPostprocessLaunch<'a> {
    pub(crate) page_size: usize,
    pub(crate) packed_qkv: &'a CudaSlice<bf16>,
    pub(crate) query: &'a mut CudaSlice<bf16>,
    pub(crate) query_norm: &'a CudaSlice<bf16>,
    pub(crate) key_norm: &'a CudaSlice<bf16>,
    pub(crate) inv_freq: &'a CudaSlice<f32>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) key_cache: &'a mut CudaSlice<bf16>,
    pub(crate) value_cache: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
    pub(crate) eps: f32,
}

pub(crate) struct QkPostprocessKernels {
    ps16: KernelLaunch,
    ps32: KernelLaunch,
    packed_ps16: KernelLaunch,
    packed_ps32: KernelLaunch,
}

impl KernelSet for QkPostprocessKernels {
    const MODULE_NAME: &'static str = "qk_postprocess";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/qk_postprocess.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "qk_norm_rope_kv_write_decode_ps16",
        )?;
        let ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "qk_norm_rope_kv_write_decode_ps32",
        )?;
        let packed_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "qk_norm_rope_kv_write_packed_decode_ps16",
        )?;
        let packed_ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "qk_norm_rope_kv_write_packed_decode_ps32",
        )?;

        for (name, function) in [
            ("PS16 fused QK", &ps16),
            ("PS32 fused QK", &ps32),
            ("PS16 packed QK", &packed_ps16),
            ("PS32 packed QK", &packed_ps32),
        ] {
            ensure!(
                function.max_threads_per_block()? >= BLOCK_SIZE as i32,
                "{name} kernel cannot launch required 256-thread block"
            );
        }

        Ok(Self {
            ps16: KernelLaunch::new_with_multiple(ps16, BLOCK_SIZE, 32)?,
            ps32: KernelLaunch::new_with_multiple(ps32, BLOCK_SIZE, 32)?,
            packed_ps16: KernelLaunch::new_with_multiple(packed_ps16, BLOCK_SIZE, 32)?,
            packed_ps32: KernelLaunch::new_with_multiple(packed_ps32, BLOCK_SIZE, 32)?,
        })
    }
}

impl QkPostprocessKernels {
    pub(crate) unsafe fn launch_decode(
        &self,
        stream: &CudaStream,
        launch: QkPostprocessLaunch<'_>,
    ) -> Result<()> {
        let QkPostprocessLaunch {
            page_size,
            query,
            key,
            value,
            query_norm,
            key_norm,
            inv_freq,
            position_ids,
            slot_mapping,
            key_cache,
            value_cache,
            num_tokens,
            num_pages,
            eps,
        } = launch;
        ensure!(num_tokens > 0, "fused QK postprocess requires tokens");
        ensure!(num_pages > 0, "fused QK postprocess requires cache pages");
        let q_required = num_tokens
            .checked_mul(Q_WIDTH)
            .context("fused Q storage overflow")?;
        let kv_required = num_tokens
            .checked_mul(KV_WIDTH)
            .context("fused KV storage overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("fused cache storage overflow")?;

        ensure!(query.len() >= q_required, "fused Q storage too small");
        ensure!(key.len() >= kv_required, "fused K storage too small");
        ensure!(value.len() >= kv_required, "fused V storage too small");
        validate_common(
            query_norm,
            key_norm,
            inv_freq,
            position_ids,
            slot_mapping,
            key_cache,
            value_cache,
            num_tokens,
            cache_required,
        )?;

        let kernel = match page_size {
            16 => &self.ps16,
            32 => &self.ps32,
            other => anyhow::bail!("unsupported fused QK page size {other}"),
        };
        let config = launch_config(num_tokens)?;
        let mut args = stream.launch_builder(kernel.function());
        args.arg(query)
            .arg(key)
            .arg(value)
            .arg(query_norm)
            .arg(key_norm)
            .arg(inv_freq)
            .arg(position_ids)
            .arg(slot_mapping)
            .arg(key_cache)
            .arg(value_cache)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&eps);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_packed_decode(
        &self,
        stream: &CudaStream,
        launch: QkPackedPostprocessLaunch<'_>,
    ) -> Result<()> {
        let QkPackedPostprocessLaunch {
            page_size,
            packed_qkv,
            query,
            query_norm,
            key_norm,
            inv_freq,
            position_ids,
            slot_mapping,
            key_cache,
            value_cache,
            num_tokens,
            num_pages,
            eps,
        } = launch;
        ensure!(num_tokens > 0, "packed QK postprocess requires tokens");
        ensure!(num_pages > 0, "packed QK postprocess requires cache pages");
        let packed_required = num_tokens
            .checked_mul(PACKED_QKV_WIDTH)
            .context("packed QKV storage overflow")?;
        let q_required = num_tokens
            .checked_mul(Q_WIDTH)
            .context("packed Q storage overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("packed cache storage overflow")?;
        ensure!(
            packed_qkv.len() >= packed_required,
            "packed QKV storage too small"
        );
        ensure!(query.len() >= q_required, "packed Q output storage too small");
        validate_common(
            query_norm,
            key_norm,
            inv_freq,
            position_ids,
            slot_mapping,
            key_cache,
            value_cache,
            num_tokens,
            cache_required,
        )?;

        let kernel = match page_size {
            16 => &self.packed_ps16,
            32 => &self.packed_ps32,
            other => anyhow::bail!("unsupported packed QK page size {other}"),
        };
        let config = launch_config(num_tokens)?;
        let mut args = stream.launch_builder(kernel.function());
        args.arg(packed_qkv)
            .arg(query)
            .arg(query_norm)
            .arg(key_norm)
            .arg(inv_freq)
            .arg(position_ids)
            .arg(slot_mapping)
            .arg(key_cache)
            .arg(value_cache)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&eps);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    query_norm: &CudaSlice<bf16>,
    key_norm: &CudaSlice<bf16>,
    inv_freq: &CudaSlice<f32>,
    position_ids: &CudaSlice<u32>,
    slot_mapping: &CudaSlice<i64>,
    key_cache: &CudaSlice<bf16>,
    value_cache: &CudaSlice<bf16>,
    num_tokens: usize,
    cache_required: usize,
) -> Result<()> {
    ensure!(query_norm.len() >= 64, "query norm weight too small");
    ensure!(key_norm.len() >= 64, "key norm weight too small");
    ensure!(inv_freq.len() >= 32, "RoPE frequency storage too small");
    ensure!(
        position_ids.len() >= num_tokens,
        "position storage too small"
    );
    ensure!(
        slot_mapping.len() >= num_tokens,
        "slot mapping storage too small"
    );
    ensure!(
        key_cache.len() >= cache_required,
        "K cache storage too small"
    );
    ensure!(
        value_cache.len() >= cache_required,
        "V cache storage too small"
    );
    Ok(())
}

fn launch_config(num_tokens: usize) -> Result<LaunchConfig> {
    let grid_x = u32::try_from(num_tokens).context("fused QK grid size exceeds u32")?;
    Ok(LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}
