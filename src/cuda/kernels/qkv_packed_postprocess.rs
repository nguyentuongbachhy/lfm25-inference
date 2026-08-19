use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const BLOCK_SIZE: u32 = 256;
const Q_ELEMENTS: usize = 32 * 64;
const KV_ELEMENTS: usize = 8 * 64;
const QKV_ELEMENTS: usize = Q_ELEMENTS + 2 * KV_ELEMENTS;

pub(crate) struct PackedQkvPostprocessLaunch<'a> {
    pub(crate) page_size: usize,
    pub(crate) packed_qkv: &'a CudaSlice<bf16>,
    pub(crate) query_out: &'a mut CudaSlice<bf16>,
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

pub(crate) struct PackedQkvPostprocessKernels {
    ps16: KernelLaunch,
    ps32: KernelLaunch,
}

impl KernelSet for PackedQkvPostprocessKernels {
    const MODULE_NAME: &'static str = "qkv_packed_postprocess";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/qkv_packed_postprocess.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let ps16 = load_function(&module, Self::MODULE_NAME, "packed_qkv_postprocess_ps16")?;
        let ps32 = load_function(&module, Self::MODULE_NAME, "packed_qkv_postprocess_ps32")?;
        for (name, function) in [("ps16", &ps16), ("ps32", &ps32)] {
            ensure!(
                function.max_threads_per_block()? >= BLOCK_SIZE as i32,
                "packed QKV postprocess {name} cannot launch required 256-thread block"
            );
        }
        Ok(Self {
            ps16: KernelLaunch::new_with_multiple(ps16, BLOCK_SIZE, 32)?,
            ps32: KernelLaunch::new_with_multiple(ps32, BLOCK_SIZE, 32)?,
        })
    }
}

impl PackedQkvPostprocessKernels {
    pub(crate) unsafe fn launch_decode(
        &self,
        stream: &CudaStream,
        launch: PackedQkvPostprocessLaunch<'_>,
    ) -> Result<()> {
        let PackedQkvPostprocessLaunch {
            page_size,
            packed_qkv,
            query_out,
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
        ensure!(num_tokens > 0, "packed QKV postprocess requires tokens");
        ensure!(num_pages > 0, "packed QKV postprocess requires cache pages");
        let packed_required = num_tokens
            .checked_mul(QKV_ELEMENTS)
            .context("packed QKV storage overflow")?;
        let query_required = num_tokens
            .checked_mul(Q_ELEMENTS)
            .context("packed Q output storage overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("packed QKV cache storage overflow")?;
        ensure!(packed_qkv.len() >= packed_required, "packed QKV storage too small");
        ensure!(query_out.len() >= query_required, "packed Q output storage too small");
        ensure!(query_norm.len() >= 64, "query norm weight too small");
        ensure!(key_norm.len() >= 64, "key norm weight too small");
        ensure!(inv_freq.len() >= 32, "RoPE frequency storage too small");
        ensure!(position_ids.len() >= num_tokens, "position storage too small");
        ensure!(slot_mapping.len() >= num_tokens, "slot mapping storage too small");
        ensure!(key_cache.len() >= cache_required, "K cache storage too small");
        ensure!(value_cache.len() >= cache_required, "V cache storage too small");

        let kernel = match page_size {
            16 => &self.ps16,
            32 => &self.ps32,
            other => anyhow::bail!("unsupported packed QKV page size {other}"),
        };
        let grid_x = u32::try_from(num_tokens).context("packed QKV grid size exceeds u32")?;
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = stream.launch_builder(kernel.function());
        args.arg(packed_qkv)
            .arg(query_out)
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
