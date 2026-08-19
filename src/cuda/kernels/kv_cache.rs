use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const KV_CACHE_LFM2_BLOCK_SIZE: u32 = 256;

pub(crate) struct KvCacheWriteLaunch<'a> {
    pub(crate) page_size: usize,
    pub(crate) key: &'a CudaSlice<bf16>,
    pub(crate) value: &'a CudaSlice<bf16>,
    pub(crate) key_cache: &'a mut CudaSlice<bf16>,
    pub(crate) value_cache: &'a mut CudaSlice<bf16>,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
}

pub(crate) struct KvCacheKernels {
    lfm2_ps16: KernelLaunch,
    lfm2_ps32: KernelLaunch,
}

impl KernelSet for KvCacheKernels {
    const MODULE_NAME: &'static str = "kv_cache";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/kv_cache.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let lfm2_ps16_fn =
            load_function(&module, Self::MODULE_NAME, "kv_cache_write_lfm2_bf16_ps16")?;
        let lfm2_ps32_fn =
            load_function(&module, Self::MODULE_NAME, "kv_cache_write_lfm2_bf16_ps32")?;
        Ok(Self {
            lfm2_ps16: KernelLaunch::new(lfm2_ps16_fn, KV_CACHE_LFM2_BLOCK_SIZE)?,
            lfm2_ps32: KernelLaunch::new(lfm2_ps32_fn, KV_CACHE_LFM2_BLOCK_SIZE)?,
        })
    }
}

impl KvCacheKernels {
    pub(crate) unsafe fn launch_write_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: KvCacheWriteLaunch<'_>,
    ) -> Result<()> {
        let KvCacheWriteLaunch {
            page_size,
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            num_pages,
        } = launch;

        ensure!(num_tokens > 0, "KV cache write requires at least one token");
        ensure!(num_pages > 0, "KV cache requires at least one page");
        let required = num_tokens
            .checked_mul(8 * 64)
            .context("KV input size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("KV cache size overflow")?;
        ensure!(key.len() >= required, "K storage too small");
        ensure!(value.len() >= required, "V storage too small");
        ensure!(
            key_cache.len() >= cache_required,
            "K cache storage too small"
        );
        ensure!(
            value_cache.len() >= cache_required,
            "V cache storage too small"
        );
        ensure!(
            slot_mapping.len() >= num_tokens,
            "slot mapping storage too small"
        );

        let kernel = match page_size {
            16 => &self.lfm2_ps16,
            32 => &self.lfm2_ps32,
            other => anyhow::bail!("unsupported KV page size {other}"),
        };
        let config = kernel.policy().exact_blocks(num_tokens)?;
        let mut args = stream.launch_builder(kernel.function());
        args.arg(key)
            .arg(value)
            .arg(key_cache)
            .arg(value_cache)
            .arg(slot_mapping)
            .arg(&num_tokens)
            .arg(&num_pages);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
