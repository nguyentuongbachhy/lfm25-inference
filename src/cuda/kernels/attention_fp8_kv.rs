use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const BLOCK_SIZE: u32 = 256;
const PAGE_SIZE: usize = 16;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;

pub(crate) struct Fp8KvQuantizeLaunch<'a> {
    pub(crate) key_cache: &'a CudaSlice<bf16>,
    pub(crate) value_cache: &'a CudaSlice<bf16>,
    pub(crate) key_fp8: &'a mut CudaSlice<u8>,
    pub(crate) value_fp8: &'a mut CudaSlice<u8>,
    pub(crate) key_scales: &'a mut CudaSlice<f32>,
    pub(crate) value_scales: &'a mut CudaSlice<f32>,
    pub(crate) num_pages: usize,
}

pub(crate) struct Fp8KvAttentionLaunch<'a> {
    pub(crate) query: &'a CudaSlice<bf16>,
    pub(crate) key_cache: &'a CudaSlice<u8>,
    pub(crate) value_cache: &'a CudaSlice<u8>,
    pub(crate) key_scales: &'a CudaSlice<f32>,
    pub(crate) value_scales: &'a CudaSlice<f32>,
    pub(crate) block_table: &'a CudaSlice<u32>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
}

pub(crate) struct Fp8KvKernels {
    quantize_ps16: KernelLaunch,
    attention_ps16: KernelLaunch,
}

impl KernelSet for Fp8KvKernels {
    const MODULE_NAME: &'static str = "attention_fp8_kv";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/attention_fp8_kv.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let quantize_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "quantize_paged_kv_lfm2_e4m3_ps16",
        )?;
        let attention_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_gqa_lfm2_fp8_kv_ps16",
        )?;
        for (name, function) in [
            ("quantize_ps16", &quantize_ps16),
            ("attention_ps16", &attention_ps16),
        ] {
            ensure!(
                function.max_threads_per_block()? >= BLOCK_SIZE as i32,
                "FP8 KV {name} cannot launch required 256-thread block"
            );
        }
        Ok(Self {
            quantize_ps16: KernelLaunch::new_with_multiple(quantize_ps16, BLOCK_SIZE, 32)?,
            attention_ps16: KernelLaunch::new_with_multiple(attention_ps16, BLOCK_SIZE, 32)?,
        })
    }
}

impl Fp8KvKernels {
    fn launch_config(blocks: usize) -> Result<LaunchConfig> {
        let grid_x = u32::try_from(blocks).context("FP8 KV grid size exceeds u32")?;
        Ok(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    pub(crate) unsafe fn launch_quantize_ps16(
        &self,
        stream: &CudaStream,
        launch: Fp8KvQuantizeLaunch<'_>,
    ) -> Result<()> {
        let Fp8KvQuantizeLaunch {
            key_cache,
            value_cache,
            key_fp8,
            value_fp8,
            key_scales,
            value_scales,
            num_pages,
        } = launch;
        ensure!(num_pages > 0, "FP8 KV quantization requires cache pages");
        let cache_elements = num_pages
            .checked_mul(NUM_KV_HEADS)
            .and_then(|value| value.checked_mul(PAGE_SIZE))
            .and_then(|value| value.checked_mul(HEAD_DIM))
            .context("FP8 KV cache size overflow")?;
        let scale_elements = num_pages
            .checked_mul(NUM_KV_HEADS)
            .context("FP8 KV scale size overflow")?;
        ensure!(key_cache.len() >= cache_elements, "BF16 K cache too small");
        ensure!(value_cache.len() >= cache_elements, "BF16 V cache too small");
        ensure!(key_fp8.len() >= cache_elements, "FP8 K cache too small");
        ensure!(value_fp8.len() >= cache_elements, "FP8 V cache too small");
        ensure!(key_scales.len() >= scale_elements, "FP8 K scales too small");
        ensure!(
            value_scales.len() >= scale_elements,
            "FP8 V scales too small"
        );

        let config = Self::launch_config(scale_elements)?;
        let mut args = stream.launch_builder(self.quantize_ps16.function());
        args.arg(key_cache)
            .arg(value_cache)
            .arg(key_fp8)
            .arg(value_fp8)
            .arg(key_scales)
            .arg(value_scales)
            .arg(&num_pages);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_attention_ps16(
        &self,
        stream: &CudaStream,
        launch: Fp8KvAttentionLaunch<'_>,
    ) -> Result<()> {
        let Fp8KvAttentionLaunch {
            query,
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            block_table,
            position_ids,
            output,
            num_tokens,
            num_pages,
        } = launch;
        ensure!(num_tokens > 0, "FP8 KV attention requires tokens");
        ensure!(num_pages > 0, "FP8 KV attention requires cache pages");
        ensure!(!block_table.is_empty(), "FP8 KV block table is empty");
        let query_elements = num_tokens
            .checked_mul(32 * HEAD_DIM)
            .context("FP8 KV query size overflow")?;
        let cache_elements = num_pages
            .checked_mul(NUM_KV_HEADS)
            .and_then(|value| value.checked_mul(PAGE_SIZE))
            .and_then(|value| value.checked_mul(HEAD_DIM))
            .context("FP8 KV cache size overflow")?;
        let scale_elements = num_pages
            .checked_mul(NUM_KV_HEADS)
            .context("FP8 KV scale size overflow")?;
        ensure!(query.len() >= query_elements, "FP8 KV query storage too small");
        ensure!(output.len() >= query_elements, "FP8 KV output storage too small");
        ensure!(key_cache.len() >= cache_elements, "FP8 K cache storage too small");
        ensure!(
            value_cache.len() >= cache_elements,
            "FP8 V cache storage too small"
        );
        ensure!(key_scales.len() >= scale_elements, "FP8 K scales too small");
        ensure!(
            value_scales.len() >= scale_elements,
            "FP8 V scales too small"
        );
        ensure!(
            position_ids.len() >= num_tokens,
            "FP8 KV position storage too small"
        );

        let blocks = num_tokens
            .checked_mul(NUM_KV_HEADS)
            .context("FP8 KV attention grid size overflow")?;
        let config = Self::launch_config(blocks)?;
        let block_table_length = block_table.len();
        let mut args = stream.launch_builder(self.attention_ps16.function());
        args.arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(key_scales)
            .arg(value_scales)
            .arg(block_table)
            .arg(position_ids)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&block_table_length);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
