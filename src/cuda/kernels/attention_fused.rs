use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const BLOCK_SIZE: u32 = 256;

pub(crate) struct FusedAttentionCommon<'a> {
    pub(crate) page_size: usize,
    pub(crate) query_raw: &'a CudaSlice<bf16>,
    pub(crate) key_raw: &'a CudaSlice<bf16>,
    pub(crate) value_raw: &'a CudaSlice<bf16>,
    pub(crate) query_norm: &'a CudaSlice<bf16>,
    pub(crate) key_norm: &'a CudaSlice<bf16>,
    pub(crate) inv_freq: &'a CudaSlice<f32>,
    pub(crate) key_cache: &'a mut CudaSlice<bf16>,
    pub(crate) value_cache: &'a mut CudaSlice<bf16>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
    pub(crate) eps: f32,
}

pub(crate) struct FusedDecodeLaunch<'a> {
    pub(crate) common: FusedAttentionCommon<'a>,
    pub(crate) block_table: &'a CudaSlice<u32>,
}

pub(crate) struct FusedRaggedDecodeLaunch<'a> {
    pub(crate) common: FusedAttentionCommon<'a>,
    pub(crate) block_tables: &'a CudaSlice<u32>,
    pub(crate) request_slots: &'a CudaSlice<u32>,
    pub(crate) block_table_stride: usize,
}

pub(crate) struct FusedAttentionKernels {
    ps16: KernelLaunch,
    ps32: KernelLaunch,
    ragged_ps16: KernelLaunch,
    ragged_ps32: KernelLaunch,
}

impl KernelSet for FusedAttentionKernels {
    const MODULE_NAME: &'static str = "attention_fused";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/attention_fused.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "fused_decode_attention_lfm2_bf16_ps16",
        )?;
        let ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "fused_decode_attention_lfm2_bf16_ps32",
        )?;
        let ragged_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "fused_ragged_decode_attention_lfm2_bf16_ps16",
        )?;
        let ragged_ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "fused_ragged_decode_attention_lfm2_bf16_ps32",
        )?;

        for (name, function) in [
            ("ps16", &ps16),
            ("ps32", &ps32),
            ("ragged_ps16", &ragged_ps16),
            ("ragged_ps32", &ragged_ps32),
        ] {
            ensure!(
                function.max_threads_per_block()? >= BLOCK_SIZE as i32,
                "fused attention {name} cannot launch required 256-thread block"
            );
        }

        Ok(Self {
            ps16: KernelLaunch::new_with_multiple(ps16, BLOCK_SIZE, 32)?,
            ps32: KernelLaunch::new_with_multiple(ps32, BLOCK_SIZE, 32)?,
            ragged_ps16: KernelLaunch::new_with_multiple(ragged_ps16, BLOCK_SIZE, 32)?,
            ragged_ps32: KernelLaunch::new_with_multiple(ragged_ps32, BLOCK_SIZE, 32)?,
        })
    }
}

impl FusedAttentionKernels {
    fn launch_config(blocks: usize) -> Result<LaunchConfig> {
        let grid_x = u32::try_from(blocks).context("fused attention grid size exceeds u32")?;
        Ok(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    pub(crate) unsafe fn launch_decode(
        &self,
        stream: &CudaStream,
        launch: FusedDecodeLaunch<'_>,
    ) -> Result<()> {
        Self::validate_common(&launch.common)?;
        ensure!(
            !launch.block_table.is_empty(),
            "fused attention block table is empty"
        );

        let FusedDecodeLaunch {
            common,
            block_table,
        } = launch;
        let FusedAttentionCommon {
            page_size,
            query_raw,
            key_raw,
            value_raw,
            query_norm,
            key_norm,
            inv_freq,
            key_cache,
            value_cache,
            position_ids,
            slot_mapping,
            output,
            num_tokens,
            num_pages,
            eps,
        } = common;
        let kernel = match page_size {
            16 => &self.ps16,
            32 => &self.ps32,
            other => anyhow::bail!("unsupported fused attention page size {other}"),
        };
        let blocks = num_tokens
            .checked_mul(8)
            .context("fused attention grid size overflow")?;
        let config = Self::launch_config(blocks)?;
        let block_table_length = block_table.len();
        let mut args = stream.launch_builder(kernel.function());
        args.arg(query_raw)
            .arg(key_raw)
            .arg(value_raw)
            .arg(query_norm)
            .arg(key_norm)
            .arg(inv_freq)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_table)
            .arg(position_ids)
            .arg(slot_mapping)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&block_table_length)
            .arg(&eps);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_ragged_decode(
        &self,
        stream: &CudaStream,
        launch: FusedRaggedDecodeLaunch<'_>,
    ) -> Result<()> {
        Self::validate_common(&launch.common)?;
        ensure!(
            launch.block_table_stride > 0,
            "ragged fused block table stride is zero"
        );
        ensure!(
            launch.block_tables.len() >= launch.block_table_stride,
            "ragged fused block table storage too small"
        );
        ensure!(
            launch
                .block_tables
                .len()
                .is_multiple_of(launch.block_table_stride),
            "ragged fused block tables are not row aligned"
        );
        ensure!(
            launch.request_slots.len() >= launch.common.num_tokens,
            "ragged fused request slots too small"
        );

        let FusedRaggedDecodeLaunch {
            common,
            block_tables,
            request_slots,
            block_table_stride,
        } = launch;
        let FusedAttentionCommon {
            page_size,
            query_raw,
            key_raw,
            value_raw,
            query_norm,
            key_norm,
            inv_freq,
            key_cache,
            value_cache,
            position_ids,
            slot_mapping,
            output,
            num_tokens,
            num_pages,
            eps,
        } = common;
        let kernel = match page_size {
            16 => &self.ragged_ps16,
            32 => &self.ragged_ps32,
            other => anyhow::bail!("unsupported ragged fused attention page size {other}"),
        };
        let blocks = num_tokens
            .checked_mul(8)
            .context("ragged fused attention grid size overflow")?;
        let config = Self::launch_config(blocks)?;
        let block_table_rows = block_tables.len() / block_table_stride;
        let mut args = stream.launch_builder(kernel.function());
        args.arg(query_raw)
            .arg(key_raw)
            .arg(value_raw)
            .arg(query_norm)
            .arg(key_norm)
            .arg(inv_freq)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_tables)
            .arg(request_slots)
            .arg(position_ids)
            .arg(slot_mapping)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&block_table_stride)
            .arg(&block_table_rows)
            .arg(&eps);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    fn validate_common(input: &FusedAttentionCommon<'_>) -> Result<()> {
        ensure!(input.num_tokens > 0, "fused attention requires tokens");
        ensure!(input.num_pages > 0, "fused attention requires cache pages");
        ensure!(
            matches!(input.page_size, 16 | 32),
            "unsupported fused page size"
        );
        let q_required = input
            .num_tokens
            .checked_mul(32 * 64)
            .context("fused Q storage overflow")?;
        let kv_required = input
            .num_tokens
            .checked_mul(8 * 64)
            .context("fused KV storage overflow")?;
        let cache_required = input
            .num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(input.page_size))
            .and_then(|value| value.checked_mul(64))
            .context("fused cache storage overflow")?;
        ensure!(
            input.query_raw.len() >= q_required,
            "fused query storage too small"
        );
        ensure!(
            input.key_raw.len() >= kv_required,
            "fused key storage too small"
        );
        ensure!(
            input.value_raw.len() >= kv_required,
            "fused value storage too small"
        );
        ensure!(
            input.query_norm.len() >= 64,
            "fused query norm weight too small"
        );
        ensure!(
            input.key_norm.len() >= 64,
            "fused key norm weight too small"
        );
        ensure!(
            input.inv_freq.len() >= 32,
            "fused RoPE frequency storage too small"
        );
        ensure!(
            input.key_cache.len() >= cache_required,
            "fused K cache too small"
        );
        ensure!(
            input.value_cache.len() >= cache_required,
            "fused V cache too small"
        );
        ensure!(
            input.position_ids.len() >= input.num_tokens,
            "fused positions too small"
        );
        ensure!(
            input.slot_mapping.len() >= input.num_tokens,
            "fused slot mapping too small"
        );
        ensure!(
            input.output.len() >= q_required,
            "fused output storage too small"
        );
        Ok(())
    }
}
