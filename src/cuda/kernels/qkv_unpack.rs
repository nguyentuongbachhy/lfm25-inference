use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const BLOCK_SIZE: u32 = 256;
const Q_WIDTH: usize = 32 * 64;
const KV_WIDTH: usize = 8 * 64;
const PACKED_WIDTH: usize = Q_WIDTH + 2 * KV_WIDTH;

pub(crate) struct QkvUnpackLaunch<'a> {
    pub(crate) packed: &'a CudaSlice<bf16>,
    pub(crate) query: &'a mut CudaSlice<bf16>,
    pub(crate) key: &'a mut CudaSlice<bf16>,
    pub(crate) value: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
}

pub(crate) struct QkvUnpackKernels {
    bf16: KernelLaunch,
}

impl KernelSet for QkvUnpackKernels {
    const MODULE_NAME: &'static str = "qkv_unpack";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/qkv_unpack.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "unpack_qkv_bf16")?;
        Ok(Self {
            bf16: KernelLaunch::new(function, BLOCK_SIZE)?,
        })
    }
}

impl QkvUnpackKernels {
    pub(crate) unsafe fn launch_bf16(
        &self,
        stream: &CudaStream,
        launch: QkvUnpackLaunch<'_>,
    ) -> Result<()> {
        let QkvUnpackLaunch {
            packed,
            query,
            key,
            value,
            num_tokens,
        } = launch;
        ensure!(num_tokens > 0, "packed QKV unpack requires tokens");
        let packed_required = num_tokens
            .checked_mul(PACKED_WIDTH)
            .context("packed QKV size overflow")?;
        let query_required = num_tokens
            .checked_mul(Q_WIDTH)
            .context("packed QKV query size overflow")?;
        let kv_required = num_tokens
            .checked_mul(KV_WIDTH)
            .context("packed QKV KV size overflow")?;
        ensure!(packed.len() >= packed_required, "packed QKV input too small");
        ensure!(query.len() >= query_required, "packed QKV query output too small");
        ensure!(key.len() >= kv_required, "packed QKV key output too small");
        ensure!(value.len() >= kv_required, "packed QKV value output too small");

        let blocks = u32::try_from(packed_required.div_ceil(BLOCK_SIZE as usize))
            .context("packed QKV grid exceeds u32")?;
        let config = LaunchConfig {
            grid_dim: (blocks.max(1), 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = stream.launch_builder(self.bf16.function());
        args.arg(packed)
            .arg(query)
            .arg(key)
            .arg(value)
            .arg(&num_tokens);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
