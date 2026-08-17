use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;
const MAX_HEAD_DIM: usize = 512;

pub(crate) struct RopeLaunch<'a> {
    pub(crate) query: &'a mut CudaSlice<bf16>,
    pub(crate) key: &'a mut CudaSlice<bf16>,
    pub(crate) inv_freq: &'a CudaSlice<f32>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) num_tokens: usize,
    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
}

pub(crate) struct RopeKernels {
    rope_qk_bf16_inplace: KernelLaunch,
}

impl KernelSet for RopeKernels {
    const MODULE_NAME: &'static str = "rope";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "rope_qk_bf16_inplace")?;
        let rope_qk_bf16_inplace = KernelLaunch::new(function, MAX_BLOCK_SIZE)?;
        Ok(Self {
            rope_qk_bf16_inplace,
        })
    }
}

impl RopeKernels {
    pub(crate) unsafe fn launch_qk_bf16_inplace(
        &self,
        stream: &CudaStream,
        launch: RopeLaunch<'_>,
    ) -> Result<()> {
        let RopeLaunch {
            query,
            key,
            inv_freq,
            position_ids,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
        } = launch;

        ensure!(num_tokens > 0, "RoPE requires at least one token");
        ensure!(num_q_heads > 0, "RoPE requires at least one Q head");
        ensure!(num_kv_heads > 0, "RoPE requires at least one KV head");
        ensure!(head_dim > 0, "RoPE head_dim must be > 0");
        ensure!(
            head_dim.is_multiple_of(2),
            "RoPE head_dim must be even, got {head_dim}"
        );
        ensure!(
            head_dim <= MAX_HEAD_DIM,
            "RoPE head_dim={head_dim} exceeds kernel maximum {MAX_HEAD_DIM}"
        );

        let half_dim = head_dim / 2;
        ensure!(
            inv_freq.len() >= half_dim,
            "RoPE inv_freq storage too small: required={half_dim}, actual={}",
            inv_freq.len()
        );
        ensure!(
            position_ids.len() >= num_tokens,
            "RoPE position_ids storage too small: required={num_tokens}, actual={}",
            position_ids.len()
        );

        let query_required = num_tokens
            .checked_mul(num_q_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .context("RoPE query size overflow")?;
        let key_required = num_tokens
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .context("RoPE key size overflow")?;
        ensure!(
            query.len() >= query_required,
            "RoPE query storage too small: required={query_required}, actual={}",
            query.len()
        );
        ensure!(
            key.len() >= key_required,
            "RoPE key storage too small: required={key_required}, actual={}",
            key.len()
        );

        let config = self
            .rope_qk_bf16_inplace
            .policy()
            .exact_blocks(num_tokens)?;
        let mut args = stream.launch_builder(self.rope_qk_bf16_inplace.function());
        args.arg(query)
            .arg(key)
            .arg(inv_freq)
            .arg(position_ids)
            .arg(&num_tokens)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn block_size(&self) -> u32 {
        self.rope_qk_bf16_inplace.policy().block_size()
    }
}
