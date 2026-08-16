use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct EmbeddingKernels {
    embedding_bf16: KernelLaunch,
}

impl KernelSet for EmbeddingKernels {
    const MODULE_NAME: &'static str = "embedding";

    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/embedding.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "embedding_bf16")?;

        Ok(Self {
            embedding_bf16: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
        })
    }
}

impl EmbeddingKernels {
    pub(crate) unsafe fn launch_bf16(
        &self,
        stream: &CudaStream,
        token_ids: &CudaSlice<u32>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        num_tokens: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        let config = self.embedding_bf16.policy().exact_blocks(num_tokens)?;

        let mut args = stream.launch_builder(self.embedding_bf16.function());

        args.arg(token_ids)
            .arg(weight)
            .arg(out)
            .arg(&num_tokens)
            .arg(&vocab_size)
            .arg(&hidden_size);

        unsafe {
            args.launch(config)?;
        }

        Ok(())
    }
}
