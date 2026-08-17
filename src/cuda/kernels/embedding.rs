use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct EmbeddingLaunch<'a> {
    pub(crate) token_ids: &'a CudaSlice<u32>,
    pub(crate) weight: &'a CudaSlice<bf16>,
    pub(crate) out: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
}

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
        launch: EmbeddingLaunch<'_>,
    ) -> Result<()> {
        let EmbeddingLaunch {
            token_ids,
            weight,
            out,
            num_tokens,
            vocab_size,
            hidden_size,
        } = launch;
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
