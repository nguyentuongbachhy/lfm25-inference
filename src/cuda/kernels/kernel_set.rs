use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaModule};

use crate::cuda::module::load_module;

pub(crate) trait KernelSet: Sized {
    const MODULE_NAME: &'static str;
    const PTX: &'static str;

    fn from_module(module: Arc<CudaModule>) -> Result<Self>;

    fn load(context: &Arc<CudaContext>) -> Result<Self> {
        let module = load_module(context, Self::MODULE_NAME, Self::PTX)?;

        Self::from_module(module)
    }
}
