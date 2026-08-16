use std::sync::Arc;

use anyhow::{Context as _, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule},
    nvrtc::Ptx,
};

pub(crate) fn load_module(
    context: &Arc<CudaContext>,
    name: &str,
    ptx: &'static str,
) -> Result<Arc<CudaModule>> {
    context
        .load_module(Ptx::from_src(ptx))
        .with_context(|| format!("failed to load CUDA module `{name}`"))
}

pub(crate) fn load_function(
    module: &Arc<CudaModule>,
    module_name: &str,
    function_name: &str,
) -> Result<CudaFunction> {
    module.load_function(function_name).with_context(|| {
        format!("failed to load CUDA function `{function_name}` from module `{module_name}`")
    })
}
