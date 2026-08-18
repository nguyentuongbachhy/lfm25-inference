pub(crate) mod fp8;
mod plan;

use std::{
    collections::HashMap,
    ffi::c_void,
    mem,
    sync::{Arc, RwLock},
};

use anyhow::{Context as _, Result, anyhow, ensure};
use cudarc::{
    cublaslt::{result, sys},
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, sys::CUdevice_attribute},
};
use half::bf16;

use fp8::{Fp8MatmulKey, Fp8MatmulPlan, Fp8ScaleMode};
use plan::{MatmulKey, MatmulPlan};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Fp8LinearConfig {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) scale_mode: Fp8ScaleMode,
    pub(crate) output_scale: f32,
}

pub(crate) struct BlasLt {
    handle: sys::cublasLtHandle_t,
    stream: Arc<CudaStream>,
    workspace: CudaSlice<u8>,
    workspace_size: usize,
    plans: RwLock<HashMap<MatmulKey, MatmulPlan>>,
    fp8_plans: RwLock<HashMap<Fp8MatmulKey, Fp8MatmulPlan>>,
}

impl BlasLt {
    pub(crate) fn workspace_size(&self) -> usize {
        self.workspace_size
    }

    pub(crate) fn new(stream: Arc<CudaStream>) -> Result<Self> {
        stream
            .context()
            .bind_to_thread()
            .context("failed to bind CUDA context for cuBLASLt")?;
        let major = stream
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .context("failed to query GPU compute capability")?;
        let workspace_size = if major >= 9 { 32 * 1024 * 1024 } else { 4 * 1024 * 1024 };
        let workspace = unsafe { stream.alloc::<u8>(workspace_size) }
            .context("failed to allocate cuBLASLt workspace")?;
        let handle = result::create_handle().context("failed to create cuBLASLt handle")?;
        Ok(Self {
            handle,
            stream,
            workspace,
            workspace_size,
            plans: RwLock::new(HashMap::new()),
            fp8_plans: RwLock::new(HashMap::new()),
        })
    }

    fn ensure_fp8_plan(&self, key: Fp8MatmulKey) -> Result<()> {
        {
            let plans = self.fp8_plans.read().map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
            if plans.contains_key(&key) {
                return Ok(());
            }
        }
        let created = Fp8MatmulPlan::new(self.handle, &self.stream, key, self.workspace_size)?;
        let mut plans = self.fp8_plans.write().map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
        plans.entry(key).or_insert(created);
        Ok(())
    }

    fn ensure_plan(&self, key: MatmulKey) -> Result<()> {
        {
            let plans = self.plans.read().map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
            if plans.contains_key(&key) {
                return Ok(());
            }
        }
        let created = MatmulPlan::new(self.handle, key, self.workspace_size)?;
        let mut plans = self.plans.write().map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
        plans.entry(key).or_insert(created);
        Ok(())
    }

    pub(crate) fn prepare_linear_bf16(&self, m: usize, n: usize, k: usize) -> Result<()> {
        self.ensure_plan(MatmulKey::new(m, n, k)?)
    }

    pub(crate) unsafe fn linear_bf16<X>(
        &self,
        x: &X,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>
    where
        X: DevicePtr<bf16>,
    {
        let key = MatmulKey::new(m, n, k)?;
        let x_required = m.checked_mul(k).context("linear input size overflow")?;
        let weight_required = n.checked_mul(k).context("linear weight size overflow")?;
        let out_required = m.checked_mul(n).context("linear output size overflow")?;
        ensure!(x.len() >= x_required, "linear input storage too small: required={x_required}, actual={}", x.len());
        ensure!(weight.len() >= weight_required, "linear weight storage too small: required={weight_required}, actual={}", weight.len());
        ensure!(out.len() >= out_required, "linear output storage too small: required={out_required}, actual={}", out.len());

        self.ensure_plan(key)?;
        let plans = self.plans.read().map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
        let plan = plans.get(&key).ok_or_else(|| anyhow!("cuBLASLt plan missing after preparation"))?;

        self.stream.context().bind_to_thread().context("failed to bind CUDA context for cuBLASLt matmul")?;
        let (x_ptr, _x_record) = x.device_ptr(&self.stream);
        let (weight_ptr, _weight_record) = weight.device_ptr(&self.stream);
        let (out_ptr, _out_record) = out.device_ptr_mut(&self.stream);
        let (workspace_ptr, _workspace_record) = self.workspace.device_ptr(&self.stream);
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            result::matmul(
                self.handle,
                plan.desc(),
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                x_ptr as *const c_void,
                plan.a_layout(),
                weight_ptr as *const c_void,
                plan.b_layout(),
                out_ptr as *const c_void,
                plan.c_layout(),
                out_ptr as *mut c_void,
                plan.c_layout(),
                plan.algo() as *const _,
                workspace_ptr as *mut c_void,
                self.workspace_size,
                self.stream.cu_stream() as *mut _,
            )
            .context("cached cuBLASLt BF16 matmul failed")?;
        }
        Ok(())
    }

    pub(crate) fn prepare_linear_fp8(
        &self,
        m: usize,
        n: usize,
        k: usize,
        scale_mode: Fp8ScaleMode,
    ) -> Result<()> {
        self.ensure_fp8_plan(Fp8MatmulKey::new(m, n, k, scale_mode)?)
    }

    #[cfg(test)]
    pub(crate) unsafe fn linear_fp8<X>(
        &self,
        x: &X,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
        config: Fp8LinearConfig,
    ) -> Result<()>
    where
        X: DevicePtr<u8>,
    {
        unsafe { self.linear_fp8_scaled(x, weight, out, config) }
    }

    pub(crate) unsafe fn linear_fp8_scaled<X>(
        &self,
        x: &X,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
        config: Fp8LinearConfig,
    ) -> Result<()>
    where
        X: DevicePtr<u8>,
    {
        let Fp8LinearConfig { m, n, k, scale_mode, output_scale } = config;
        let key = Fp8MatmulKey::new(m, n, k, scale_mode)?;
        ensure!(output_scale.is_finite() && output_scale > 0.0, "FP8 output scale must be finite and positive");
        let x_required = m.checked_mul(k).context("FP8 linear input size overflow")?;
        let weight_required = n.checked_mul(k).context("FP8 linear weight size overflow")?;
        let out_required = m.checked_mul(n).context("FP8 linear output size overflow")?;
        ensure!(x.len() >= x_required, "FP8 linear input storage too small: required={x_required}, actual={}", x.len());
        ensure!(weight.len() >= weight_required, "FP8 linear weight storage too small: required={weight_required}, actual={}", weight.len());
        ensure!(out.len() >= out_required, "FP8 linear output storage too small: required={out_required}, actual={}", out.len());

        self.ensure_fp8_plan(key)?;
        let plans = self.fp8_plans.read().map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
        let plan = plans.get(&key).ok_or_else(|| anyhow!("cuBLASLt FP8 plan missing after preparation"))?;

        self.stream.context().bind_to_thread().context("failed to bind CUDA context for cuBLASLt FP8 matmul")?;
        let (x_ptr, _x_record) = x.device_ptr(&self.stream);
        let (weight_ptr, _weight_record) = weight.device_ptr(&self.stream);
        let (out_ptr, _out_record) = out.device_ptr_mut(&self.stream);
        let (workspace_ptr, _workspace_record) = self.workspace.device_ptr(&self.stream);
        let alpha = output_scale;
        let beta = 0.0f32;
        unsafe {
            result::matmul(
                self.handle,
                plan.desc(),
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                x_ptr as *const c_void,
                plan.a_layout(),
                weight_ptr as *const c_void,
                plan.b_layout(),
                out_ptr as *const c_void,
                plan.c_layout(),
                out_ptr as *mut c_void,
                plan.c_layout(),
                plan.algo() as *const _,
                workspace_ptr as *mut c_void,
                self.workspace_size,
                self.stream.cu_stream() as *mut _,
            )
            .context("cached cuBLASLt FP8 matmul failed")?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn cached_plan_count(&self) -> usize {
        match self.plans.read() {
            Ok(plans) => plans.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    #[cfg(test)]
    pub(crate) fn cached_fp8_plan_count(&self) -> usize {
        match self.fp8_plans.read() {
            Ok(plans) => plans.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        match self.plans.get_mut() {
            Ok(plans) => plans.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
        match self.fp8_plans.get_mut() {
            Ok(plans) => plans.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
        let handle = mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { let _ = result::destroy_handle(handle); }
        }
    }
}
