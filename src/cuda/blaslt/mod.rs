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

use plan::{MatmulKey, MatmulPlan};

use fp8::{Fp8MatmulKey, Fp8MatmulPlan, Fp8ScaleMode};

pub(crate) struct BlasLt {
    handle: sys::cublasLtHandle_t,

    stream: Arc<CudaStream>,

    workspace: CudaSlice<u8>,

    workspace_size: usize,

    plans: RwLock<HashMap<MatmulKey, Arc<MatmulPlan>>>,

    fp8_plans: RwLock<HashMap<Fp8MatmulKey, Arc<Fp8MatmulPlan>>>,
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

        let workspace_size = if major >= 9 {
            32 * 1024 * 1024
        } else {
            4 * 1024 * 1024
        };

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

    fn get_or_create_fp8_plan(&self, key: Fp8MatmulKey) -> Result<Arc<Fp8MatmulPlan>> {
        {
            let plans = self
                .fp8_plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;

            if let Some(plan) = plans.get(&key) {
                return Ok(Arc::clone(plan));
            }
        }

        let created = Arc::new(Fp8MatmulPlan::new(
            self.handle,
            &self.stream,
            key,
            self.workspace_size,
        )?);

        let mut plans = self
            .fp8_plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;

        let plan = plans.entry(key).or_insert_with(|| Arc::clone(&created));

        Ok(Arc::clone(plan))
    }

    fn get_or_create_plan(&self, key: MatmulKey) -> Result<Arc<MatmulPlan>> {
        {
            let plans = self
                .plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;

            if let Some(plan) = plans.get(&key) {
                return Ok(Arc::clone(plan));
            }
        }

        /*
         * Do the expensive heuristic work
         * outside the write lock.
         */
        let created = Arc::new(MatmulPlan::new(self.handle, key, self.workspace_size)?);

        let mut plans = self
            .plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;

        /*
         * If another caller inserted the same
         * shape meanwhile, reuse the existing one.
         */
        let plan = plans.entry(key).or_insert_with(|| Arc::clone(&created));

        Ok(Arc::clone(plan))
    }

    pub(crate) fn prepare_linear_bf16(&self, m: usize, n: usize, k: usize) -> Result<()> {
        let key = MatmulKey::new(m, n, k)?;

        self.get_or_create_plan(key)?;

        Ok(())
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

        ensure!(
            x.len() >= x_required,
            "linear input storage too small: \
             required={x_required}, actual={}",
            x.len(),
        );

        ensure!(
            weight.len() >= weight_required,
            "linear weight storage too small: \
             required={weight_required}, actual={}",
            weight.len(),
        );

        ensure!(
            out.len() >= out_required,
            "linear output storage too small: \
             required={out_required}, actual={}",
            out.len(),
        );

        let plan = self.get_or_create_plan(key)?;

        self.stream
            .context()
            .bind_to_thread()
            .context("failed to bind CUDA context for cuBLASLt matmul")?;

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
        let key = Fp8MatmulKey::new(m, n, k, scale_mode)?;

        self.get_or_create_fp8_plan(key)?;

        Ok(())
    }

    #[cfg(test)]
    pub(crate) unsafe fn linear_fp8<X>(
        &self,
        x: &X,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        scale_mode: Fp8ScaleMode,
    ) -> Result<()>
    where
        X: DevicePtr<u8>,
    {
        unsafe { self.linear_fp8_scaled(x, weight, out, m, n, k, scale_mode, 1.0) }
    }

    pub(crate) unsafe fn linear_fp8_scaled<X>(
        &self,
        x: &X,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        scale_mode: Fp8ScaleMode,
        output_scale: f32,
    ) -> Result<()>
    where
        X: DevicePtr<u8>,
    {
        let key = Fp8MatmulKey::new(m, n, k, scale_mode)?;
        ensure!(
            output_scale.is_finite() && output_scale > 0.0,
            "FP8 output scale must be finite and positive"
        );

        let x_required = m.checked_mul(k).context("FP8 linear input size overflow")?;
        let weight_required = n
            .checked_mul(k)
            .context("FP8 linear weight size overflow")?;
        let out_required = m
            .checked_mul(n)
            .context("FP8 linear output size overflow")?;

        ensure!(
            x.len() >= x_required,
            "FP8 linear input storage too small: required={x_required}, actual={}",
            x.len(),
        );
        ensure!(
            weight.len() >= weight_required,
            "FP8 linear weight storage too small: required={weight_required}, actual={}",
            weight.len(),
        );
        ensure!(
            out.len() >= out_required,
            "FP8 linear output storage too small: required={out_required}, actual={}",
            out.len(),
        );

        let plan = self.get_or_create_fp8_plan(key)?;

        self.stream
            .context()
            .bind_to_thread()
            .context("failed to bind CUDA context for cuBLASLt FP8 matmul")?;

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
            Ok(plans) => {
                plans.clear();
            }

            Err(poisoned) => {
                poisoned.into_inner().clear();
            }
        }

        match self.fp8_plans.get_mut() {
            Ok(plans) => {
                plans.clear();
            }
            Err(poisoned) => {
                poisoned.into_inner().clear();
            }
        }

        let handle = mem::replace(&mut self.handle, std::ptr::null_mut());

        if !handle.is_null() {
            unsafe {
                let _ = result::destroy_handle(handle);
            }
        }
    }
}
