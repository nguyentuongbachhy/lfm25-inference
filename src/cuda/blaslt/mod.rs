pub(crate) mod fp8;
mod plan;

use std::{
    cell::UnsafeCell,
    collections::HashMap,
    ffi::c_void,
    mem,
    sync::{
        Arc, OnceLock, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread::ThreadId,
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
    shared_plans: RwLock<Option<HashMap<MatmulKey, MatmulPlan>>>,
    shared_fp8_plans: RwLock<Option<HashMap<Fp8MatmulKey, Fp8MatmulPlan>>>,
    owner_plans: UnsafeCell<Option<HashMap<MatmulKey, MatmulPlan>>>,
    owner_fp8_plans: UnsafeCell<Option<HashMap<Fp8MatmulKey, Fp8MatmulPlan>>>,
    owner_mode: AtomicBool,
    owner_thread: OnceLock<ThreadId>,
}

// Dynamic setup is synchronized through the shared maps. Owner mode is a
// one-way transition performed by the dedicated GPU thread after serving
// warmup; after that transition the owner maps have exactly one accessor.
unsafe impl Sync for BlasLt {}

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
            shared_plans: RwLock::new(Some(HashMap::new())),
            shared_fp8_plans: RwLock::new(Some(HashMap::new())),
            owner_plans: UnsafeCell::new(None),
            owner_fp8_plans: UnsafeCell::new(None),
            owner_mode: AtomicBool::new(false),
            owner_thread: OnceLock::new(),
        })
    }

    pub(crate) fn enter_owner_mode(&self) -> Result<()> {
        ensure!(
            !self.owner_mode.load(Ordering::Acquire),
            "cuBLASLt owner mode is already enabled"
        );
        self.stream
            .context()
            .bind_to_thread()
            .context("failed to bind CUDA context for cuBLASLt owner")?;
        self.owner_thread
            .set(std::thread::current().id())
            .map_err(|_| anyhow!("cuBLASLt owner thread is already set"))?;

        let mut shared_plans = self
            .shared_plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
        let mut shared_fp8_plans = self
            .shared_fp8_plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
        let plans = shared_plans
            .take()
            .context("cuBLASLt BF16 plan cache is unavailable")?;
        let fp8_plans = shared_fp8_plans
            .take()
            .context("cuBLASLt FP8 plan cache is unavailable")?;

        // SAFETY: owner_mode is still false, so owner caches are unreachable.
        unsafe {
            *self.owner_plans.get() = Some(plans);
            *self.owner_fp8_plans.get() = Some(fp8_plans);
        }
        self.owner_mode.store(true, Ordering::Release);
        Ok(())
    }

    #[inline(always)]
    fn assert_owner_thread(&self) {
        #[cfg(debug_assertions)]
        {
            let current = std::thread::current().id();
            debug_assert!(
                self.owner_thread
                    .get()
                    .is_some_and(|owner| *owner == current),
                "cuBLASLt owner cache accessed from a non-owner thread"
            );
        }
    }

    fn ensure_plan_shared(&self, key: MatmulKey) -> Result<()> {
        {
            let plans = self
                .shared_plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
            let plans = plans
                .as_ref()
                .context("cuBLASLt plan cache moved to owner mode")?;
            if plans.contains_key(&key) {
                return Ok(());
            }
        }
        let created = MatmulPlan::new(self.handle, key, self.workspace_size)?;
        let mut plans = self
            .shared_plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
        let plans = plans
            .as_mut()
            .context("cuBLASLt plan cache moved to owner mode")?;
        plans.entry(key).or_insert(created);
        Ok(())
    }

    fn ensure_fp8_plan_shared(&self, key: Fp8MatmulKey) -> Result<()> {
        {
            let plans = self
                .shared_fp8_plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
            let plans = plans
                .as_ref()
                .context("cuBLASLt FP8 plan cache moved to owner mode")?;
            if plans.contains_key(&key) {
                return Ok(());
            }
        }
        let created = Fp8MatmulPlan::new(self.handle, &self.stream, key, self.workspace_size)?;
        let mut plans = self
            .shared_fp8_plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
        let plans = plans
            .as_mut()
            .context("cuBLASLt FP8 plan cache moved to owner mode")?;
        plans.entry(key).or_insert(created);
        Ok(())
    }

    fn get_or_create_plan_owner(&self, key: MatmulKey) -> Result<&MatmulPlan> {
        self.assert_owner_thread();
        let exists = unsafe {
            (*self.owner_plans.get())
                .as_ref()
                .expect("cuBLASLt owner BF16 cache is unavailable")
                .contains_key(&key)
        };
        if !exists {
            let created = MatmulPlan::new(self.handle, key, self.workspace_size)?;
            unsafe {
                (*self.owner_plans.get())
                    .as_mut()
                    .expect("cuBLASLt owner BF16 cache is unavailable")
                    .insert(key, created);
            }
        }
        Ok(unsafe {
            (*self.owner_plans.get())
                .as_ref()
                .expect("cuBLASLt owner BF16 cache is unavailable")
                .get(&key)
                .expect("cuBLASLt owner BF16 plan missing after insertion")
        })
    }

    fn get_or_create_fp8_plan_owner(&self, key: Fp8MatmulKey) -> Result<&Fp8MatmulPlan> {
        self.assert_owner_thread();
        let exists = unsafe {
            (*self.owner_fp8_plans.get())
                .as_ref()
                .expect("cuBLASLt owner FP8 cache is unavailable")
                .contains_key(&key)
        };
        if !exists {
            let created = Fp8MatmulPlan::new(self.handle, &self.stream, key, self.workspace_size)?;
            unsafe {
                (*self.owner_fp8_plans.get())
                    .as_mut()
                    .expect("cuBLASLt owner FP8 cache is unavailable")
                    .insert(key, created);
            }
        }
        Ok(unsafe {
            (*self.owner_fp8_plans.get())
                .as_ref()
                .expect("cuBLASLt owner FP8 cache is unavailable")
                .get(&key)
                .expect("cuBLASLt owner FP8 plan missing after insertion")
        })
    }

    #[inline(always)]
    fn bind_context_if_shared(&self) -> Result<()> {
        if self.owner_mode.load(Ordering::Relaxed) {
            self.assert_owner_thread();
            return Ok(());
        }
        self.stream
            .context()
            .bind_to_thread()
            .context("failed to bind CUDA context for cuBLASLt matmul")
    }

    pub(crate) fn prepare_linear_bf16(&self, m: usize, n: usize, k: usize) -> Result<()> {
        let key = MatmulKey::new(m, n, k)?;
        if self.owner_mode.load(Ordering::Acquire) {
            self.get_or_create_plan_owner(key)?;
            Ok(())
        } else {
            self.ensure_plan_shared(key)
        }
    }

    #[inline(always)]
    unsafe fn launch_bf16<X>(
        &self,
        plan: &MatmulPlan,
        x: &X,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()>
    where
        X: DevicePtr<bf16>,
    {
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
            "linear input storage too small: required={x_required}, actual={}",
            x.len()
        );
        ensure!(
            weight.len() >= weight_required,
            "linear weight storage too small: required={weight_required}, actual={}",
            weight.len()
        );
        ensure!(
            out.len() >= out_required,
            "linear output storage too small: required={out_required}, actual={}",
            out.len()
        );

        self.bind_context_if_shared()?;
        if self.owner_mode.load(Ordering::Acquire) {
            let plan = self.get_or_create_plan_owner(key)?;
            unsafe { self.launch_bf16(plan, x, weight, out) }
        } else {
            self.ensure_plan_shared(key)?;
            let plans = self
                .shared_plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
            let plan = plans
                .as_ref()
                .context("cuBLASLt plan cache moved to owner mode")?
                .get(&key)
                .ok_or_else(|| anyhow!("cuBLASLt plan missing after preparation"))?;
            unsafe { self.launch_bf16(plan, x, weight, out) }
        }
    }

    pub(crate) fn prepare_linear_fp8(
        &self,
        m: usize,
        n: usize,
        k: usize,
        scale_mode: Fp8ScaleMode,
    ) -> Result<()> {
        let key = Fp8MatmulKey::new(m, n, k, scale_mode)?;
        if self.owner_mode.load(Ordering::Acquire) {
            self.get_or_create_fp8_plan_owner(key)?;
            Ok(())
        } else {
            self.ensure_fp8_plan_shared(key)
        }
    }

    #[inline(always)]
    unsafe fn launch_fp8<X>(
        &self,
        plan: &Fp8MatmulPlan,
        x: &X,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
        output_scale: f32,
    ) -> Result<()>
    where
        X: DevicePtr<u8>,
    {
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
        let Fp8LinearConfig {
            m,
            n,
            k,
            scale_mode,
            output_scale,
        } = config;
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
            x.len()
        );
        ensure!(
            weight.len() >= weight_required,
            "FP8 linear weight storage too small: required={weight_required}, actual={}",
            weight.len()
        );
        ensure!(
            out.len() >= out_required,
            "FP8 linear output storage too small: required={out_required}, actual={}",
            out.len()
        );

        self.bind_context_if_shared()
            .context("failed to bind CUDA context for cuBLASLt FP8 matmul")?;
        if self.owner_mode.load(Ordering::Acquire) {
            let plan = self.get_or_create_fp8_plan_owner(key)?;
            unsafe { self.launch_fp8(plan, x, weight, out, output_scale) }
        } else {
            self.ensure_fp8_plan_shared(key)?;
            let plans = self
                .shared_fp8_plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
            let plan = plans
                .as_ref()
                .context("cuBLASLt FP8 plan cache moved to owner mode")?
                .get(&key)
                .ok_or_else(|| anyhow!("cuBLASLt FP8 plan missing after preparation"))?;
            unsafe { self.launch_fp8(plan, x, weight, out, output_scale) }
        }
    }

    #[cfg(test)]
    pub(crate) fn cached_plan_count(&self) -> usize {
        if self.owner_mode.load(Ordering::Acquire) {
            self.assert_owner_thread();
            return unsafe {
                (*self.owner_plans.get())
                    .as_ref()
                    .map_or(0, HashMap::len)
            };
        }
        match self.shared_plans.read() {
            Ok(plans) => plans.as_ref().map_or(0, HashMap::len),
            Err(poisoned) => poisoned.into_inner().as_ref().map_or(0, HashMap::len),
        }
    }

    #[cfg(test)]
    pub(crate) fn cached_fp8_plan_count(&self) -> usize {
        if self.owner_mode.load(Ordering::Acquire) {
            self.assert_owner_thread();
            return unsafe {
                (*self.owner_fp8_plans.get())
                    .as_ref()
                    .map_or(0, HashMap::len)
            };
        }
        match self.shared_fp8_plans.read() {
            Ok(plans) => plans.as_ref().map_or(0, HashMap::len),
            Err(poisoned) => poisoned.into_inner().as_ref().map_or(0, HashMap::len),
        }
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        if self.owner_mode.load(Ordering::Relaxed) {
            unsafe {
                if let Some(plans) = (*self.owner_plans.get()).as_mut() {
                    plans.clear();
                }
                if let Some(plans) = (*self.owner_fp8_plans.get()).as_mut() {
                    plans.clear();
                }
            }
        } else {
            match self.shared_plans.get_mut() {
                Ok(plans) => {
                    if let Some(plans) = plans.as_mut() {
                        plans.clear();
                    }
                }
                Err(poisoned) => {
                    if let Some(plans) = poisoned.into_inner().as_mut() {
                        plans.clear();
                    }
                }
            }
            match self.shared_fp8_plans.get_mut() {
                Ok(plans) => {
                    if let Some(plans) = plans.as_mut() {
                        plans.clear();
                    }
                }
                Err(poisoned) => {
                    if let Some(plans) = poisoned.into_inner().as_mut() {
                        plans.clear();
                    }
                }
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
