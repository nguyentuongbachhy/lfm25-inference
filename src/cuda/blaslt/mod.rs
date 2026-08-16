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

use plan::{MatmulKey, MatmulPlan};

use fp8::{Fp8MatmulKey, Fp8MatmulPlan, Fp8ScaleMode};

enum PlanAccess<'a, T> {
    Owner(&'a T),
    Shared(Arc<T>),
}

impl<T> PlanAccess<'_, T> {
    #[inline(always)]
    fn get(&self) -> &T {
        match self {
            Self::Owner(plan) => plan,
            Self::Shared(plan) => plan.as_ref(),
        }
    }
}

type OwnerPlans = Vec<(MatmulKey, Arc<MatmulPlan>)>;
type OwnerFp8Plans = Vec<(Fp8MatmulKey, Arc<Fp8MatmulPlan>)>;

pub(crate) struct BlasLt {
    handle: sys::cublasLtHandle_t,
    stream: Arc<CudaStream>,
    workspace: CudaSlice<u8>,
    workspace_size: usize,
    plans: RwLock<Option<HashMap<MatmulKey, Arc<MatmulPlan>>>>,
    fp8_plans: RwLock<Option<HashMap<Fp8MatmulKey, Arc<Fp8MatmulPlan>>>>,
    owner_plans: UnsafeCell<Option<OwnerPlans>>,
    owner_fp8_plans: UnsafeCell<Option<OwnerFp8Plans>>,
    owner_mode: AtomicBool,
    owner_thread: OnceLock<ThreadId>,
}

// Dynamic setup remains protected by RwLock. Once owner mode is enabled, all
// accesses are restricted to the dedicated GPU-owner thread and use compact
// vectors without locks, hashes, or Arc refcount traffic.
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
            plans: RwLock::new(Some(HashMap::new())),
            fp8_plans: RwLock::new(Some(HashMap::new())),
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

        let mut plans = match self.plans.write() {
            Ok(plans) => plans,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut fp8_plans = match self.fp8_plans.write() {
            Ok(plans) => plans,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut owner_plans = plans
            .take()
            .context("cuBLASLt BF16 plan cache is unavailable")?
            .into_iter()
            .collect::<OwnerPlans>();
        owner_plans.sort_unstable_by_key(|(key, _)| *key);
        let owner_fp8_plans = fp8_plans
            .take()
            .context("cuBLASLt FP8 plan cache is unavailable")?
            .into_iter()
            .collect::<OwnerFp8Plans>();

        // SAFETY: owner mode is not visible until both caches have been moved.
        unsafe {
            *self.owner_plans.get() = Some(owner_plans);
            *self.owner_fp8_plans.get() = Some(owner_fp8_plans);
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

    fn get_or_create_fp8_plan_shared(&self, key: Fp8MatmulKey) -> Result<Arc<Fp8MatmulPlan>> {
        {
            let plans = self
                .fp8_plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt FP8 plan cache poisoned"))?;
            let plans = plans
                .as_ref()
                .context("cuBLASLt FP8 plan cache moved to owner mode")?;
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
        let plans = plans
            .as_mut()
            .context("cuBLASLt FP8 plan cache moved to owner mode")?;
        let plan = plans.entry(key).or_insert_with(|| Arc::clone(&created));
        Ok(Arc::clone(plan))
    }

    fn get_or_create_plan_shared(&self, key: MatmulKey) -> Result<Arc<MatmulPlan>> {
        {
            let plans = self
                .plans
                .read()
                .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
            let plans = plans
                .as_ref()
                .context("cuBLASLt plan cache moved to owner mode")?;
            if let Some(plan) = plans.get(&key) {
                return Ok(Arc::clone(plan));
            }
        }

        let created = Arc::new(MatmulPlan::new(self.handle, key, self.workspace_size)?);

        let mut plans = self
            .plans
            .write()
            .map_err(|_| anyhow!("cuBLASLt plan cache poisoned"))?;
        let plans = plans
            .as_mut()
            .context("cuBLASLt plan cache moved to owner mode")?;
        let plan = plans.entry(key).or_insert_with(|| Arc::clone(&created));
        Ok(Arc::clone(plan))
    }

    fn get_or_create_plan_owner(&self, key: MatmulKey) -> Result<&MatmulPlan> {
        self.assert_owner_thread();
        let search = unsafe {
            (*self.owner_plans.get())
                .as_ref()
                .expect("cuBLASLt owner BF16 cache is unavailable")
                .binary_search_by_key(&key, |(candidate, _)| *candidate)
        };
        let index = match search {
            Ok(index) => index,
            Err(index) => {
                let created = Arc::new(MatmulPlan::new(self.handle, key, self.workspace_size)?);
                // SAFETY: the GPU owner is the only thread mutating this vector.
                unsafe {
                    (*self.owner_plans.get())
                        .as_mut()
                        .expect("cuBLASLt owner BF16 cache is unavailable")
                        .insert(index, (key, created));
                }
                index
            }
        };
        // SAFETY: the returned reference is consumed by the current matmul call
        // before another plan resolution can mutate this owner-only vector.
        Ok(unsafe {
            (*self.owner_plans.get())
                .as_ref()
                .expect("cuBLASLt owner BF16 cache is unavailable")[index]
                .1
                .as_ref()
        })
    }

    fn get_or_create_fp8_plan_owner(&self, key: Fp8MatmulKey) -> Result<&Fp8MatmulPlan> {
        self.assert_owner_thread();
        let found = unsafe {
            (*self.owner_fp8_plans.get())
                .as_ref()
                .expect("cuBLASLt owner FP8 cache is unavailable")
                .iter()
                .position(|(candidate, _)| *candidate == key)
        };
        let index = match found {
            Some(index) => index,
            None => {
                let created = Arc::new(Fp8MatmulPlan::new(
                    self.handle,
                    &self.stream,
                    key,
                    self.workspace_size,
                )?);
                // SAFETY: the GPU owner is the only thread mutating this vector.
                let plans = unsafe {
                    (*self.owner_fp8_plans.get())
                        .as_mut()
                        .expect("cuBLASLt owner FP8 cache is unavailable")
                };
                plans.push((key, created));
                plans.len() - 1
            }
        };
        // SAFETY: the returned reference is consumed before the next owner-only
        // lookup can append to the vector.
        Ok(unsafe {
            (*self.owner_fp8_plans.get())
                .as_ref()
                .expect("cuBLASLt owner FP8 cache is unavailable")[index]
                .1
                .as_ref()
        })
    }

    #[inline]
    fn resolve_plan(&self, key: MatmulKey) -> Result<PlanAccess<'_, MatmulPlan>> {
        if self.owner_mode.load(Ordering::Acquire) {
            Ok(PlanAccess::Owner(self.get_or_create_plan_owner(key)?))
        } else {
            Ok(PlanAccess::Shared(self.get_or_create_plan_shared(key)?))
        }
    }

    #[inline]
    fn resolve_fp8_plan(&self, key: Fp8MatmulKey) -> Result<PlanAccess<'_, Fp8MatmulPlan>> {
        if self.owner_mode.load(Ordering::Acquire) {
            Ok(PlanAccess::Owner(self.get_or_create_fp8_plan_owner(key)?))
        } else {
            Ok(PlanAccess::Shared(
                self.get_or_create_fp8_plan_shared(key)?,
            ))
        }
    }

    #[inline]
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
        self.resolve_plan(key)?;
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
            x.len(),
        );
        ensure!(
            weight.len() >= weight_required,
            "linear weight storage too small: required={weight_required}, actual={}",
            weight.len(),
        );
        ensure!(
            out.len() >= out_required,
            "linear output storage too small: required={out_required}, actual={}",
            out.len(),
        );

        let plan = self.resolve_plan(key)?;
        let plan = plan.get();
        self.bind_context_if_shared()?;

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
        self.resolve_fp8_plan(key)?;
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

        let plan = self.resolve_fp8_plan(key)?;
        let plan = plan.get();
        self.bind_context_if_shared()
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
        if self.owner_mode.load(Ordering::Acquire) {
            self.assert_owner_thread();
            // SAFETY: owner thread has exclusive access.
            return unsafe {
                (*self.owner_plans.get())
                    .as_ref()
                    .map_or(0, Vec::len)
            };
        }
        match self.plans.read() {
            Ok(plans) => plans.as_ref().map_or(0, HashMap::len),
            Err(poisoned) => poisoned.into_inner().as_ref().map_or(0, HashMap::len),
        }
    }

    #[cfg(test)]
    pub(crate) fn cached_fp8_plan_count(&self) -> usize {
        if self.owner_mode.load(Ordering::Acquire) {
            self.assert_owner_thread();
            // SAFETY: owner thread has exclusive access.
            return unsafe {
                (*self.owner_fp8_plans.get())
                    .as_ref()
                    .map_or(0, Vec::len)
            };
        }
        match self.fp8_plans.read() {
            Ok(plans) => plans.as_ref().map_or(0, HashMap::len),
            Err(poisoned) => poisoned.into_inner().as_ref().map_or(0, HashMap::len),
        }
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        if self.owner_mode.load(Ordering::Relaxed) {
            // SAFETY: drop happens after owner execution has stopped.
            unsafe {
                if let Some(plans) = (*self.owner_plans.get()).as_mut() {
                    plans.clear();
                }
                if let Some(plans) = (*self.owner_fp8_plans.get()).as_mut() {
                    plans.clear();
                }
            }
        } else {
            match self.plans.get_mut() {
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
            match self.fp8_plans.get_mut() {
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
