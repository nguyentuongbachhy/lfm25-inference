use std::{ffi::c_void, mem::size_of};

use anyhow::{Context as _, Result, ensure};
use cudarc::{
    cublaslt::{result, sys},
    driver::{CudaSlice, DevicePtr, DevicePtrMut},
};
use half::bf16;

use crate::cuda::{
    CudaRuntime,
    benchmark::{BenchConfig, benchmark_gpu_paired},
};

use super::BlasLt;

const MAX_CANDIDATES: usize = 16;
const BENCH: BenchConfig = BenchConfig {
    warmup: 12,
    batches: 32,
    iterations_per_batch: 20,
};

struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn row_major(data_type: sys::cudaDataType_t, rows: usize, cols: usize) -> Result<Self> {
        let rows_u64 = u64::try_from(rows).context("paired autotune rows exceed u64")?;
        let cols_u64 = u64::try_from(cols).context("paired autotune cols exceed u64")?;
        let rows_i64 = i64::try_from(rows).context("paired autotune rows exceed i64")?;
        let row_ld = i64::try_from(cols).context("paired autotune leading dimension exceeds i64")?;
        let create_ld = rows_i64.max(row_ld);
        let raw = result::create_matrix_layout(data_type, rows_u64, cols_u64, create_ld)
            .context("failed to create paired autotune layout")?;
        let layout = Self { raw };
        let order = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW;
        unsafe {
            result::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order as *const _ as *const c_void,
                size_of::<sys::cublasLtOrder_t>(),
            )?;
            if create_ld != row_ld {
                result::set_matrix_layout_attribute(
                    layout.raw,
                    sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_LD,
                    &row_ld as *const _ as *const c_void,
                    size_of::<i64>(),
                )?;
            }
        }
        Ok(layout)
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matrix_layout(self.raw);
        }
    }
}

struct MatmulDesc {
    raw: sys::cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    fn new() -> Result<Self> {
        let raw = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;
        let desc = Self { raw };
        let trans_a = 0_i32;
        let trans_b = 1_i32;
        unsafe {
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                size_of::<i32>(),
            )?;
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                size_of::<i32>(),
            )?;
        }
        Ok(desc)
    }

    fn set_tensorwide_fp8_scales(&self, a: u64, b: u64) -> Result<()> {
        unsafe {
            result::set_matmul_desc_attribute(
                self.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &a as *const u64 as *const c_void,
                size_of::<u64>(),
            )?;
            result::set_matmul_desc_attribute(
                self.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &b as *const u64 as *const c_void,
                size_of::<u64>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_desc(self.raw);
        }
    }
}

struct Preference {
    raw: sys::cublasLtMatmulPreference_t,
}

impl Preference {
    fn new(workspace_size: usize) -> Result<Self> {
        let raw = result::create_matmul_pref()?;
        let pref = Self { raw };
        unsafe {
            result::set_matmul_pref_attribute(
                pref.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size as *const usize as *const c_void,
                size_of::<usize>(),
            )?;
        }
        Ok(pref)
    }
}

impl Drop for Preference {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_pref(self.raw);
        }
    }
}

struct Problem {
    desc: MatmulDesc,
    a: MatrixLayout,
    b: MatrixLayout,
    c: MatrixLayout,
    candidates: Vec<sys::cublasLtMatmulHeuristicResult_t>,
    _a_scale: Option<CudaSlice<f32>>,
    _b_scale: Option<CudaSlice<f32>>,
}

fn problem(blas: &BlasLt, m: usize, n: usize, k: usize, fp8: bool) -> Result<Problem> {
    let input_type = if fp8 {
        sys::cudaDataType_t::CUDA_R_8F_E4M3
    } else {
        sys::cudaDataType_t::CUDA_R_16BF
    };
    let a = MatrixLayout::row_major(input_type, m, k)?;
    let b = MatrixLayout::row_major(input_type, n, k)?;
    let c = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_16BF, m, n)?;
    let desc = MatmulDesc::new()?;
    let (a_scale, b_scale) = if fp8 {
        let a_scale = blas.stream.clone_htod(&[1.0f32])?;
        let b_scale = blas.stream.clone_htod(&[1.0f32])?;
        let (a_ptr, _) = a_scale.device_ptr(&blas.stream);
        let (b_ptr, _) = b_scale.device_ptr(&blas.stream);
        desc.set_tensorwide_fp8_scales(a_ptr, b_ptr)?;
        (Some(a_scale), Some(b_scale))
    } else {
        (None, None)
    };
    let preference = Preference::new(blas.workspace_size)?;
    let mut candidates = (0..MAX_CANDIDATES)
        .map(|_| unsafe { std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>() })
        .collect::<Vec<_>>();
    let mut returned = 0_i32;
    unsafe {
        sys::cublasLtMatmulAlgoGetHeuristic(
            blas.handle,
            desc.raw,
            a.raw,
            b.raw,
            c.raw,
            c.raw,
            preference.raw,
            i32::try_from(MAX_CANDIDATES)?,
            candidates.as_mut_ptr(),
            &mut returned,
        )
        .result()?;
    }
    candidates.truncate(usize::try_from(returned)?);
    candidates.retain(|candidate| {
        candidate.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            && candidate.workspaceSize <= blas.workspace_size
    });
    ensure!(!candidates.is_empty(), "no legal paired autotune candidates");
    Ok(Problem {
        desc,
        a,
        b,
        c,
        candidates,
        _a_scale: a_scale,
        _b_scale: b_scale,
    })
}

unsafe fn launch<T: cudarc::driver::DeviceRepr>(
    blas: &BlasLt,
    problem: &Problem,
    candidate: usize,
    x: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(candidate < problem.candidates.len(), "candidate index out of range");
    let (x_ptr, _) = x.device_ptr(&blas.stream);
    let (weight_ptr, _) = weight.device_ptr(&blas.stream);
    let (out_ptr, _) = out.device_ptr_mut(&blas.stream);
    let (workspace_ptr, _) = blas.workspace.device_ptr(&blas.stream);
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        result::matmul(
            blas.handle,
            problem.desc.raw,
            &alpha as *const f32 as *const c_void,
            &beta as *const f32 as *const c_void,
            x_ptr as *const c_void,
            problem.a.raw,
            weight_ptr as *const c_void,
            problem.b.raw,
            out_ptr as *const c_void,
            problem.c.raw,
            out_ptr as *mut c_void,
            problem.c.raw,
            &problem.candidates[candidate].algo as *const _,
            workspace_ptr as *mut c_void,
            blas.workspace_size,
            blas.stream.cu_stream() as *mut _,
        )?;
    }
    Ok(())
}

fn bf16_data(elements: usize, mul: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % 257) as f32 - 128.0) / 128.0))
        .collect()
}

fn fp8_data(elements: usize, mul: usize) -> Vec<u8> {
    // E4M3 +1.0 = 0x38 and -1.0 = 0xB8. Non-zero input makes the
    // exact-output check meaningful instead of benchmarking an all-zero GEMM.
    (0..elements)
        .map(|index| if (index * mul) & 1 == 0 { 0x38 } else { 0xB8 })
        .collect()
}

fn paired_bf16(runtime: &CudaRuntime, n: usize, k: usize, candidate: usize) -> Result<()> {
    let m = 1usize;
    let blas = runtime.blaslt();
    let problem = problem(blas, m, n, k, false)?;
    ensure!(candidate < problem.candidates.len(), "requested BF16 candidate is unavailable");
    let x = blas.stream.clone_htod(&bf16_data(m * k, 37))?;
    let weight = blas.stream.clone_htod(&bf16_data(n * k, 17))?;
    let mut reference_out = blas.stream.alloc_zeros::<bf16>(m * n)?;
    let mut candidate_out = blas.stream.alloc_zeros::<bf16>(m * n)?;
    unsafe {
        launch(blas, &problem, 0, &x, &weight, &mut reference_out)?;
        launch(blas, &problem, candidate, &x, &weight, &mut candidate_out)?;
    }
    blas.stream.synchronize()?;
    let exact = blas.stream.clone_dtoh(&reference_out)? == blas.stream.clone_dtoh(&candidate_out)?;
    let stats = benchmark_gpu_paired(
        blas.stream.context(),
        &blas.stream,
        BENCH,
        || unsafe { launch(blas, &problem, 0, &x, &weight, &mut reference_out) },
        || unsafe { launch(blas, &problem, candidate, &x, &weight, &mut candidate_out) },
    )?;
    println!(
        "blaslt_autotune_paired dtype=bf16 M=1 N={} K={} candidate={} reference_mean_us={:.3} candidate_mean_us={:.3} speedup_mean={:.4}x reference_p95_us={:.3} candidate_p95_us={:.3} speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x exact={}",
        n, k, candidate, stats.reference.mean_us, stats.candidate.mean_us, stats.speedup_mean,
        stats.reference.p95_us, stats.candidate.p95_us, stats.speedup_p95,
        stats.speedup_min, stats.speedup_max, exact,
    );
    Ok(())
}

fn paired_fp8(runtime: &CudaRuntime, n: usize, k: usize, candidate: usize) -> Result<()> {
    let m = 1usize;
    let blas = runtime.blaslt();
    let problem = problem(blas, m, n, k, true)?;
    ensure!(candidate < problem.candidates.len(), "requested FP8 candidate is unavailable");
    let x = blas.stream.clone_htod(&fp8_data(m * k, 3))?;
    let weight = blas.stream.clone_htod(&fp8_data(n * k, 5))?;
    let mut reference_out = blas.stream.alloc_zeros::<bf16>(m * n)?;
    let mut candidate_out = blas.stream.alloc_zeros::<bf16>(m * n)?;
    unsafe {
        launch(blas, &problem, 0, &x, &weight, &mut reference_out)?;
        launch(blas, &problem, candidate, &x, &weight, &mut candidate_out)?;
    }
    blas.stream.synchronize()?;
    let exact = blas.stream.clone_dtoh(&reference_out)? == blas.stream.clone_dtoh(&candidate_out)?;
    let stats = benchmark_gpu_paired(
        blas.stream.context(),
        &blas.stream,
        BENCH,
        || unsafe { launch(blas, &problem, 0, &x, &weight, &mut reference_out) },
        || unsafe { launch(blas, &problem, candidate, &x, &weight, &mut candidate_out) },
    )?;
    println!(
        "blaslt_autotune_paired dtype=fp8 M=1 N={} K={} candidate={} reference_mean_us={:.3} candidate_mean_us={:.3} speedup_mean={:.4}x reference_p95_us={:.3} candidate_p95_us={:.3} speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x exact={}",
        n, k, candidate, stats.reference.mean_us, stats.candidate.mean_us, stats.speedup_mean,
        stats.reference.p95_us, stats.candidate.p95_us, stats.speedup_p95,
        stats.speedup_min, stats.speedup_max, exact,
    );
    Ok(())
}

#[test]
#[ignore = "paired confirmation of promising cuBLASLt decode algorithms"]
fn bench_blaslt_decode_heuristics_paired() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    // Controls from the first sweep.
    paired_bf16(&runtime, 2048, 2048, 4)?;
    paired_fp8(&runtime, 16384, 2048, 1)?;

    // Primary candidates from the first sweep.
    paired_bf16(&runtime, 6144, 2048, 5)?;
    paired_fp8(&runtime, 2048, 8192, 3)?;
    Ok(())
}
