use std::{ffi::c_void, mem::size_of};

use anyhow::{Context as _, Result, ensure};
use cudarc::{
    cublaslt::{result, sys},
    driver::{CudaSlice, DevicePtr, DevicePtrMut},
};
use half::bf16;

use crate::cuda::{CudaRuntime, benchmark::{BenchConfig, benchmark_gpu}};

use super::BlasLt;

const MAX_CANDIDATES: usize = 16;

struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn row_major(data_type: sys::cudaDataType_t, rows: usize, cols: usize) -> Result<Self> {
        let rows_u64 = u64::try_from(rows).context("autotune rows exceed u64")?;
        let cols_u64 = u64::try_from(cols).context("autotune cols exceed u64")?;
        let rows_i64 = i64::try_from(rows).context("autotune rows exceed i64")?;
        let row_ld = i64::try_from(cols).context("autotune leading dimension exceeds i64")?;
        let create_ld = rows_i64.max(row_ld);
        let raw = result::create_matrix_layout(data_type, rows_u64, cols_u64, create_ld)
            .context("failed to create autotune matrix layout")?;
        let layout = Self { raw };
        let order = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW;
        unsafe {
            result::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order as *const _ as *const c_void,
                size_of::<sys::cublasLtOrder_t>(),
            )
            .context("failed to set autotune row-major order")?;
            if create_ld != row_ld {
                result::set_matrix_layout_attribute(
                    layout.raw,
                    sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_LD,
                    &row_ld as *const _ as *const c_void,
                    size_of::<i64>(),
                )
                .context("failed to set autotune leading dimension")?;
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
        )
        .context("failed to create autotune matmul descriptor")?;
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
        let raw = result::create_matmul_pref().context("failed to create autotune preference")?;
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

struct CandidateProblem {
    desc: MatmulDesc,
    a: MatrixLayout,
    b: MatrixLayout,
    c: MatrixLayout,
    candidates: Vec<sys::cublasLtMatmulHeuristicResult_t>,
    _a_scale: Option<CudaSlice<f32>>,
    _b_scale: Option<CudaSlice<f32>>,
}

fn query_candidates(
    blas: &BlasLt,
    m: usize,
    n: usize,
    k: usize,
    fp8: bool,
) -> Result<CandidateProblem> {
    let a_type = if fp8 {
        sys::cudaDataType_t::CUDA_R_8F_E4M3
    } else {
        sys::cudaDataType_t::CUDA_R_16BF
    };
    let b_type = a_type;
    let a = MatrixLayout::row_major(a_type, m, k)?;
    let b = MatrixLayout::row_major(b_type, n, k)?;
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
        .result()
        .context("cuBLASLt multi-heuristic query failed")?;
    }
    let returned = usize::try_from(returned).context("negative cuBLASLt candidate count")?;
    ensure!(returned <= MAX_CANDIDATES, "cuBLASLt returned too many candidates");
    candidates.truncate(returned);
    candidates.retain(|candidate| {
        candidate.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            && candidate.workspaceSize <= blas.workspace_size
    });
    ensure!(!candidates.is_empty(), "cuBLASLt returned no legal autotune candidates");

    Ok(CandidateProblem {
        desc,
        a,
        b,
        c,
        candidates,
        _a_scale: a_scale,
        _b_scale: b_scale,
    })
}

unsafe fn launch_bf16(
    blas: &BlasLt,
    problem: &CandidateProblem,
    candidate: usize,
    x: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let algo = &problem.candidates[candidate].algo;
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
            algo as *const _,
            workspace_ptr as *mut c_void,
            blas.workspace_size,
            blas.stream.cu_stream() as *mut _,
        )?;
    }
    Ok(())
}

unsafe fn launch_fp8(
    blas: &BlasLt,
    problem: &CandidateProblem,
    candidate: usize,
    x: &CudaSlice<u8>,
    weight: &CudaSlice<u8>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let algo = &problem.candidates[candidate].algo;
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
            algo as *const _,
            workspace_ptr as *mut c_void,
            blas.workspace_size,
            blas.stream.cu_stream() as *mut _,
        )?;
    }
    Ok(())
}

fn similarity(reference: &[bf16], candidate: &[bf16]) -> (f64, f64, f32, usize) {
    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut candidate_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut non_finite = 0usize;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = reference.to_f32();
        let candidate = candidate.to_f32();
        if !reference.is_finite() || !candidate.is_finite() {
            non_finite += 1;
            continue;
        }
        let error = candidate - reference;
        error_sq += f64::from(error * error);
        reference_sq += f64::from(reference * reference);
        candidate_sq += f64::from(candidate * candidate);
        dot += f64::from(reference * candidate);
        max_abs = max_abs.max(error.abs());
    }
    let nrmse = if reference_sq == 0.0 {
        if error_sq == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (error_sq / reference_sq).sqrt()
    };
    let cosine = if reference_sq == 0.0 || candidate_sq == 0.0 {
        if reference_sq == candidate_sq { 1.0 } else { 0.0 }
    } else {
        dot / (reference_sq.sqrt() * candidate_sq.sqrt())
    };
    (nrmse, cosine, max_abs, non_finite)
}

fn fill_bf16(elements: usize, mul: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % 257) as f32 - 128.0) / 128.0))
        .collect()
}

fn benchmark_bf16_shape(runtime: &CudaRuntime, m: usize, n: usize, k: usize) -> Result<()> {
    let blas = runtime.blaslt();
    let problem = query_candidates(blas, m, n, k, false)?;
    let x = blas.stream.clone_htod(&fill_bf16(m * k, 37))?;
    let weight = blas.stream.clone_htod(&fill_bf16(n * k, 17))?;
    let mut reference_out = blas.stream.alloc_zeros::<bf16>(m * n)?;
    unsafe { launch_bf16(blas, &problem, 0, &x, &weight, &mut reference_out)?; }
    blas.stream.synchronize()?;
    let reference = blas.stream.clone_dtoh(&reference_out)?;

    let mut best_mean = f64::INFINITY;
    let mut best_index = 0usize;
    for candidate in 0..problem.candidates.len() {
        let mut out = blas.stream.alloc_zeros::<bf16>(m * n)?;
        unsafe { launch_bf16(blas, &problem, candidate, &x, &weight, &mut out)?; }
        blas.stream.synchronize()?;
        let actual = blas.stream.clone_dtoh(&out)?;
        let (nrmse, cosine, max_abs, non_finite) = similarity(&reference, &actual);
        let stats = benchmark_gpu(
            blas.stream.context(),
            &blas.stream,
            BenchConfig { warmup: 5, batches: 20, iterations_per_batch: 20 },
            || unsafe { launch_bf16(blas, &problem, candidate, &x, &weight, &mut out) },
        )?;
        if stats.mean_us < best_mean {
            best_mean = stats.mean_us;
            best_index = candidate;
        }
        println!(
            "blaslt_autotune dtype=bf16 M={} N={} K={} candidate={} candidates={} workspace={} waves={:.3} mean_us={:.3} p50_us={:.3} p95_us={:.3} nrmse={:.8} cosine={:.8} max_abs={:.6} non_finite={}",
            m, n, k, candidate, problem.candidates.len(), problem.candidates[candidate].workspaceSize,
            problem.candidates[candidate].wavesCount, stats.mean_us, stats.p50_us, stats.p95_us,
            nrmse, cosine, max_abs, non_finite,
        );
    }
    let baseline = {
        let mut out = blas.stream.alloc_zeros::<bf16>(m * n)?;
        benchmark_gpu(
            blas.stream.context(),
            &blas.stream,
            BenchConfig { warmup: 5, batches: 20, iterations_per_batch: 20 },
            || unsafe { launch_bf16(blas, &problem, 0, &x, &weight, &mut out) },
        )?
    };
    println!(
        "blaslt_autotune_best dtype=bf16 M={} N={} K={} baseline_us={:.3} best_candidate={} best_us={:.3} speedup={:.4}x",
        m, n, k, baseline.mean_us, best_index, best_mean, baseline.mean_us / best_mean,
    );
    Ok(())
}

fn benchmark_fp8_shape(runtime: &CudaRuntime, m: usize, n: usize, k: usize) -> Result<()> {
    let blas = runtime.blaslt();
    let problem = query_candidates(blas, m, n, k, true)?;
    let x = blas.stream.alloc_zeros::<u8>(m * k)?;
    let weight = blas.stream.alloc_zeros::<u8>(n * k)?;
    let mut reference_out = blas.stream.alloc_zeros::<bf16>(m * n)?;
    unsafe { launch_fp8(blas, &problem, 0, &x, &weight, &mut reference_out)?; }
    blas.stream.synchronize()?;
    let reference = blas.stream.clone_dtoh(&reference_out)?;

    let mut best_mean = f64::INFINITY;
    let mut best_index = 0usize;
    for candidate in 0..problem.candidates.len() {
        let mut out = blas.stream.alloc_zeros::<bf16>(m * n)?;
        unsafe { launch_fp8(blas, &problem, candidate, &x, &weight, &mut out)?; }
        blas.stream.synchronize()?;
        let actual = blas.stream.clone_dtoh(&out)?;
        let (nrmse, cosine, max_abs, non_finite) = similarity(&reference, &actual);
        let stats = benchmark_gpu(
            blas.stream.context(),
            &blas.stream,
            BenchConfig { warmup: 5, batches: 20, iterations_per_batch: 20 },
            || unsafe { launch_fp8(blas, &problem, candidate, &x, &weight, &mut out) },
        )?;
        if stats.mean_us < best_mean {
            best_mean = stats.mean_us;
            best_index = candidate;
        }
        println!(
            "blaslt_autotune dtype=fp8 M={} N={} K={} candidate={} candidates={} workspace={} waves={:.3} mean_us={:.3} p50_us={:.3} p95_us={:.3} nrmse={:.8} cosine={:.8} max_abs={:.6} non_finite={}",
            m, n, k, candidate, problem.candidates.len(), problem.candidates[candidate].workspaceSize,
            problem.candidates[candidate].wavesCount, stats.mean_us, stats.p50_us, stats.p95_us,
            nrmse, cosine, max_abs, non_finite,
        );
    }
    let baseline = {
        let mut out = blas.stream.alloc_zeros::<bf16>(m * n)?;
        benchmark_gpu(
            blas.stream.context(),
            &blas.stream,
            BenchConfig { warmup: 5, batches: 20, iterations_per_batch: 20 },
            || unsafe { launch_fp8(blas, &problem, 0, &x, &weight, &mut out) },
        )?
    };
    println!(
        "blaslt_autotune_best dtype=fp8 M={} N={} K={} baseline_us={:.3} best_candidate={} best_us={:.3} speedup={:.4}x",
        m, n, k, baseline.mean_us, best_index, best_mean, baseline.mean_us / best_mean,
    );
    Ok(())
}

#[test]
#[ignore = "RTX 5060 cuBLASLt multi-heuristic autotune probe"]
fn bench_blaslt_decode_heuristics() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    benchmark_bf16_shape(&runtime, 1, 2048, 2048)?;
    benchmark_bf16_shape(&runtime, 1, 6144, 2048)?;
    benchmark_fp8_shape(&runtime, 1, 16384, 2048)?;
    benchmark_fp8_shape(&runtime, 1, 2048, 8192)?;
    benchmark_fp8_shape(&runtime, 1, 65536, 2048)?;
    Ok(())
}
