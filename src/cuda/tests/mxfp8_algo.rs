use std::{ffi::c_void, mem::size_of, sync::Arc};

use anyhow::{Context as _, Result, ensure};
use cudarc::{
    cublaslt::{result, sys},
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg},
};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu},
        blaslt::{Fp8LinearConfig, fp8::Fp8ScaleMode},
        launch::KernelLaunch,
        module::{load_function, load_module},
        testing::readback,
    },
    tensor::Shape,
};

const E4M3_MAX_FINITE: f32 = 448.0;
const MXFP8_BLOCK: usize = 32;
const MXFP8_OUTER_TILE: usize = 128;
const MXFP8_SCALE_TILE_BYTES: usize = 512;
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const MAX_HEURISTICS: usize = 32;

fn scale_storage_len(outer: usize, inner: usize) -> Result<usize> {
    ensure!(outer > 0, "MXFP8 outer dimension must be positive");
    ensure!(
        inner > 0 && inner.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 inner dimension must be positive and divisible by 32"
    );
    let outer_tiles = outer.div_ceil(MXFP8_OUTER_TILE);
    let inner_tiles = inner.div_ceil(128);
    outer_tiles
        .checked_mul(inner_tiles)
        .and_then(|tiles| tiles.checked_mul(MXFP8_SCALE_TILE_BYTES))
        .context("MXFP8 scale storage overflow")
}

struct Mxfp8Quantizer {
    _module: Arc<CudaModule>,
    kernel: KernelLaunch,
}

impl Mxfp8Quantizer {
    fn load(runtime: &CudaRuntime) -> Result<Self> {
        let module = load_module(
            runtime.context(),
            "mxfp8_research",
            include_str!(concat!(env!("OUT_DIR"), "/mxfp8_research.ptx")),
        )?;
        let function = load_function(&module, "mxfp8_research", "quantize_bf16_mxfp8_vec32")?;
        Ok(Self {
            _module: module,
            kernel: KernelLaunch::new_with_multiple(function, 256, 32)?,
        })
    }

    unsafe fn launch(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<u8>,
        scales: &mut CudaSlice<u8>,
        outer: usize,
        inner: usize,
    ) -> Result<()> {
        let elements = outer
            .checked_mul(inner)
            .context("MXFP8 tensor size overflow")?;
        let scale_len = scale_storage_len(outer, inner)?;
        ensure!(input.len() >= elements, "MXFP8 input storage too small");
        ensure!(output.len() >= elements, "MXFP8 output storage too small");
        ensure!(scales.len() >= scale_len, "MXFP8 scale storage too small");

        let logical_warps = outer
            .checked_mul(inner / MXFP8_BLOCK)
            .context("MXFP8 warp count overflow")?;
        let warps_per_block = (self.kernel.policy().block_size() as usize) / 32;
        ensure!(warps_per_block > 0, "MXFP8 launch has no complete warp");
        let blocks = logical_warps.div_ceil(warps_per_block);
        let config = self.kernel.policy().exact_blocks(blocks)?;
        let mut args = stream.launch_builder(self.kernel.function());
        args.arg(input)
            .arg(output)
            .arg(scales)
            .arg(&outer)
            .arg(&inner)
            .arg(&scale_len);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}

struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn row_major(data_type: sys::cudaDataType_t, rows: usize, cols: usize) -> Result<Self> {
        let rows_u64 = u64::try_from(rows)?;
        let cols_u64 = u64::try_from(cols)?;
        let rows_i64 = i64::try_from(rows)?;
        let row_ld = i64::try_from(cols)?;
        let create_ld = rows_i64.max(row_ld);
        let raw = result::create_matrix_layout(data_type, rows_u64, cols_u64, create_ld)?;
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
    fn mxfp8(a_scale_ptr: u64, b_scale_ptr: u64) -> Result<Self> {
        let raw = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;
        let desc = Self { raw };
        let trans_a = 0_i32;
        let trans_b = 1_i32;
        let scale_mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
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
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                &scale_mode as *const _ as *const c_void,
                size_of::<sys::cublasLtMatmulMatrixScale_t>(),
            )?;
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                &scale_mode as *const _ as *const c_void,
                size_of::<sys::cublasLtMatmulMatrixScale_t>(),
            )?;
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &a_scale_ptr as *const u64 as *const c_void,
                size_of::<u64>(),
            )?;
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &b_scale_ptr as *const u64 as *const c_void,
                size_of::<u64>(),
            )?;
        }
        Ok(desc)
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
    fn new() -> Result<Self> {
        let raw = result::create_matmul_pref()?;
        let pref = Self { raw };
        unsafe {
            result::set_matmul_pref_attribute(
                pref.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &WORKSPACE_BYTES as *const usize as *const c_void,
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

#[derive(Clone)]
struct AlgoCandidate {
    algo: sys::cublasLtMatmulAlgo_t,
    workspace_size: usize,
    waves_count: f32,
}

struct Mxfp8AlgoPlan {
    handle: sys::cublasLtHandle_t,
    stream: Arc<CudaStream>,
    desc: MatmulDesc,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,
    candidates: Vec<AlgoCandidate>,
    workspace: CudaSlice<u8>,
}

impl Mxfp8AlgoPlan {
    fn new(
        runtime: &CudaRuntime,
        m: usize,
        n: usize,
        k: usize,
        a_scales: &CudaSlice<u8>,
        b_scales: &CudaSlice<u8>,
    ) -> Result<Self> {
        ensure!(k.is_multiple_of(32), "MXFP8 K must be divisible by 32");
        let stream = Arc::clone(runtime.stream());
        let handle = result::create_handle()?;
        let (a_scale_ptr, _) = a_scales.device_ptr(&stream);
        let (b_scale_ptr, _) = b_scales.device_ptr(&stream);
        let desc = MatmulDesc::mxfp8(a_scale_ptr, b_scale_ptr)?;
        let a_layout = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_8F_E4M3, m, k)?;
        let b_layout = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_8F_E4M3, n, k)?;
        let c_layout = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_16BF, m, n)?;
        let preference = Preference::new()?;

        let mut storage =
            Vec::<std::mem::MaybeUninit<sys::cublasLtMatmulHeuristicResult_t>>::with_capacity(
                MAX_HEURISTICS,
            );
        storage.resize_with(MAX_HEURISTICS, std::mem::MaybeUninit::uninit);
        let mut returned = 0_i32;
        unsafe {
            sys::cublasLtMatmulAlgoGetHeuristic(
                handle,
                desc.raw,
                a_layout.raw,
                b_layout.raw,
                c_layout.raw,
                c_layout.raw,
                preference.raw,
                i32::try_from(MAX_HEURISTICS)?,
                storage.as_mut_ptr().cast(),
                &mut returned,
            )
            .result()
            .context("cuBLASLt MXFP8 heuristic enumeration failed")?;
        }
        let returned = usize::try_from(returned)?;
        ensure!(returned > 0, "cuBLASLt returned no MXFP8 algorithms");
        ensure!(returned <= MAX_HEURISTICS, "invalid MXFP8 heuristic count");

        let mut candidates = Vec::with_capacity(returned);
        for slot in storage.into_iter().take(returned) {
            let heuristic = unsafe { slot.assume_init() };
            if heuristic.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && heuristic.workspaceSize <= WORKSPACE_BYTES
            {
                candidates.push(AlgoCandidate {
                    algo: heuristic.algo,
                    workspace_size: heuristic.workspaceSize,
                    waves_count: heuristic.wavesCount,
                });
            }
        }
        ensure!(
            !candidates.is_empty(),
            "no usable MXFP8 heuristic algorithms"
        );
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_BYTES) }?;

        Ok(Self {
            handle,
            stream,
            desc,
            a_layout,
            b_layout,
            c_layout,
            candidates,
            workspace,
        })
    }

    unsafe fn matmul(
        &self,
        candidate: usize,
        x: &CudaSlice<u8>,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let candidate = self
            .candidates
            .get(candidate)
            .context("MXFP8 algorithm index out of range")?;
        self.stream.context().bind_to_thread()?;
        let (x_ptr, _) = x.device_ptr(&self.stream);
        let (weight_ptr, _) = weight.device_ptr(&self.stream);
        let (out_ptr, _) = out.device_ptr_mut(&self.stream);
        let (workspace_ptr, _) = self.workspace.device_ptr(&self.stream);
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            result::matmul(
                self.handle,
                self.desc.raw,
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                x_ptr as *const c_void,
                self.a_layout.raw,
                weight_ptr as *const c_void,
                self.b_layout.raw,
                out_ptr as *const c_void,
                self.c_layout.raw,
                out_ptr as *mut c_void,
                self.c_layout.raw,
                &candidate.algo as *const _,
                workspace_ptr as *mut c_void,
                WORKSPACE_BYTES,
                self.stream.cu_stream() as *mut _,
            )?;
        }
        Ok(())
    }
}

impl Drop for Mxfp8AlgoPlan {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = result::destroy_handle(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

fn deterministic_outlier_values(
    elements: usize,
    seed: usize,
    outlier_period: usize,
    outlier_multiplier: f32,
) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let mixed = index
                .wrapping_mul(1_664_525usize.wrapping_add(seed * 2 + 1))
                .wrapping_add(1_013_904_223usize.wrapping_add(seed));
            let unit = (mixed & 0xffff) as f32 / 65_535.0;
            let mut value = (unit - 0.5) * 1.5;
            if index % outlier_period == 0 {
                value *= outlier_multiplier;
            }
            bf16::from_f32(value)
        })
        .collect()
}

fn tensorwide_scale(values: &[bf16]) -> (f32, f32) {
    let amax = values
        .iter()
        .map(|value| value.to_f32().abs())
        .fold(0.0f32, f32::max);
    if amax == 0.0 {
        return (1.0, 1.0);
    }
    let dequantize = amax / E4M3_MAX_FINITE;
    (1.0 / dequantize, dequantize)
}

fn output_metrics(reference: &[bf16], candidate: &[bf16]) -> (f64, f64, f64) {
    let mut squared_error = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = f64::from(reference.to_f32());
        let candidate = f64::from(candidate.to_f32());
        let delta = reference - candidate;
        squared_error += delta * delta;
        reference_energy += reference * reference;
        candidate_energy += candidate * candidate;
        dot += reference * candidate;
        max_abs = max_abs.max(delta.abs());
    }
    let rel_l2 = (squared_error / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let cosine = dot
        / (reference_energy * candidate_energy)
            .sqrt()
            .max(f64::MIN_POSITIVE);
    (rel_l2, cosine, max_abs)
}

#[test]
#[ignore = "GPU research benchmark for cuBLASLt MXFP8 heuristic candidates"]
fn bench_mxfp8_block32_algorithm_sweep() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let quantizer = Mxfp8Quantizer::load(&runtime)?;
    let bench = BenchConfig {
        warmup: 6,
        batches: 16,
        iterations_per_batch: 20,
    };

    for (site, m, n, k) in [
        ("down", 1usize, 2_048usize, 8_192usize),
        ("down", 2usize, 2_048usize, 8_192usize),
        ("gate_up", 1usize, 16_384usize, 2_048usize),
    ] {
        let x_host = deterministic_outlier_values(m * k, 17 + m, 127, 16.0);
        let weight_host = deterministic_outlier_values(n * k, 11, 509, 24.0);
        let x = runtime.upload(&x_host, Shape::new([m, k]))?;
        let weight = runtime.upload(&weight_host, Shape::new([n, k]))?;

        let mut x_mxfp8 = runtime.zeros::<u8>(Shape::new([m, k]))?;
        let mut weight_mxfp8 = runtime.zeros::<u8>(Shape::new([n, k]))?;
        let mut x_scales = runtime.zeros::<u8>(Shape::new([scale_storage_len(m, k)?]))?;
        let mut weight_scales = runtime.zeros::<u8>(Shape::new([scale_storage_len(n, k)?]))?;
        unsafe {
            quantizer.launch(
                runtime.stream(),
                x.storage(),
                x_mxfp8.storage_mut(),
                x_scales.storage_mut(),
                m,
                k,
            )?;
            quantizer.launch(
                runtime.stream(),
                weight.storage(),
                weight_mxfp8.storage_mut(),
                weight_scales.storage_mut(),
                n,
                k,
            )?;
        }

        let (x_quantize_scale, x_dequantize_scale) = tensorwide_scale(&x_host);
        let (weight_quantize_scale, weight_dequantize_scale) = tensorwide_scale(&weight_host);
        let mut x_tensorwide = runtime.zeros::<u8>(Shape::new([m, k]))?;
        let mut weight_tensorwide = runtime.zeros::<u8>(Shape::new([n, k]))?;
        unsafe {
            runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                runtime.stream(),
                x.storage(),
                x_tensorwide.storage_mut(),
                m * k,
                x_quantize_scale,
            )?;
            runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                runtime.stream(),
                weight.storage(),
                weight_tensorwide.storage_mut(),
                n * k,
                weight_quantize_scale,
            )?;
        }

        let mut bf16_out = runtime.zeros::<bf16>(Shape::new([m, n]))?;
        let mut tensorwide_out = runtime.zeros::<bf16>(Shape::new([m, n]))?;
        let mut mxfp8_out = runtime.zeros::<bf16>(Shape::new([m, n]))?;
        runtime.blaslt().prepare_linear_bf16(m, n, k)?;
        runtime
            .blaslt()
            .prepare_linear_fp8(m, n, k, Fp8ScaleMode::Tensorwide)?;
        let tensorwide_config = Fp8LinearConfig {
            m,
            n,
            k,
            scale_mode: Fp8ScaleMode::Tensorwide,
            output_scale: x_dequantize_scale * weight_dequantize_scale,
        };
        let plan = Mxfp8AlgoPlan::new(
            &runtime,
            m,
            n,
            k,
            x_scales.storage(),
            weight_scales.storage(),
        )?;
        runtime.synchronize()?;

        unsafe {
            runtime.blaslt().linear_bf16(
                x.storage(),
                weight.storage(),
                bf16_out.storage_mut(),
                m,
                n,
                k,
            )?;
            runtime.blaslt().linear_fp8_scaled(
                x_tensorwide.storage(),
                weight_tensorwide.storage(),
                tensorwide_out.storage_mut(),
                tensorwide_config,
            )?;
        }
        runtime.synchronize()?;

        let tensorwide_stats =
            benchmark_gpu(runtime.context(), runtime.stream(), bench, || unsafe {
                runtime.blaslt().linear_fp8_scaled(
                    x_tensorwide.storage(),
                    weight_tensorwide.storage(),
                    tensorwide_out.storage_mut(),
                    tensorwide_config,
                )
            })?;

        let mut timings = Vec::with_capacity(plan.candidates.len());
        for rank in 0..plan.candidates.len() {
            unsafe {
                plan.matmul(
                    rank,
                    x_mxfp8.storage(),
                    weight_mxfp8.storage(),
                    mxfp8_out.storage_mut(),
                )?;
            }
            runtime.synchronize()?;
            let stats = benchmark_gpu(runtime.context(), runtime.stream(), bench, || unsafe {
                plan.matmul(
                    rank,
                    x_mxfp8.storage(),
                    weight_mxfp8.storage(),
                    mxfp8_out.storage_mut(),
                )
            })?;
            let candidate = &plan.candidates[rank];
            println!(
                "mxfp8_algo site={site} M={m} N={n} K={k} rank={rank} workspace={} waves={:.4} mean_us={:.3} p50_us={:.3} p95_us={:.3}",
                candidate.workspace_size,
                candidate.waves_count,
                stats.mean_us,
                stats.p50_us,
                stats.p95_us,
            );
            timings.push(stats.mean_us);
        }

        let (best_rank, best_us) = timings
            .iter()
            .copied()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .context("MXFP8 algorithm sweep produced no timings")?;
        let first_us = timings[0];
        unsafe {
            plan.matmul(
                best_rank,
                x_mxfp8.storage(),
                weight_mxfp8.storage(),
                mxfp8_out.storage_mut(),
            )?;
        }
        runtime.synchronize()?;

        let bf16_host = readback(&runtime, &bf16_out)?;
        let mxfp8_host = readback(&runtime, &mxfp8_out)?;
        ensure!(
            mxfp8_host.iter().all(|value| value.to_f32().is_finite()),
            "MXFP8 best-algorithm output contains non-finite values"
        );
        let (rel_l2, cosine, max_abs) = output_metrics(&bf16_host, &mxfp8_host);
        ensure!(
            rel_l2 < 0.10,
            "MXFP8 algorithm-sweep rel_l2 too large: {rel_l2}"
        );
        ensure!(
            cosine > 0.995,
            "MXFP8 algorithm-sweep cosine too low: {cosine}"
        );

        println!(
            "mxfp8_algo_best site={site} M={m} N={n} K={k} candidates={} first_us={first_us:.3} best_rank={best_rank} best_us={best_us:.3} speedup_vs_first={:.4}x tensorwide_gemm_us={:.3} mxfp8_vs_tensorwide={:.4}x rel_l2={rel_l2:.8} cosine={cosine:.8} max_abs={max_abs:.6}",
            plan.candidates.len(),
            first_us / best_us,
            tensorwide_stats.mean_us,
            tensorwide_stats.mean_us / best_us,
        );
    }

    Ok(())
}
