use std::mem::size_of;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use half::bf16;

use super::{
    CudaRuntime,
    benchmark::{BenchConfig, benchmark_gpu_paired},
    module::{load_function, load_module},
};

const MODULE_NAME: &str = "dsm_handoff";
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/dsm_handoff.ptx"));
const CLUSTER_BLOCKS: u32 = 8;
const THREADS: u32 = 256;
const CONSUMER_BLOCKS: usize = CLUSTER_BLOCKS as usize - 1;
const TILE_ELEMENTS: &[usize] = &[4096, 8192];

struct DsmHandoffKernels {
    dsm: CudaFunction,
    global: CudaFunction,
}

impl DsmHandoffKernels {
    fn load(runtime: &CudaRuntime) -> Result<Self> {
        let module = load_module(runtime.context(), MODULE_NAME, PTX)?;
        Ok(Self {
            dsm: load_function(&module, MODULE_NAME, "dsm_handoff_bf16")?,
            global: load_function(&module, MODULE_NAME, "global_handoff_bf16")?,
        })
    }
}

unsafe fn launch_dsm(
    runtime: &CudaRuntime,
    kernels: &DsmHandoffKernels,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    elements: usize,
) -> Result<()> {
    let elements_u32 = u32::try_from(elements).context("DSM tile elements exceed u32")?;
    let shared_mem_bytes = u32::try_from(
        elements
            .checked_mul(size_of::<bf16>())
            .context("DSM shared-memory size overflow")?,
    )
    .context("DSM shared-memory size exceeds u32")?;
    let config = LaunchConfig {
        grid_dim: (CLUSTER_BLOCKS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes,
    };
    let mut args = runtime.stream().launch_builder(&kernels.dsm);
    args.arg(input).arg(output).arg(&elements_u32);
    unsafe {
        args.launch(config)?;
    }
    Ok(())
}

unsafe fn launch_global(
    runtime: &CudaRuntime,
    kernels: &DsmHandoffKernels,
    input: &CudaSlice<bf16>,
    scratch: &mut CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    elements: usize,
) -> Result<()> {
    let elements_u32 = u32::try_from(elements).context("global tile elements exceed u32")?;
    let config = LaunchConfig {
        grid_dim: (CLUSTER_BLOCKS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut args = runtime.stream().launch_builder(&kernels.global);
    args.arg(input)
        .arg(scratch)
        .arg(output)
        .arg(&elements_u32);
    unsafe {
        args.launch(config)?;
    }
    Ok(())
}

fn deterministic_input(elements: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * 37 % 509) as f32 - 254.0) / 128.0))
        .collect()
}

fn check_exact_handoff(
    runtime: &CudaRuntime,
    kernels: &DsmHandoffKernels,
    elements: usize,
) -> Result<bool> {
    let host_input = deterministic_input(elements);
    let input = runtime.stream().clone_htod(&host_input)?;
    let mut scratch = runtime.stream().alloc_zeros::<bf16>(elements)?;
    let output_elements = elements
        .checked_mul(CONSUMER_BLOCKS)
        .context("DSM output size overflow")?;
    let mut global_output = runtime.stream().alloc_zeros::<bf16>(output_elements)?;
    let mut dsm_output = runtime.stream().alloc_zeros::<bf16>(output_elements)?;

    unsafe {
        launch_global(
            runtime,
            kernels,
            &input,
            &mut scratch,
            &mut global_output,
            elements,
        )?;
        launch_dsm(runtime, kernels, &input, &mut dsm_output, elements)?;
    }
    runtime.stream().synchronize()?;

    let global_host = runtime.stream().clone_dtoh(&global_output)?;
    let dsm_host = runtime.stream().clone_dtoh(&dsm_output)?;
    if global_host != dsm_host {
        return Ok(false);
    }
    for consumer in 0..CONSUMER_BLOCKS {
        let begin = consumer * elements;
        let end = begin + elements;
        if global_host[begin..end] != host_input {
            return Ok(false);
        }
    }
    Ok(true)
}

#[test]
#[ignore = "SM120 thread-block-cluster DSM producer/consumer benchmark"]
fn bench_cluster_dsm_handoff_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let kernels = DsmHandoffKernels::load(&runtime)?;

    for &elements in TILE_ELEMENTS {
        let exact = check_exact_handoff(&runtime, &kernels, elements)?;
        ensure!(exact, "DSM handoff failed exactness at {elements} BF16 elements");

        let host_input = deterministic_input(elements);
        let input = runtime.stream().clone_htod(&host_input)?;
        let mut scratch = runtime.stream().alloc_zeros::<bf16>(elements)?;
        let output_elements = elements
            .checked_mul(CONSUMER_BLOCKS)
            .context("DSM benchmark output size overflow")?;
        let mut global_output = runtime.stream().alloc_zeros::<bf16>(output_elements)?;
        let mut dsm_output = runtime.stream().alloc_zeros::<bf16>(output_elements)?;

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            BenchConfig {
                warmup: 20,
                batches: 50,
                iterations_per_batch: 50,
            },
            || unsafe {
                launch_global(
                    &runtime,
                    &kernels,
                    &input,
                    &mut scratch,
                    &mut global_output,
                    elements,
                )
            },
            || unsafe { launch_dsm(&runtime, &kernels, &input, &mut dsm_output, elements) },
        )?;

        println!(
            "dsm_handoff bytes={} elements={} global_mean_us={:.3} dsm_mean_us={:.3} speedup_mean={:.4}x global_p50_us={:.3} dsm_p50_us={:.3} speedup_p50={:.4}x global_p95_us={:.3} dsm_p95_us={:.3} speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x exact={}",
            elements * size_of::<bf16>(),
            elements,
            stats.reference.mean_us,
            stats.candidate.mean_us,
            stats.speedup_mean,
            stats.reference.p50_us,
            stats.candidate.p50_us,
            stats.speedup_p50,
            stats.reference.p95_us,
            stats.candidate.p95_us,
            stats.speedup_p95,
            stats.speedup_min,
            stats.speedup_max,
            exact,
        );
    }

    Ok(())
}
