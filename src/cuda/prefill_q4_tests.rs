use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use half::bf16;

use super::{
    CudaRuntime,
    benchmark::{BenchConfig, benchmark_gpu_paired},
    module::{load_function, load_module},
};
use crate::tensor::Shape;

const MODULE_NAME: &str = "attention_prefill_q4";
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_prefill_q4.ptx"));
const THREADS: u32 = 256;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const Q4_TILE: usize = 4;
const TOKEN_COUNTS: &[usize] = &[512, 2048, 8192];

struct Q4Kernel {
    function: CudaFunction,
}

impl Q4Kernel {
    fn load(runtime: &CudaRuntime) -> Result<Self> {
        let module = load_module(runtime.context(), MODULE_NAME, PTX)?;
        Ok(Self {
            function: load_function(&module, MODULE_NAME, "prefill_gqa_lfm2_bf16_q4")?,
        })
    }
}

fn deterministic_bf16(elements: usize, seed: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let mixed = index
                .wrapping_mul(1_103_515_245)
                .wrapping_add(seed.wrapping_mul(12_345));
            let bucket = (mixed >> 8) & 255;
            bf16::from_f32((bucket as f32 - 127.5) / 512.0)
        })
        .collect()
}

unsafe fn launch_q4(
    runtime: &CudaRuntime,
    kernel: &Q4Kernel,
    query: &cudarc::driver::CudaSlice<bf16>,
    key: &cudarc::driver::CudaSlice<bf16>,
    value: &cudarc::driver::CudaSlice<bf16>,
    output: &mut cudarc::driver::CudaSlice<bf16>,
    num_tokens: usize,
) -> Result<()> {
    let query_tiles = num_tokens.div_ceil(Q4_TILE);
    let blocks = query_tiles
        .checked_mul(KV_HEADS)
        .context("Q4 prefill grid size overflow")?;
    let grid_x = u32::try_from(blocks).context("Q4 prefill grid exceeds u32")?;
    let config = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut args = runtime.stream().launch_builder(&kernel.function);
    args.arg(query)
        .arg(key)
        .arg(value)
        .arg(output)
        .arg(&num_tokens);
    unsafe {
        args.launch(config)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark: research Q4 contiguous prefill attention"]
fn bench_prefill_attention_q4_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let q4 = Q4Kernel::load(&runtime)?;

    for &num_tokens in TOKEN_COUNTS {
        let q_elements = num_tokens
            .checked_mul(Q_HEADS * HEAD_DIM)
            .context("Q4 query element count overflow")?;
        let kv_elements = num_tokens
            .checked_mul(KV_HEADS * HEAD_DIM)
            .context("Q4 KV element count overflow")?;

        let query = runtime.upload(
            &deterministic_bf16(q_elements, 17),
            Shape::new([num_tokens, Q_HEADS, HEAD_DIM]),
        )?;
        let key = runtime.upload(
            &deterministic_bf16(kv_elements, 29),
            Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let value = runtime.upload(
            &deterministic_bf16(kv_elements, 43),
            Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let mut reference = runtime.alloc_bf16(Shape::new([num_tokens, Q_HEADS, HEAD_DIM]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([num_tokens, Q_HEADS, HEAD_DIM]))?;

        unsafe {
            runtime.kernels().attention().launch_prefill_lfm2_bf16(
                runtime.stream(),
                query.storage(),
                key.storage(),
                value.storage(),
                reference.storage_mut(),
                num_tokens,
            )?;
            launch_q4(
                &runtime,
                &q4,
                query.storage(),
                key.storage(),
                value.storage(),
                candidate.storage_mut(),
                num_tokens,
            )?;
        }
        runtime.synchronize()?;
        let reference_host = runtime.download(&reference)?;
        let candidate_host = runtime.download(&candidate)?;
        let exact = reference_host == candidate_host;
        ensure!(
            exact,
            "Q4 prefill output is not bit-exact at N={num_tokens}"
        );

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            BenchConfig {
                warmup: 3,
                batches: 10,
                iterations_per_batch: 1,
            },
            || unsafe {
                runtime.kernels().attention().launch_prefill_lfm2_bf16(
                    runtime.stream(),
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    reference.storage_mut(),
                    num_tokens,
                )
            },
            || unsafe {
                launch_q4(
                    &runtime,
                    &q4,
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    candidate.storage_mut(),
                    num_tokens,
                )
            },
        )?;

        println!(
            "prefill_q4 N={} q2_mean_us={:.3} q4_mean_us={:.3} speedup_mean={:.4}x q2_p50_us={:.3} q4_p50_us={:.3} speedup_p50={:.4}x q2_p95_us={:.3} q4_p95_us={:.3} speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x exact={}",
            num_tokens,
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
