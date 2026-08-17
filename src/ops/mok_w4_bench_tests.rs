use anyhow::Result;
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvCache},
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    tensor::Shape,
};

#[test]
#[ignore = "GPU benchmark"]
fn bench_mok_async_w4_one_exp_paired_ab() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query = runtime.upload(
        &vec![bf16::from_f32(0.01); 32 * 64],
        Shape::new([1, 32, 64]),
    )?;
    let config = BenchConfig {
        warmup: 20,
        batches: 60,
        iterations_per_batch: 20,
    };

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        for sequence_length in [16usize, 32, 128, 512, 2048, 8192] {
            let cache = PagedKvCache::new(&runtime, sequence_length, page_size)?;
            let position = runtime.upload(
                &[u32::try_from(sequence_length - 1)?],
                Shape::new([1]),
            )?;
            let mut w8_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;
            let mut w4_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;

            let paired = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                config,
                || {
                    unsafe {
                        runtime.kernels().attention_async().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            w8_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                },
                || {
                    unsafe {
                        runtime.kernels().attention_async_w4().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            w4_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                },
            )?;

            println!(
                "mok_async_w4 page_size={} context={} w8_two_exp_mean={:.3}us w8_two_exp_p50={:.3}us w8_two_exp_p95={:.3}us w4_one_exp_mean={:.3}us w4_one_exp_p50={:.3}us w4_one_exp_p95={:.3}us paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x paired_speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
                page_size.value(),
                sequence_length,
                paired.reference.mean_us,
                paired.reference.p50_us,
                paired.reference.p95_us,
                paired.candidate.mean_us,
                paired.candidate.p50_us,
                paired.candidate.p95_us,
                paired.speedup_mean,
                paired.speedup_p50,
                paired.speedup_p95,
                paired.speedup_min,
                paired.speedup_max,
            );
        }
    }
    Ok(())
}
