use anyhow::{Result, ensure};
use half::bf16;

use super::{
    CudaRuntime,
    benchmark::{BenchConfig, benchmark_gpu_paired},
};
use crate::tensor::Shape;

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;

#[derive(Debug)]
struct NumericalMetrics {
    max_abs: f64,
    nrmse: f64,
    cosine: f64,
    non_finite: usize,
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

fn numerical_metrics(reference: &[bf16], candidate: &[bf16]) -> NumericalMetrics {
    assert_eq!(reference.len(), candidate.len());
    let mut max_abs = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut squared_candidate = 0.0f64;
    let mut dot = 0.0f64;
    let mut non_finite = 0usize;

    for (&expected, &observed) in reference.iter().zip(candidate) {
        let expected = f64::from(expected.to_f32());
        let observed = f64::from(observed.to_f32());
        if !expected.is_finite() || !observed.is_finite() {
            non_finite += 1;
            continue;
        }
        let difference = observed - expected;
        let absolute = difference.abs();
        max_abs = max_abs.max(absolute);
        squared_error += difference * difference;
        squared_reference += expected * expected;
        squared_candidate += observed * observed;
        dot += expected * observed;
    }

    let nrmse = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else if squared_error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let cosine = if squared_reference > 0.0 && squared_candidate > 0.0 {
        dot / (squared_reference * squared_candidate).sqrt()
    } else if squared_reference == squared_candidate {
        1.0
    } else {
        0.0
    };

    NumericalMetrics {
        max_abs,
        nrmse,
        cosine,
        non_finite,
    }
}

#[test]
fn test_segmented_prefill_flash_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    let test_cases: &[&[usize]] = &[
        &[16, 32],
        &[64, 128, 64],
        &[128, 128, 128, 128],
        &[512, 512],
    ];

    for segment_lens in test_cases {
        let num_segments = segment_lens.len();
        let mut offsets = Vec::with_capacity(num_segments + 1);
        offsets.push(0u32);
        let mut running = 0usize;
        for &len in *segment_lens {
            running += len;
            offsets.push(u32::try_from(running)?);
        }
        let total_tokens = running;
        let max_tokens = segment_lens.iter().copied().max().unwrap_or(0);

        let q_elements = total_tokens * Q_HEADS * HEAD_DIM;
        let kv_elements = total_tokens * KV_HEADS * HEAD_DIM;

        let query = runtime.upload(
            &deterministic_bf16(q_elements, 17),
            Shape::new([total_tokens, Q_HEADS, HEAD_DIM]),
        )?;
        let key = runtime.upload(
            &deterministic_bf16(kv_elements, 29),
            Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let value = runtime.upload(
            &deterministic_bf16(kv_elements, 43),
            Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let offsets_gpu = runtime.upload(&offsets, Shape::new([num_segments + 1]))?;

        let mut candidate = runtime.alloc_bf16(Shape::new([total_tokens, Q_HEADS, HEAD_DIM]))?;

        unsafe {
            runtime
                .kernels()
                .attention()
                .launch_segmented_prefill_flash_lfm2_bf16(
                    runtime.stream(),
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    offsets_gpu.storage(),
                    candidate.storage_mut(),
                    num_segments,
                    max_tokens,
                    total_tokens,
                )?;
        }
        runtime.synchronize()?;

        let candidate_host = runtime.download(&candidate)?;

        // Verify each segment independently against single-sequence prefill reference
        for (seg_idx, &len) in segment_lens.iter().enumerate() {
            let seg_start = offsets[seg_idx] as usize;
            let seg_end = offsets[seg_idx + 1] as usize;

            // Slice candidate for this segment
            let seg_cand =
                &candidate_host[seg_start * Q_HEADS * HEAD_DIM..seg_end * Q_HEADS * HEAD_DIM];

            // Run reference prefill kernel on this segment
            let q_host = deterministic_bf16(q_elements, 17);
            let k_host = deterministic_bf16(kv_elements, 29);
            let v_host = deterministic_bf16(kv_elements, 43);

            let sub_q = runtime.upload(
                &q_host[seg_start * Q_HEADS * HEAD_DIM..seg_end * Q_HEADS * HEAD_DIM],
                Shape::new([len, Q_HEADS, HEAD_DIM]),
            )?;
            let sub_k = runtime.upload(
                &k_host[seg_start * KV_HEADS * HEAD_DIM..seg_end * KV_HEADS * HEAD_DIM],
                Shape::new([len, KV_HEADS, HEAD_DIM]),
            )?;
            let sub_v = runtime.upload(
                &v_host[seg_start * KV_HEADS * HEAD_DIM..seg_end * KV_HEADS * HEAD_DIM],
                Shape::new([len, KV_HEADS, HEAD_DIM]),
            )?;
            let mut ref_out = runtime.alloc_bf16(Shape::new([len, Q_HEADS, HEAD_DIM]))?;

            unsafe {
                runtime.kernels().attention().launch_prefill_lfm2_bf16(
                    runtime.stream(),
                    sub_q.storage(),
                    sub_k.storage(),
                    sub_v.storage(),
                    ref_out.storage_mut(),
                    len,
                )?;
            }
            runtime.synchronize()?;
            let ref_host = runtime.download(&ref_out)?;

            let metrics = numerical_metrics(&ref_host, seg_cand);
            ensure!(
                metrics.non_finite == 0,
                "segmented flash prefill produced non-finite output at seg {seg_idx} len={len}"
            );
            ensure!(
                metrics.cosine >= 0.9999,
                "segmented flash prefill cosine too low at seg {seg_idx} len={len}: {metrics:?}"
            );
            ensure!(
                metrics.nrmse <= 0.05,
                "segmented flash prefill NRMSE too high at seg {seg_idx} len={len}: {metrics:?}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark: Segmented Tensor Core FlashAttention multi-sequence prefill"]
fn bench_segmented_prefill_flash_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let shapes: &[&[usize]] = &[
        &[512, 512],           // 2 x 512 (N=1024)
        &[512, 512, 512, 512], // 4 x 512 (N=2048)
        &[2048, 2048],         // 2 x 2048 (N=4096)
    ];

    let bench_config = BenchConfig {
        warmup: 3,
        batches: 10,
        iterations_per_batch: 1,
    };

    println!(
        "\n=== Segmented Tensor Core FlashAttention vs Hybrid Ragged Scalar ABBA Benchmark ==="
    );

    for segment_lens in shapes {
        let num_segments = segment_lens.len();
        let mut offsets = Vec::with_capacity(num_segments + 1);
        offsets.push(0u32);
        let mut running = 0usize;
        for &len in *segment_lens {
            running += len;
            offsets.push(u32::try_from(running)?);
        }
        let total_tokens = running;
        let max_tokens = segment_lens.iter().copied().max().unwrap_or(0);

        let q_elements = total_tokens * Q_HEADS * HEAD_DIM;
        let kv_elements = total_tokens * KV_HEADS * HEAD_DIM;

        let query = runtime.upload(
            &deterministic_bf16(q_elements, 17),
            Shape::new([total_tokens, Q_HEADS, HEAD_DIM]),
        )?;
        let key = runtime.upload(
            &deterministic_bf16(kv_elements, 29),
            Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let value = runtime.upload(
            &deterministic_bf16(kv_elements, 43),
            Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let offsets_gpu = runtime.upload(&offsets, Shape::new([num_segments + 1]))?;

        // Setup hybrid ragged attention input structures (dummy cache for prefix=0)
        let page_size = 16usize;
        let num_pages = total_tokens.div_ceil(page_size).max(1);
        let dummy_k_cache =
            runtime.alloc_bf16(Shape::new([num_pages, KV_HEADS, page_size, HEAD_DIM]))?;
        let dummy_v_cache =
            runtime.alloc_bf16(Shape::new([num_pages, KV_HEADS, page_size, HEAD_DIM]))?;
        let block_table_stride = total_tokens.div_ceil(page_size).max(1);
        let block_tables = runtime.zeros::<u32>(Shape::new([num_segments, block_table_stride]))?;
        let mut request_slots_host = Vec::with_capacity(total_tokens);
        let mut positions_host = Vec::with_capacity(total_tokens);
        for (seg_idx, &len) in segment_lens.iter().enumerate() {
            for p in 0..len {
                request_slots_host.push(seg_idx as u32);
                positions_host.push(p as u32);
            }
        }
        let request_slots = runtime.upload(&request_slots_host, Shape::new([total_tokens]))?;
        let position_ids = runtime.upload(&positions_host, Shape::new([total_tokens]))?;

        let mut out_ref = runtime.alloc_bf16(Shape::new([total_tokens, Q_HEADS, HEAD_DIM]))?;
        let mut out_cand = runtime.alloc_bf16(Shape::new([total_tokens, Q_HEADS, HEAD_DIM]))?;

        let paired = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            bench_config,
            || unsafe {
                runtime
                    .kernels()
                    .attention()
                    .launch_hybrid_ragged_lfm2_bf16(
                        runtime.stream(),
                        crate::cuda::HybridAttentionLaunch {
                            page_size,
                            query: query.storage(),
                            current_key: key.storage(),
                            current_value: value.storage(),
                            key_cache: dummy_k_cache.storage(),
                            value_cache: dummy_v_cache.storage(),
                            block_tables: block_tables.storage(),
                            request_slots: request_slots.storage(),
                            position_ids: position_ids.storage(),
                            segment_offsets: offsets_gpu.storage(),
                            output: out_ref.storage_mut(),
                            num_tokens: total_tokens,
                            num_pages,
                            block_table_stride,
                            num_segments,
                        },
                    )?;
                Ok(())
            },
            || unsafe {
                runtime
                    .kernels()
                    .attention()
                    .launch_segmented_prefill_flash_lfm2_bf16(
                        runtime.stream(),
                        query.storage(),
                        key.storage(),
                        value.storage(),
                        offsets_gpu.storage(),
                        out_cand.storage_mut(),
                        num_segments,
                        max_tokens,
                        total_tokens,
                    )?;
                Ok(())
            },
        )?;

        let ref_host = runtime.download(&out_ref)?;
        let cand_host = runtime.download(&out_cand)?;
        let metrics = numerical_metrics(&ref_host, &cand_host);

        println!(
            "Shape {:?}: total_N={}\n  Ref (Hybrid Ragged Scalar): mean={:.3}us p50={:.3}us p95={:.3}us\n  Cand (Segmented Flash WMMA): mean={:.3}us p50={:.3}us p95={:.3}us\n  Speedup: mean={:.4}x p50={:.4}x p95={:.4}x\n  Numerical: max_abs={:.6} nrmse={:.6} cosine={:.6} non_finite={}\n",
            segment_lens,
            total_tokens,
            paired.reference.mean_us,
            paired.reference.p50_us,
            paired.reference.p95_us,
            paired.candidate.mean_us,
            paired.candidate.p50_us,
            paired.candidate.p95_us,
            paired.speedup_mean,
            paired.speedup_p50,
            paired.speedup_p95,
            metrics.max_abs,
            metrics.nrmse,
            metrics.cosine,
            metrics.non_finite
        );
    }
    Ok(())
}
