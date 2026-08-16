use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu},
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

fn silu_reference(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_mul_reference(gate: &[bf16], up: &[bf16]) -> Vec<bf16> {
    assert_eq!(gate.len(), up.len(),);

    gate.iter()
        .zip(up.iter())
        .map(|(&gate, &up)| bf16::from_f32(silu_reference(gate.to_f32()) * up.to_f32()))
        .collect()
}

fn make_gate(n: usize) -> Vec<bf16> {
    (0..n)
        .map(|i| {
            let value = ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 32.0;

            bf16::from_f32(value)
        })
        .collect()
}

fn make_up(n: usize) -> Vec<bf16> {
    (0..n)
        .map(|i| {
            let value = ((i.wrapping_mul(17) % 193) as f32 - 96.0) / 32.0;

            bf16::from_f32(value)
        })
        .collect()
}

fn run_case(runtime: &CudaRuntime, n: usize) -> Result<()> {
    let gate_host = make_gate(n);

    let up_host = make_up(n);

    let expected = silu_mul_reference(&gate_host, &up_host);

    let gate = runtime.upload(&gate_host, Shape::new([n]))?;

    let up = runtime.upload(&up_host, Shape::new([n]))?;

    let mut out = runtime.zeros::<bf16>(Shape::new([n]))?;

    unsafe {
        runtime.kernels().silu_mul().launch_bf16(
            runtime.stream(),
            gate.storage(),
            up.storage(),
            out.storage_mut(),
            n,
        )?;
    }

    let actual = readback(runtime, &out)?;

    /*
     * GPU implementation uses __expf(),
     * while CPU oracle uses Rust's exp().
     *
     * BF16 output already quantizes the final
     * result, but leave a little tolerance for
     * the approximate GPU exponential.
     */
    assert_close_bf16(&actual, &expected, 0.01, 0.01);

    Ok(())
}

#[test]
fn silu_mul_bf16_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    for n in [
        1,
        2,
        3,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        255,
        256,
        257,
        511,
        512,
        513,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        65_535,
        65_536,
        65_537,
        1 << 20,
        (1 << 20) + 1,
    ] {
        run_case(&runtime, n)?;
    }

    Ok(())
}

#[test]
fn silu_mul_bf16_special_values() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    let gate_host = [
        -20.0, -10.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 10.0, 20.0,
    ]
    .map(bf16::from_f32);

    let up_host = [1.0, -1.0, 0.5, 2.0, -2.0, 4.0, 3.0, -3.0, 0.25, 1.5, -0.5].map(bf16::from_f32);

    let expected = silu_mul_reference(&gate_host, &up_host);

    let gate = runtime.upload(&gate_host, Shape::new([gate_host.len()]))?;

    let up = runtime.upload(&up_host, Shape::new([up_host.len()]))?;

    let mut out = runtime.zeros::<bf16>(Shape::new([gate_host.len()]))?;

    unsafe {
        runtime.kernels().silu_mul().launch_bf16(
            runtime.stream(),
            gate.storage(),
            up.storage(),
            out.storage_mut(),
            gate_host.len(),
        )?;
    }

    let actual = readback(&runtime, &out)?;

    assert_close_bf16(&actual, &expected, 0.01, 0.01);

    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_silu_mul_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    println!("block_size={}", runtime.kernels().silu_mul().block_size(),);

    for n in [
        1 << 10,
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
    ] {
        let gate_host = make_gate(n);

        let up_host = make_up(n);

        let gate = runtime.upload(&gate_host, Shape::new([n]))?;

        let up = runtime.upload(&up_host, Shape::new([n]))?;

        let mut out = runtime.zeros::<bf16>(Shape::new([n]))?;

        let stats = benchmark_gpu(
            runtime.context(),
            runtime.stream(),
            BenchConfig::default(),
            || {
                unsafe {
                    runtime.kernels().silu_mul().launch_bf16(
                        runtime.stream(),
                        gate.storage(),
                        up.storage(),
                        out.storage_mut(),
                        n,
                    )?;
                }

                Ok(())
            },
        )?;

        /*
         * gate: read  BF16 = 2 bytes
         * up:   read  BF16 = 2 bytes
         * out:  write BF16 = 2 bytes
         *
         * logical traffic:
         *
         * 6 bytes / element
         */
        let logical_bytes = n as f64 * 6.0;

        let seconds = stats.mean_us * 1e-6;

        let bandwidth = logical_bytes / seconds / 1e9;

        let elements_per_second = n as f64 / seconds;

        println!(
            "N={n:>10} | \
             mean={:>8.3} us | \
             p50={:>8.3} us | \
             p95={:>8.3} us | \
             min={:>8.3} us | \
             logical={:>8.2} GB/s | \
             {:>12.0} elem/s",
            stats.mean_us, stats.p50_us, stats.p95_us, stats.min_us, bandwidth, elements_per_second,
        );
    }

    Ok(())
}
