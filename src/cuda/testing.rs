use anyhow::{Context as _, Result};
use cudarc::driver::DeviceRepr;
use half::bf16;

use crate::{cuda::CudaRuntime, tensor::Tensor};

pub(crate) fn readback<T>(runtime: &CudaRuntime, tensor: &Tensor<T>) -> Result<Vec<T>>
where
    T: DeviceRepr,
{
    runtime
        .stream()
        .clone_dtoh(tensor.storage())
        .context("failed to read GPU tensor back to host")
}

pub(crate) fn assert_eq_bf16(actual: &[bf16], expected: &[bf16]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch",);

    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "BF16 mismatch at index {i}: actual={}, expected={}",
            a.to_f32(),
            e.to_f32(),
        );
    }
}

pub(crate) fn assert_close_bf16(actual: &[bf16], expected: &[bf16], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(),);

    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.to_f32();

        let expected = expected.to_f32();

        let tolerance = atol + rtol * expected.abs();

        assert!(
            (actual - expected).abs() <= tolerance,
            "BF16 mismatch at {i}: \
             actual={actual}, \
             expected={expected}, \
             tolerance={tolerance}",
        );
    }
}
