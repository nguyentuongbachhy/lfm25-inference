#[allow(dead_code)]
mod attention;
mod attention_async;
mod attention_async_fast;
mod attention_fused;
#[cfg(test)]
mod attention_async_fast_tests;
#[cfg(test)]
mod attention_async_w4_tests;
#[cfg(test)]
mod attention_regression_tests;
mod embedding;
mod gather;
mod linear;
#[cfg(test)]
mod mok_bench_tests;
#[cfg(test)]
mod mok_dispatch_bench_tests;
#[cfg(test)]
mod mok_fast_exp_bench_tests;
#[cfg(test)]
mod mok_fused_bench_tests;
#[cfg(test)]
mod mok_oneexp_bench_tests;
#[cfg(test)]
mod mok_w4_bench_tests;
mod qk_postprocess;
mod residual;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod silu_mul;

pub use attention::{hybrid_ragged_attention_lfm2_bf16, prefill_attention_lfm2_bf16};
pub use attention_async::paged_attention_lfm2_bf16;
pub(crate) use attention_async::paged_ragged_attention_lfm2_bf16;
pub(crate) use attention_async_fast::{
    paged_attention_fast_lfm2_bf16, paged_ragged_attention_fast_lfm2_bf16,
};
pub(crate) use attention_fused::{
    fused_paged_attention_decode_lfm2_bf16,
    fused_ragged_paged_attention_decode_lfm2_bf16,
};
pub use embedding::embedding_bf16;
pub use gather::gather_rows_bf16;
pub use linear::{linear_bf16, linear_last_row_bf16};
pub(crate) use linear::{linear_fp8_e4m3, linear_last_row_fp8_e4m3, quantize_weight_e4m3};
pub(crate) use qk_postprocess::{
    qk_norm_rope_kv_write_arena_decode_bf16, qk_norm_rope_kv_write_decode_bf16,
};
#[allow(unused_imports)]
pub use residual::residual_add_bf16;
pub use rms_norm::{residual_rms_norm_bf16, rms_norm_bf16};
pub use rope::rope_qk_bf16_inplace;
pub use sampling::{argmax_bf16, argmax_rows_bf16};
pub use short_conv::{short_conv_lfm2_bf16, short_conv_segmented_lfm2_bf16};
#[allow(unused_imports)]
pub use silu_mul::{silu_mul_bf16, silu_mul_packed_bf16};
