mod attention;
mod attention_async_fast;
#[cfg(test)]
mod attention_async_fast_tests;
#[cfg(test)]
mod attention_cta_bench_tests;
#[cfg(test)]
mod attention_splitk_bench_tests;
#[cfg(test)]
mod attention_splitk_tests;
mod attention_fused;
#[cfg(test)]
mod attention_regression_tests;
mod embedding;
mod gather;
mod int8_tiny_m;
#[cfg(test)]
mod int8_tiny_m_tests;
#[cfg(test)]
mod w8a16_tiny_m_tests;
mod linear;
mod mok_dispatch;
#[cfg(test)]
mod mok_dispatch_bench_tests;
#[cfg(test)]
mod mok_fast_exp_bench_tests;
#[cfg(test)]
mod mok_fused_bench_tests;
mod qk_postprocess;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod silu_mul;
mod splitk_policy;

#[cfg(test)]
pub(crate) use attention::paged_attention_lfm2_bf16_sync;
pub use attention::prefill_attention_lfm2_bf16;
pub(crate) use attention::{HybridRaggedAttentionInput, hybrid_ragged_attention_lfm2_bf16};
pub(crate) use attention_async_fast::{
    FastRaggedAttentionInput, paged_attention_fast_lfm2_bf16,
    paged_ragged_attention_fast_lfm2_bf16, paged_ragged_attention_fast_lfm2_bf16_into,
    paged_ragged_attention_splitk_lfm2_bf16_into, splitk_workspace_elements,
};
pub(crate) use attention_fused::{
    FusedAttentionInput, FusedPagedAttentionInput, FusedRaggedAttentionInput,
    fused_paged_attention_decode_lfm2_bf16, fused_ragged_paged_attention_decode_lfm2_bf16,
    fused_ragged_paged_attention_decode_lfm2_bf16_into,
};
pub use embedding::embedding_bf16;
pub(crate) use embedding::embedding_bf16_into;
pub use gather::gather_rows_bf16;
pub(crate) use int8_tiny_m::{
    Int8PerChannelWeight, Int8TinyMWorkspace, linear_int8_tiny_m_prequantized_into,
    linear_w8a16_tiny_m_into, quantize_weight_s8_per_channel,
    silu_mul_packed_bf16_to_int8_tiny_m_into,
};
#[cfg(test)]
pub(crate) use int8_tiny_m::{
    linear_int8_tiny_m, linear_int8_tiny_m_into, linear_w8a16_tiny_m,
    quantize_int8_tiny_m_input_into,
};
pub use linear::{linear_bf16, linear_last_row_bf16};
pub(crate) use linear::{
    linear_bf16_into, linear_fp8_e4m3, linear_last_row_fp8_e4m3, quantize_weight_e4m3,
};
pub(crate) use mok_dispatch::should_use_mok_one_kernel;
pub(crate) use qk_postprocess::{
    QkPostprocessInput, qk_norm_rope_kv_write_arena_decode_bf16, qk_norm_rope_kv_write_decode_bf16,
};
pub use rms_norm::{residual_rms_norm_bf16, rms_norm_bf16};
pub(crate) use rms_norm::{residual_rms_norm_bf16_into, rms_norm_bf16_into};
pub use rope::rope_qk_bf16_inplace;
pub use sampling::{argmax_bf16, argmax_rows_bf16};
pub(crate) use sampling::argmax_rows_bf16_into;
pub use short_conv::{short_conv_lfm2_bf16, short_conv_segmented_lfm2_bf16};
pub(crate) use short_conv::short_conv_segmented_lfm2_bf16_into;
pub use silu_mul::silu_mul_packed_bf16;
pub(crate) use silu_mul::{silu_mul_packed_bf16_into, silu_mul_packed_bf16_to_e4m3_into};
pub(crate) use splitk_policy::splitk_decode_splits;