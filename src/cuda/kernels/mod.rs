mod attention;
mod attention_async_fast;
mod attention_fused;
mod embedding;
mod fp8_quantize;
mod gather;
mod kernel_set;
mod kv_cache;
mod metadata;
mod qk_postprocess;
mod qkv_packed_postprocess;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod silu_mul;

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaContext;

pub(crate) use attention::{HybridAttentionLaunch, PagedAttentionLaunch};
#[cfg(test)]
pub(crate) use attention::RaggedAttentionLaunch;
use attention::AttentionKernels;
pub(crate) use attention_async_fast::FastRaggedAttentionLaunch;
use attention_async_fast::AsyncAttentionFastKernels;
pub(crate) use attention_fused::{
    FusedAttentionCommon, FusedDecodeLaunch, FusedRaggedDecodeLaunch,
};
use attention_fused::FusedAttentionKernels;
pub(crate) use embedding::EmbeddingLaunch;
use embedding::EmbeddingKernels;
use fp8_quantize::Fp8QuantizeKernels;
pub(crate) use gather::GatherLaunch;
use gather::GatherKernels;
use kernel_set::KernelSet;
pub(crate) use kv_cache::KvCacheWriteLaunch;
use kv_cache::KvCacheKernels;
use metadata::MetadataKernels;
pub(crate) use qk_postprocess::QkPostprocessLaunch;
use qk_postprocess::QkPostprocessKernels;
pub(crate) use qkv_packed_postprocess::PackedQkvPostprocessLaunch;
use qkv_packed_postprocess::PackedQkvPostprocessKernels;
pub(crate) use rms_norm::{ResidualRmsNormLaunch, RmsNormLaunch};
use rms_norm::RmsNormKernels;
pub(crate) use rope::RopeLaunch;
use rope::RopeKernels;
use sampling::SamplingKernels;
#[cfg(test)]
pub(crate) use short_conv::RaggedShortConvLaunch;
pub(crate) use short_conv::{SegmentedShortConvLaunch, ShortConvLaunch};
use short_conv::ShortConvKernels;
use silu_mul::SiluMulKernels;

pub(crate) struct Kernels {
    embedding: EmbeddingKernels,
    rms_norm: RmsNormKernels,
    silu_mul: SiluMulKernels,
    rope: RopeKernels,
    kv_cache: KvCacheKernels,
    attention: AttentionKernels,
    attention_async_fast: AsyncAttentionFastKernels,
    attention_fused: FusedAttentionKernels,
    qk_postprocess: QkPostprocessKernels,
    qkv_packed_postprocess: PackedQkvPostprocessKernels,
    short_conv: ShortConvKernels,
    sampling: SamplingKernels,
    fp8_quantize: Fp8QuantizeKernels,
    gather: GatherKernels,
    metadata: MetadataKernels,
}

impl Kernels {
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            embedding: EmbeddingKernels::load(context)?,
            rms_norm: RmsNormKernels::load(context)?,
            silu_mul: SiluMulKernels::load(context)?,
            rope: RopeKernels::load(context)?,
            kv_cache: KvCacheKernels::load(context)?,
            attention: AttentionKernels::load(context)?,
            attention_async_fast: AsyncAttentionFastKernels::load(context)?,
            attention_fused: FusedAttentionKernels::load(context)?,
            qk_postprocess: QkPostprocessKernels::load(context)?,
            qkv_packed_postprocess: PackedQkvPostprocessKernels::load(context)?,
            short_conv: ShortConvKernels::load(context)?,
            sampling: SamplingKernels::load(context)?,
            fp8_quantize: Fp8QuantizeKernels::load(context)?,
            gather: GatherKernels::load(context)?,
            metadata: MetadataKernels::load(context)?,
        })
    }

    pub(crate) fn embedding(&self) -> &EmbeddingKernels {
        &self.embedding
    }

    pub(crate) fn rms_norm(&self) -> &RmsNormKernels {
        &self.rms_norm
    }

    pub(crate) fn silu_mul(&self) -> &SiluMulKernels {
        &self.silu_mul
    }

    pub(crate) fn rope(&self) -> &RopeKernels {
        &self.rope
    }

    pub(crate) fn kv_cache(&self) -> &KvCacheKernels {
        &self.kv_cache
    }

    pub(crate) fn attention(&self) -> &AttentionKernels {
        &self.attention
    }

    pub(crate) fn attention_async_fast(&self) -> &AsyncAttentionFastKernels {
        &self.attention_async_fast
    }

    pub(crate) fn attention_fused(&self) -> &FusedAttentionKernels {
        &self.attention_fused
    }

    pub(crate) fn qk_postprocess(&self) -> &QkPostprocessKernels {
        &self.qk_postprocess
    }

    pub(crate) fn qkv_packed_postprocess(&self) -> &PackedQkvPostprocessKernels {
        &self.qkv_packed_postprocess
    }

    pub(crate) fn short_conv(&self) -> &ShortConvKernels {
        &self.short_conv
    }

    pub(crate) fn sampling(&self) -> &SamplingKernels {
        &self.sampling
    }

    pub(crate) fn fp8_quantize(&self) -> &Fp8QuantizeKernels {
        &self.fp8_quantize
    }

    pub(crate) fn gather(&self) -> &GatherKernels {
        &self.gather
    }

    pub(crate) fn metadata(&self) -> &MetadataKernels {
        &self.metadata
    }
}

#[cfg(test)]
mod tests;
