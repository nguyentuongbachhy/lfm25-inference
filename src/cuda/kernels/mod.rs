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
mod residual_rms_fp8;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod silu_mul;

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaContext;

use attention::AttentionKernels;
#[cfg(test)]
pub(crate) use attention::RaggedAttentionLaunch;
pub(crate) use attention::{
    HybridAttentionLaunch, PagedAttentionLaunch, SegmentedFlashPrefillLaunch,
};
use attention_async_fast::AsyncAttentionFastKernels;
pub(crate) use attention_async_fast::{FastRaggedAttentionLaunch, SplitKRaggedAttentionLaunch};
use attention_fused::FusedAttentionKernels;
pub(crate) use attention_fused::{
    FusedAttentionCommon, FusedDecodeLaunch, FusedRaggedDecodeLaunch,
};
use embedding::EmbeddingKernels;
pub(crate) use embedding::EmbeddingLaunch;
use fp8_quantize::Fp8QuantizeKernels;
use gather::GatherKernels;
pub(crate) use gather::GatherLaunch;
use kernel_set::KernelSet;
use kv_cache::KvCacheKernels;
pub(crate) use kv_cache::KvCacheWriteLaunch;
use metadata::MetadataKernels;
pub(crate) use metadata::ScatterMetadataLaunch;
use qk_postprocess::QkPostprocessKernels;
pub(crate) use qk_postprocess::QkPostprocessLaunch;
use residual_rms_fp8::ResidualRmsFp8Kernels;
pub(crate) use residual_rms_fp8::ResidualRmsNormFp8Launch;
use rms_norm::RmsNormKernels;
pub(crate) use rms_norm::{ResidualRmsNormLaunch, RmsNormLaunch};
use rope::RopeKernels;
pub(crate) use rope::RopeLaunch;
use sampling::SamplingKernels;
#[cfg(test)]
pub(crate) use short_conv::RaggedShortConvLaunch;
use short_conv::ShortConvKernels;
pub(crate) use short_conv::{
    SegmentedShortConvLaunch, ShortConvLaunch, ShortConvWithHistoryLaunch,
};
use silu_mul::SiluMulKernels;

pub(crate) struct Kernels {
    embedding: EmbeddingKernels,
    rms_norm: RmsNormKernels,
    residual_rms_fp8: ResidualRmsFp8Kernels,
    silu_mul: SiluMulKernels,
    rope: RopeKernels,
    kv_cache: KvCacheKernels,
    attention: AttentionKernels,
    attention_async_fast: AsyncAttentionFastKernels,
    attention_fused: FusedAttentionKernels,
    qk_postprocess: QkPostprocessKernels,
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
            residual_rms_fp8: ResidualRmsFp8Kernels::load(context)?,
            silu_mul: SiluMulKernels::load(context)?,
            rope: RopeKernels::load(context)?,
            kv_cache: KvCacheKernels::load(context)?,
            attention: AttentionKernels::load(context)?,
            attention_async_fast: AsyncAttentionFastKernels::load(context)?,
            attention_fused: FusedAttentionKernels::load(context)?,
            qk_postprocess: QkPostprocessKernels::load(context)?,
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

    pub(crate) fn residual_rms_fp8(&self) -> &ResidualRmsFp8Kernels {
        &self.residual_rms_fp8
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
