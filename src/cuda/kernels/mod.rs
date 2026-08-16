mod attention;
mod attention_async;
mod embedding;
mod fp8_quantize;
mod gather;
mod kernel_set;
mod kv_cache;
mod residual;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod silu_mul;

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaContext;

use attention::AttentionKernels;
use attention_async::AsyncAttentionKernels;
use embedding::EmbeddingKernels;
use fp8_quantize::Fp8QuantizeKernels;
use gather::GatherKernels;
use kernel_set::KernelSet;
use kv_cache::KvCacheKernels;
use residual::ResidualKernels;
use rms_norm::RmsNormKernels;
use rope::RopeKernels;
use sampling::SamplingKernels;
use short_conv::ShortConvKernels;
use silu_mul::SiluMulKernels;

pub(crate) struct Kernels {
    #[allow(dead_code)]
    residual: ResidualKernels,
    embedding: EmbeddingKernels,
    rms_norm: RmsNormKernels,
    silu_mul: SiluMulKernels,
    rope: RopeKernels,
    kv_cache: KvCacheKernels,
    attention: AttentionKernels,
    attention_async: AsyncAttentionKernels,
    short_conv: ShortConvKernels,
    sampling: SamplingKernels,
    fp8_quantize: Fp8QuantizeKernels,
    gather: GatherKernels,
}

impl Kernels {
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            residual: ResidualKernels::load(context)?,
            embedding: EmbeddingKernels::load(context)?,
            rms_norm: RmsNormKernels::load(context)?,
            silu_mul: SiluMulKernels::load(context)?,
            rope: RopeKernels::load(context)?,
            kv_cache: KvCacheKernels::load(context)?,
            attention: AttentionKernels::load(context)?,
            attention_async: AsyncAttentionKernels::load(context)?,
            short_conv: ShortConvKernels::load(context)?,
            sampling: SamplingKernels::load(context)?,
            fp8_quantize: Fp8QuantizeKernels::load(context)?,
            gather: GatherKernels::load(context)?,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn residual(&self) -> &ResidualKernels {
        &self.residual
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

    pub(crate) fn attention_async(&self) -> &AsyncAttentionKernels {
        &self.attention_async
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
}

#[cfg(test)]
mod tests;
