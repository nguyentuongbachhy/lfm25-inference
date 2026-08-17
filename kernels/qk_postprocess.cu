#include <cuda_bf16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int QK_POSTPROCESS_BLOCK_SIZE = 256;
constexpr uint32_t LFM2_NUM_Q_HEADS = 32U;
constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_HALF_DIM = 32U;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
        value += __shfl_down_sync(0xffffffffU, value, delta);
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float rms_weight_bf16(
    float value,
    float inv_rms,
    __nv_bfloat16 weight
) {
    // Match the reference RMSNorm path's BF16 rounding boundary:
    // normalize -> BF16 -> BF16 multiply by weight -> float for RoPE.
    const __nv_bfloat16 normalized = __float2bfloat16_rn(value * inv_rms);
    return __bfloat162float(__hmul(normalized, weight));
}

template <int PAGE_SIZE>
__device__ __forceinline__ size_t kv_cache_index(
    size_t page,
    uint32_t head,
    size_t page_offset,
    uint32_t dim
) {
    return (((page * LFM2_NUM_KV_HEADS + head) * PAGE_SIZE + page_offset)
        * LFM2_HEAD_DIM + dim);
}

template <int PAGE_SIZE>
__device__ __forceinline__ void qk_norm_rope_kv_write_decode_body(
    __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    size_t num_tokens,
    size_t num_pages,
    float eps
) {
    const size_t token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const int64_t slot_value = slot_mapping[token];
    if (slot_value < 0) {
        return;
    }
    const size_t physical_slot = static_cast<size_t>(slot_value);
    const size_t page = physical_slot / PAGE_SIZE;
    const size_t page_offset = physical_slot % PAGE_SIZE;
    if (page >= num_pages) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t position = position_ids[token];

    __shared__ float sin_cache[LFM2_HALF_DIM];
    __shared__ float cos_cache[LFM2_HALF_DIM];

    // All eight warps need the same RoPE coefficients. Computing them once
    // avoids 8x duplicate transcendental work while costing one CTA barrier.
    if (warp == 0U) {
        sincosf(
            static_cast<float>(position) * inv_freq[lane],
            &sin_cache[lane],
            &cos_cache[lane]
        );
    }
    __syncthreads();

    const float sin_value = sin_cache[lane];
    const float cos_value = cos_cache[lane];
    const __nv_bfloat16 q_weight0 = query_norm[lane];
    const __nv_bfloat16 q_weight1 = query_norm[lane + LFM2_HALF_DIM];

    // Eight warps process four Q heads each. Every head is normalized entirely
    // inside one warp, so no CTA-wide synchronization is needed for RMSNorm.
    for (uint32_t head = warp; head < LFM2_NUM_Q_HEADS; head += 8U) {
        const size_t base = (token * LFM2_NUM_Q_HEADS + head) * LFM2_HEAD_DIM;
        const size_t index0 = base + lane;
        const size_t index1 = index0 + LFM2_HALF_DIM;
        const float raw0 = __bfloat162float(query[index0]);
        const float raw1 = __bfloat162float(query[index1]);
        const float sum = warp_sum(fmaf(raw0, raw0, raw1 * raw1));
        const float inv_rms = __frsqrt_rn(sum * (1.0f / 64.0f) + eps);
        const float x0 = rms_weight_bf16(raw0, inv_rms, q_weight0);
        const float x1 = rms_weight_bf16(raw1, inv_rms, q_weight1);

        query[index0] = __float2bfloat16_rn(fmaf(-x1, sin_value, x0 * cos_value));
        query[index1] = __float2bfloat16_rn(fmaf(x0, sin_value, x1 * cos_value));
    }

    // Exactly one warp owns each KV head. K is normalized + rotated directly
    // into the paged cache, so the normalized/rotated K tensor never needs to
    // be materialized in global memory on decode. V is copied into the same
    // physical slot while the source line is already hot.
    if (warp < LFM2_NUM_KV_HEADS) {
        const uint32_t head = warp;
        const size_t base = (token * LFM2_NUM_KV_HEADS + head) * LFM2_HEAD_DIM;
        const size_t index0 = base + lane;
        const size_t index1 = index0 + LFM2_HALF_DIM;
        const float raw0 = __bfloat162float(key[index0]);
        const float raw1 = __bfloat162float(key[index1]);
        const float sum = warp_sum(fmaf(raw0, raw0, raw1 * raw1));
        const float inv_rms = __frsqrt_rn(sum * (1.0f / 64.0f) + eps);
        const float x0 = rms_weight_bf16(raw0, inv_rms, key_norm[lane]);
        const float x1 = rms_weight_bf16(
            raw1,
            inv_rms,
            key_norm[lane + LFM2_HALF_DIM]
        );
        const __nv_bfloat16 rotated0 =
            __float2bfloat16_rn(fmaf(-x1, sin_value, x0 * cos_value));
        const __nv_bfloat16 rotated1 =
            __float2bfloat16_rn(fmaf(x0, sin_value, x1 * cos_value));

        const size_t cache0 = kv_cache_index<PAGE_SIZE>(page, head, page_offset, lane);
        const size_t cache1 = kv_cache_index<PAGE_SIZE>(
            page,
            head,
            page_offset,
            lane + LFM2_HALF_DIM
        );
        key_cache[cache0] = rotated0;
        key_cache[cache1] = rotated1;
        value_cache[cache0] = value[index0];
        value_cache[cache1] = value[index1];
    }
}

extern "C" __global__
__launch_bounds__(QK_POSTPROCESS_BLOCK_SIZE)
void qk_norm_rope_kv_write_decode_ps16(
    __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    size_t num_tokens,
    size_t num_pages,
    float eps
) {
    qk_norm_rope_kv_write_decode_body<16>(
        query,
        key,
        value,
        query_norm,
        key_norm,
        inv_freq,
        position_ids,
        slot_mapping,
        key_cache,
        value_cache,
        num_tokens,
        num_pages,
        eps
    );
}

extern "C" __global__
__launch_bounds__(QK_POSTPROCESS_BLOCK_SIZE)
void qk_norm_rope_kv_write_decode_ps32(
    __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    size_t num_tokens,
    size_t num_pages,
    float eps
) {
    qk_norm_rope_kv_write_decode_body<32>(
        query,
        key,
        value,
        query_norm,
        key_norm,
        inv_freq,
        position_ids,
        slot_mapping,
        key_cache,
        value_cache,
        num_tokens,
        num_pages,
        eps
    );
}
