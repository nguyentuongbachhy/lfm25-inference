#include <cuda_bf16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int PACKED_QKV_BLOCK_SIZE = 256;
constexpr uint32_t LFM2_NUM_Q_HEADS = 32U;
constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_HALF_DIM = 32U;
constexpr size_t LFM2_Q_ELEMENTS = LFM2_NUM_Q_HEADS * LFM2_HEAD_DIM;
constexpr size_t LFM2_KV_ELEMENTS = LFM2_NUM_KV_HEADS * LFM2_HEAD_DIM;
constexpr size_t LFM2_QKV_ELEMENTS = LFM2_Q_ELEMENTS + 2 * LFM2_KV_ELEMENTS;

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
__device__ __forceinline__ void packed_qkv_postprocess_body(
    const __nv_bfloat16* __restrict__ packed_qkv,
    __nv_bfloat16* __restrict__ query_out,
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
    const size_t packed_base = token * LFM2_QKV_ELEMENTS;

    __shared__ float sin_cache[LFM2_HALF_DIM];
    __shared__ float cos_cache[LFM2_HALF_DIM];

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

    // Q occupies the first 2048 elements in each packed row. Materialize only
    // the normalized+rotated Q tensor needed by the following paged-attention
    // kernel; raw K/V never get standalone global tensors on this path.
    for (uint32_t head = warp; head < LFM2_NUM_Q_HEADS; head += 8U) {
        const size_t q_offset = static_cast<size_t>(head) * LFM2_HEAD_DIM;
        const size_t source0 = packed_base + q_offset + lane;
        const size_t source1 = source0 + LFM2_HALF_DIM;
        const float raw0 = __bfloat162float(packed_qkv[source0]);
        const float raw1 = __bfloat162float(packed_qkv[source1]);
        const float sum = warp_sum(fmaf(raw0, raw0, raw1 * raw1));
        const float inv_rms = __frsqrt_rn(sum * (1.0f / 64.0f) + eps);
        const float x0 = rms_weight_bf16(raw0, inv_rms, q_weight0);
        const float x1 = rms_weight_bf16(raw1, inv_rms, q_weight1);
        const size_t destination0 =
            (token * LFM2_NUM_Q_HEADS + head) * LFM2_HEAD_DIM + lane;
        const size_t destination1 = destination0 + LFM2_HALF_DIM;
        query_out[destination0] =
            __float2bfloat16_rn(fmaf(-x1, sin_value, x0 * cos_value));
        query_out[destination1] =
            __float2bfloat16_rn(fmaf(x0, sin_value, x1 * cos_value));
    }

    if (warp < LFM2_NUM_KV_HEADS) {
        const uint32_t head = warp;
        const size_t kv_offset = static_cast<size_t>(head) * LFM2_HEAD_DIM;
        const size_t key_base = packed_base + LFM2_Q_ELEMENTS + kv_offset;
        const size_t value_base =
            packed_base + LFM2_Q_ELEMENTS + LFM2_KV_ELEMENTS + kv_offset;
        const size_t key0 = key_base + lane;
        const size_t key1 = key0 + LFM2_HALF_DIM;
        const float raw0 = __bfloat162float(packed_qkv[key0]);
        const float raw1 = __bfloat162float(packed_qkv[key1]);
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
        value_cache[cache0] = packed_qkv[value_base + lane];
        value_cache[cache1] = packed_qkv[value_base + lane + LFM2_HALF_DIM];
    }
}

extern "C" __global__
__launch_bounds__(PACKED_QKV_BLOCK_SIZE)
void packed_qkv_postprocess_ps16(
    const __nv_bfloat16* __restrict__ packed_qkv,
    __nv_bfloat16* __restrict__ query_out,
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
    packed_qkv_postprocess_body<16>(
        packed_qkv,
        query_out,
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
__launch_bounds__(PACKED_QKV_BLOCK_SIZE)
void packed_qkv_postprocess_ps32(
    const __nv_bfloat16* __restrict__ packed_qkv,
    __nv_bfloat16* __restrict__ query_out,
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
    packed_qkv_postprocess_body<32>(
        packed_qkv,
        query_out,
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
