#include <cuda_bf16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int ATTENTION_MAX_BLOCK_SIZE = 256;
constexpr uint32_t LFM2_NUM_Q_HEADS = 32U;
constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_Q_PER_KV = 4U;
constexpr float LFM2_ATTN_SCALE = 0.125f;
constexpr uint32_t PREFILL_QUERY_TILE = 2U;
constexpr uint32_t PREFILL_KEY_TILE = 32U;

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void prefill_gqa_lfm2_bf16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens
) {
    const size_t query_tile = blockIdx.x / LFM2_NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % LFM2_NUM_KV_HEADS;
    const size_t query_start = query_tile * PREFILL_QUERY_TILE;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t num_warps = blockDim.x >> 5;
    const uint32_t tasks = PREFILL_QUERY_TILE * LFM2_Q_PER_KV;
    const uint32_t task_waves = (tasks + num_warps - 1U) / num_warps;
    __shared__ __nv_bfloat16 key_tile[PREFILL_KEY_TILE * LFM2_HEAD_DIM];
    __shared__ __nv_bfloat16 value_tile[PREFILL_KEY_TILE * LFM2_HEAD_DIM];

    for (uint32_t wave = 0U; wave < task_waves; ++wave) {
        const uint32_t task = wave * num_warps + warp;
        const uint32_t query_offset = task / LFM2_Q_PER_KV;
        const uint32_t q_offset = task % LFM2_Q_PER_KV;
        const size_t token = query_start + query_offset;
        const bool active = task < tasks && token < num_tokens;
        const uint32_t q_head = kv_head * LFM2_Q_PER_KV + q_offset;
        const size_t query_base =
            (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;
        const size_t dim0 = lane;
        const size_t dim1 = lane + 32U;
        const float q0 = active
            ? __bfloat162float(query[query_base + dim0])
            : 0.0f;
        const float q1 = active
            ? __bfloat162float(query[query_base + dim1])
            : 0.0f;
        float maximum = -INFINITY;
        float denominator = 0.0f;
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;

        const size_t max_query_position =
            query_start + PREFILL_QUERY_TILE - 1 < num_tokens
                ? query_start + PREFILL_QUERY_TILE - 1
                : num_tokens - 1;
        for (
            size_t key_start = 0;
            key_start <= max_query_position;
            key_start += PREFILL_KEY_TILE
        ) {
            const size_t remaining = max_query_position + 1 - key_start;
            const size_t tile_tokens = remaining < PREFILL_KEY_TILE
                ? remaining
                : PREFILL_KEY_TILE;
            const size_t tile_elements = tile_tokens * LFM2_HEAD_DIM;

            for (
                size_t element = threadIdx.x;
                element < tile_elements;
                element += blockDim.x
            ) {
                const size_t key_offset = element / LFM2_HEAD_DIM;
                const size_t dim = element % LFM2_HEAD_DIM;
                const size_t source =
                    ((key_start + key_offset) * LFM2_NUM_KV_HEADS + kv_head)
                    * LFM2_HEAD_DIM
                    + dim;
                key_tile[element] = key[source];
                value_tile[element] = value[source];
            }
            __syncthreads();

            if (active && key_start <= token) {
                const size_t valid_tokens = token + 1 - key_start < tile_tokens
                    ? token + 1 - key_start
                    : tile_tokens;
                for (size_t key_offset = 0; key_offset < valid_tokens; ++key_offset) {
                    const size_t key_base = key_offset * LFM2_HEAD_DIM;
                    const size_t index0 = key_base + dim0;
                    const size_t index1 = key_base + dim1;
                    float dot =
                        q0 * __bfloat162float(key_tile[index0])
                        + q1 * __bfloat162float(key_tile[index1]);

                    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                        dot += __shfl_down_sync(0xffffffffU, dot, delta);
                    }

                    dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;
                    const float next_maximum = fmaxf(maximum, dot);
                    const float old_scale = expf(maximum - next_maximum);
                    const float new_scale = expf(dot - next_maximum);
                    const float value0 = __bfloat162float(value_tile[index0]);
                    const float value1 = __bfloat162float(value_tile[index1]);
                    accumulator0 = accumulator0 * old_scale + value0 * new_scale;
                    accumulator1 = accumulator1 * old_scale + value1 * new_scale;
                    denominator = denominator * old_scale + new_scale;
                    maximum = next_maximum;
                }
            }
            __syncthreads();
        }

        if (active) {
            output[query_base + dim0] =
                __float2bfloat16_rn(accumulator0 / denominator);
            output[query_base + dim1] =
                __float2bfloat16_rn(accumulator1 / denominator);
        }
    }
}

template <int PAGE_SIZE>
__device__ __forceinline__ size_t cache_index(
    size_t page,
    size_t kv_head,
    size_t offset,
    size_t dim
) {
    return (
        ((page * LFM2_NUM_KV_HEADS + kv_head) * PAGE_SIZE + offset)
        * LFM2_HEAD_DIM
        + dim
    );
}

template <int PAGE_SIZE, bool RAGGED>
__device__ __forceinline__ void paged_gqa_lfm2_bf16_body(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length,
    size_t block_table_stride
) {
    const size_t token = blockIdx.x / LFM2_NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % LFM2_NUM_KV_HEADS;

    if (token >= num_tokens) {
        return;
    }

    const uint32_t position = position_ids[token];
    const size_t capacity = block_table_length * PAGE_SIZE;
    const size_t request_slot = RAGGED ? request_slots[token] : 0;
    const uint32_t* __restrict__ token_block_table =
        block_table + request_slot * block_table_stride;

    if (position >= capacity) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t num_warps = blockDim.x >> 5;
    __shared__ __nv_bfloat16 key_tile[PAGE_SIZE * LFM2_HEAD_DIM];
    __shared__ __nv_bfloat16 value_tile[PAGE_SIZE * LFM2_HEAD_DIM];

    const uint32_t q_waves =
        (LFM2_Q_PER_KV + num_warps - 1U) / num_warps;

    for (uint32_t wave = 0U; wave < q_waves; ++wave) {
        const uint32_t q_offset = wave * num_warps + warp;
        const bool active = q_offset < LFM2_Q_PER_KV;
        const uint32_t q_head = kv_head * LFM2_Q_PER_KV + q_offset;
        const size_t query_base =
            (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;

        const size_t dim0 = lane;
        const size_t dim1 = lane + 32U;

        const float q0 = active
            ? __bfloat162float(query[query_base + dim0])
            : 0.0f;
        const float q1 = active
            ? __bfloat162float(query[query_base + dim1])
            : 0.0f;

        float maximum = -INFINITY;
        float denominator = 0.0f;
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;

        const size_t last_page = position / PAGE_SIZE;
        for (size_t page = 0; page <= last_page; ++page) {
            const size_t physical_page = token_block_table[page];

            if (physical_page >= num_pages) {
                return;
            }
            const size_t page_start = page * PAGE_SIZE;
            const size_t remaining = static_cast<size_t>(position) + 1 - page_start;
            const size_t page_tokens = remaining < PAGE_SIZE ? remaining : PAGE_SIZE;
            const size_t tile_elements = page_tokens * LFM2_HEAD_DIM;

            for (
                size_t element = threadIdx.x;
                element < tile_elements;
                element += blockDim.x
            ) {
                const size_t offset = element / LFM2_HEAD_DIM;
                const size_t dim = element % LFM2_HEAD_DIM;
                const size_t index =
                    cache_index<PAGE_SIZE>(physical_page, kv_head, offset, dim);
                key_tile[element] = key_cache[index];
                value_tile[element] = value_cache[index];
            }
            __syncthreads();

            if (active) {
                for (size_t key_offset = 0; key_offset < page_tokens; ++key_offset) {
                    const size_t key_base = key_offset * LFM2_HEAD_DIM;
                    const size_t key_index0 = key_base + dim0;
                    const size_t key_index1 = key_base + dim1;

                    float dot =
                        q0 * __bfloat162float(key_tile[key_index0])
                        + q1 * __bfloat162float(key_tile[key_index1]);

                    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                        dot += __shfl_down_sync(0xffffffffU, dot, delta);
                    }

                    dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;

                    const float next_maximum = fmaxf(maximum, dot);
                    const float old_scale = expf(maximum - next_maximum);
                    const float new_scale = expf(dot - next_maximum);

                    const float value0 = __bfloat162float(value_tile[key_index0]);
                    const float value1 = __bfloat162float(value_tile[key_index1]);

                    accumulator0 = accumulator0 * old_scale + value0 * new_scale;
                    accumulator1 = accumulator1 * old_scale + value1 * new_scale;
                    denominator = denominator * old_scale + new_scale;
                    maximum = next_maximum;
                }
            }
            __syncthreads();
        }

        if (active) {
            output[query_base + dim0] =
                __float2bfloat16_rn(accumulator0 / denominator);
            output[query_base + dim1] =
                __float2bfloat16_rn(accumulator1 / denominator);
        }
    }
}

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void paged_gqa_lfm2_bf16_ps16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length
) {
    paged_gqa_lfm2_bf16_body<16, false>(
        query,
        key_cache,
        value_cache,
        block_table,
        nullptr,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_length,
        block_table_length
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void paged_gqa_lfm2_bf16_ps32(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length
) {
    paged_gqa_lfm2_bf16_body<32, false>(
        query,
        key_cache,
        value_cache,
        block_table,
        nullptr,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_length,
        block_table_length
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void paged_ragged_gqa_lfm2_bf16_ps16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length,
    size_t block_table_stride
) {
    paged_gqa_lfm2_bf16_body<16, true>(
        query,
        key_cache,
        value_cache,
        block_table,
        request_slots,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_length,
        block_table_stride
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void paged_ragged_gqa_lfm2_bf16_ps32(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length,
    size_t block_table_stride
) {
    paged_gqa_lfm2_bf16_body<32, true>(
        query,
        key_cache,
        value_cache,
        block_table,
        request_slots,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_length,
        block_table_stride
    );
}

template <int PAGE_SIZE>
__device__ __forceinline__ void hybrid_ragged_gqa_lfm2_bf16_body(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ current_key,
    const __nv_bfloat16* __restrict__ current_value,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    const uint32_t* __restrict__ segment_offsets,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t num_segments
) {
    const size_t token = blockIdx.x / LFM2_NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % LFM2_NUM_KV_HEADS;
    if (token >= num_tokens) {
        return;
    }

    size_t segment_begin = 0;
    for (size_t segment = 0; segment < num_segments; ++segment) {
        const size_t begin = segment_offsets[segment];
        const size_t end = segment_offsets[segment + 1];
        if (token >= begin && token < end) {
            segment_begin = begin;
            break;
        }
    }
    const uint32_t position = position_ids[token];
    const size_t current_offset = token - segment_begin;
    if (position < current_offset) {
        return;
    }
    const size_t prefix_tokens = position - current_offset;
    const size_t request_slot = request_slots[token];
    const uint32_t* __restrict__ token_block_table =
        block_table + request_slot * block_table_stride;

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t num_warps = blockDim.x >> 5;
    __shared__ __nv_bfloat16 key_tile[PAGE_SIZE * LFM2_HEAD_DIM];
    __shared__ __nv_bfloat16 value_tile[PAGE_SIZE * LFM2_HEAD_DIM];
    const uint32_t q_waves =
        (LFM2_Q_PER_KV + num_warps - 1U) / num_warps;

    for (uint32_t wave = 0U; wave < q_waves; ++wave) {
        const uint32_t q_offset = wave * num_warps + warp;
        const bool active = q_offset < LFM2_Q_PER_KV;
        const uint32_t q_head = kv_head * LFM2_Q_PER_KV + q_offset;
        const size_t query_base =
            (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;
        const size_t dim0 = lane;
        const size_t dim1 = lane + 32U;
        const float q0 = active
            ? __bfloat162float(query[query_base + dim0])
            : 0.0f;
        const float q1 = active
            ? __bfloat162float(query[query_base + dim1])
            : 0.0f;
        float maximum = -INFINITY;
        float denominator = 0.0f;
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;

        for (size_t page_start = 0; page_start < prefix_tokens; page_start += PAGE_SIZE) {
            const size_t logical_page = page_start / PAGE_SIZE;
            const size_t physical_page = token_block_table[logical_page];
            if (physical_page >= num_pages) {
                return;
            }
            const size_t remaining = prefix_tokens - page_start;
            const size_t page_tokens = remaining < PAGE_SIZE ? remaining : PAGE_SIZE;
            const size_t tile_elements = page_tokens * LFM2_HEAD_DIM;
            for (
                size_t element = threadIdx.x;
                element < tile_elements;
                element += blockDim.x
            ) {
                const size_t offset = element / LFM2_HEAD_DIM;
                const size_t dim = element % LFM2_HEAD_DIM;
                const size_t index =
                    cache_index<PAGE_SIZE>(physical_page, kv_head, offset, dim);
                key_tile[element] = key_cache[index];
                value_tile[element] = value_cache[index];
            }
            __syncthreads();
            if (active) {
                for (size_t key_offset = 0; key_offset < page_tokens; ++key_offset) {
                    const size_t key_base = key_offset * LFM2_HEAD_DIM;
                    float dot =
                        q0 * __bfloat162float(key_tile[key_base + dim0])
                        + q1 * __bfloat162float(key_tile[key_base + dim1]);
                    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                        dot += __shfl_down_sync(0xffffffffU, dot, delta);
                    }
                    dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;
                    const float next_maximum = fmaxf(maximum, dot);
                    const float old_scale = expf(maximum - next_maximum);
                    const float new_scale = expf(dot - next_maximum);
                    accumulator0 = accumulator0 * old_scale
                        + __bfloat162float(value_tile[key_base + dim0]) * new_scale;
                    accumulator1 = accumulator1 * old_scale
                        + __bfloat162float(value_tile[key_base + dim1]) * new_scale;
                    denominator = denominator * old_scale + new_scale;
                    maximum = next_maximum;
                }
            }
            __syncthreads();
        }

        const size_t current_tokens = current_offset + 1;
        for (
            size_t tile_start = 0;
            tile_start < current_tokens;
            tile_start += PAGE_SIZE
        ) {
            const size_t remaining = current_tokens - tile_start;
            const size_t tile_tokens = remaining < PAGE_SIZE ? remaining : PAGE_SIZE;
            const size_t tile_elements = tile_tokens * LFM2_HEAD_DIM;
            for (
                size_t element = threadIdx.x;
                element < tile_elements;
                element += blockDim.x
            ) {
                const size_t offset = element / LFM2_HEAD_DIM;
                const size_t dim = element % LFM2_HEAD_DIM;
                const size_t source =
                    ((segment_begin + tile_start + offset) * LFM2_NUM_KV_HEADS + kv_head)
                    * LFM2_HEAD_DIM
                    + dim;
                key_tile[element] = current_key[source];
                value_tile[element] = current_value[source];
            }
            __syncthreads();
            if (active) {
                for (size_t key_offset = 0; key_offset < tile_tokens; ++key_offset) {
                    const size_t key_base = key_offset * LFM2_HEAD_DIM;
                    float dot =
                        q0 * __bfloat162float(key_tile[key_base + dim0])
                        + q1 * __bfloat162float(key_tile[key_base + dim1]);
                    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                        dot += __shfl_down_sync(0xffffffffU, dot, delta);
                    }
                    dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;
                    const float next_maximum = fmaxf(maximum, dot);
                    const float old_scale = expf(maximum - next_maximum);
                    const float new_scale = expf(dot - next_maximum);
                    accumulator0 = accumulator0 * old_scale
                        + __bfloat162float(value_tile[key_base + dim0]) * new_scale;
                    accumulator1 = accumulator1 * old_scale
                        + __bfloat162float(value_tile[key_base + dim1]) * new_scale;
                    denominator = denominator * old_scale + new_scale;
                    maximum = next_maximum;
                }
            }
            __syncthreads();
        }

        if (active) {
            output[query_base + dim0] =
                __float2bfloat16_rn(accumulator0 / denominator);
            output[query_base + dim1] =
                __float2bfloat16_rn(accumulator1 / denominator);
        }
    }
}

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void hybrid_ragged_gqa_lfm2_bf16_ps16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ current_key,
    const __nv_bfloat16* __restrict__ current_value,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    const uint32_t* __restrict__ segment_offsets,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t num_segments
) {
    hybrid_ragged_gqa_lfm2_bf16_body<16>(
        query, current_key, current_value, key_cache, value_cache, block_table,
        request_slots, position_ids, segment_offsets, output, num_tokens,
        num_pages, block_table_stride, num_segments
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void hybrid_ragged_gqa_lfm2_bf16_ps32(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ current_key,
    const __nv_bfloat16* __restrict__ current_value,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    const uint32_t* __restrict__ segment_offsets,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t num_segments
) {
    hybrid_ragged_gqa_lfm2_bf16_body<32>(
        query, current_key, current_value, key_cache, value_cache, block_table,
        request_slots, position_ids, segment_offsets, output, num_tokens,
        num_pages, block_table_stride, num_segments
    );
}
