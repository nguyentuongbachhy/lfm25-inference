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
constexpr uint32_t PREFILL_QUERY_TILE_Q4 = 4U;
constexpr uint32_t PREFILL_KEY_TILE = 32U;

// Research-only Q4 contiguous causal GQA prefill kernel.
//
// Production Q2 assigns one query/head state to each of the eight warps. Q4
// keeps two query states per warp. Both states have the same GQA q-head offset,
// so one K/V tile loaded into shared memory is reused by four query tokens
// instead of two. Each query still scans keys in the same increasing order and
// uses the same online-softmax update as the production Q2 kernel.
extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void prefill_gqa_lfm2_bf16_q4(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens
) {
    const size_t query_tile = blockIdx.x / LFM2_NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % LFM2_NUM_KV_HEADS;
    const size_t query_start = query_tile * PREFILL_QUERY_TILE_Q4;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;

    // Eight warps, sixteen tasks. Warp w owns task w and task w+8. This maps
    // to the same q-head offset at query offsets {0,2} or {1,3}.
    constexpr uint32_t STATES_PER_WARP = 2U;
    bool active[STATES_PER_WARP];
    size_t token[STATES_PER_WARP];
    size_t query_base[STATES_PER_WARP];
    float q0[STATES_PER_WARP];
    float q1[STATES_PER_WARP];
    float maximum[STATES_PER_WARP];
    float denominator[STATES_PER_WARP];
    float accumulator0[STATES_PER_WARP];
    float accumulator1[STATES_PER_WARP];

    const size_t dim0 = lane;
    const size_t dim1 = lane + 32U;

    #pragma unroll
    for (uint32_t state = 0U; state < STATES_PER_WARP; ++state) {
        const uint32_t task = warp + state * 8U;
        const uint32_t query_offset = task / LFM2_Q_PER_KV;
        const uint32_t q_offset = task % LFM2_Q_PER_KV;
        token[state] = query_start + query_offset;
        active[state] = token[state] < num_tokens;
        const uint32_t q_head = kv_head * LFM2_Q_PER_KV + q_offset;
        query_base[state] =
            (token[state] * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;
        q0[state] = active[state]
            ? __bfloat162float(query[query_base[state] + dim0])
            : 0.0f;
        q1[state] = active[state]
            ? __bfloat162float(query[query_base[state] + dim1])
            : 0.0f;
        maximum[state] = -INFINITY;
        denominator[state] = 0.0f;
        accumulator0[state] = 0.0f;
        accumulator1[state] = 0.0f;
    }

    __shared__ __nv_bfloat16 key_tile[PREFILL_KEY_TILE * LFM2_HEAD_DIM];
    __shared__ __nv_bfloat16 value_tile[PREFILL_KEY_TILE * LFM2_HEAD_DIM];

    const size_t max_query_position =
        query_start + PREFILL_QUERY_TILE_Q4 - 1 < num_tokens
            ? query_start + PREFILL_QUERY_TILE_Q4 - 1
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

        #pragma unroll
        for (uint32_t state = 0U; state < STATES_PER_WARP; ++state) {
            if (active[state] && key_start <= token[state]) {
                const size_t valid_tokens = token[state] + 1 - key_start < tile_tokens
                    ? token[state] + 1 - key_start
                    : tile_tokens;

                for (size_t key_offset = 0; key_offset < valid_tokens; ++key_offset) {
                    const size_t key_base = key_offset * LFM2_HEAD_DIM;
                    const size_t index0 = key_base + dim0;
                    const size_t index1 = key_base + dim1;
                    float dot =
                        q0[state] * __bfloat162float(key_tile[index0])
                        + q1[state] * __bfloat162float(key_tile[index1]);

                    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                        dot += __shfl_down_sync(0xffffffffU, dot, delta);
                    }

                    dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;
                    const float next_maximum = fmaxf(maximum[state], dot);
                    const float old_scale = expf(maximum[state] - next_maximum);
                    const float new_scale = expf(dot - next_maximum);
                    const float value0 = __bfloat162float(value_tile[index0]);
                    const float value1 = __bfloat162float(value_tile[index1]);
                    accumulator0[state] =
                        accumulator0[state] * old_scale + value0 * new_scale;
                    accumulator1[state] =
                        accumulator1[state] * old_scale + value1 * new_scale;
                    denominator[state] = denominator[state] * old_scale + new_scale;
                    maximum[state] = next_maximum;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t state = 0U; state < STATES_PER_WARP; ++state) {
        if (active[state]) {
            output[query_base[state] + dim0] =
                __float2bfloat16_rn(accumulator0[state] / denominator[state]);
            output[query_base[state] + dim1] =
                __float2bfloat16_rn(accumulator1[state] / denominator[state]);
        }
    }
}
