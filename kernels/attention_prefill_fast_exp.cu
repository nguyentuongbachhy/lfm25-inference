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

// Research-only copy of the production Q2 contiguous prefill kernel.
// The only intentional arithmetic change is expf -> __expf in the
// branch-free online-softmax recurrence.
extern "C" __global__
__launch_bounds__(ATTENTION_MAX_BLOCK_SIZE)
void prefill_gqa_lfm2_bf16_fast_exp(
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
                    const float old_scale = __expf(maximum - next_maximum);
                    const float new_scale = __expf(dot - next_maximum);
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
