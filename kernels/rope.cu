#include<cuda_bf16.h>

#include<math.h>
#include<stddef.h>
#include<stdint.h>

constexpr int ROPE_MAX_BLOCK_SIZE = 256;
constexpr size_t ROPE_MAX_HEAD_DIM = 512;
constexpr size_t ROPE_MAX_PAIRS = ROPE_MAX_HEAD_DIM >> 1;

extern "C" __global__ __launch_bounds__(ROPE_MAX_BLOCK_SIZE) void rope_qk_bf16_inplace(
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    const float* __restrict__ inv_freq,
    const uint32_t* __restrict__ position_ids,
    size_t num_tokens,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    if (head_dim == 0 || (head_dim & 1ULL) != 0 || head_dim > ROPE_MAX_HEAD_DIM) {
        return;
    }

    const size_t token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const size_t tid = threadIdx.x;
    const size_t bsize = blockDim.x;
    const size_t half_dim = head_dim >> 1;

    __shared__ float cos_cache[ROPE_MAX_PAIRS];
    __shared__ float sin_cache[ROPE_MAX_PAIRS];

    const uint32_t position = position_ids[token];

    for (size_t pair = tid; pair < half_dim; pair += bsize) {
        float angle = position * inv_freq[pair];
        sincosf(angle, &sin_cache[pair], &cos_cache[pair]);
    }

    __syncthreads();

    const size_t q_pairs = num_q_heads * half_dim;
    for (size_t work = tid; work < q_pairs; work += bsize) {
        const size_t head = work / half_dim;
        const size_t pair = work - head * half_dim;
        const size_t base = (token * num_q_heads + head) * head_dim;

        const float cos_val = cos_cache[pair];
        const float sin_val = sin_cache[pair];

        const size_t idx1 = base + pair;
        const size_t idx2 = idx1 + half_dim;

        float x1 = __bfloat162float(query[idx1]);
        float x2 = __bfloat162float(query[idx2]);

        query[idx1] = __float2bfloat16_rn(fmaf(-x2, sin_val, x1 * cos_val));
        query[idx2] = __float2bfloat16_rn(fmaf(x1, sin_val, x2 * cos_val));
    }

    const size_t k_pairs = num_kv_heads * half_dim;
    for (size_t work = tid; work < k_pairs; work += bsize) {
        const size_t head = work / half_dim;
        const size_t pair = work - head * half_dim;
        const size_t base = (token * num_kv_heads + head) * head_dim;

        const float cos_val = cos_cache[pair];
        const float sin_val = sin_cache[pair];

        const size_t idx1 = base + pair;
        const size_t idx2 = idx1 + half_dim;

        float x1 = __bfloat162float(key[idx1]);
        float x2 = __bfloat162float(key[idx2]);

        key[idx1] = __float2bfloat16_rn(fmaf(-x2, sin_val, x1 * cos_val));
        key[idx2] = __float2bfloat16_rn(fmaf(x1, sin_val, x2 * cos_val));
    }
}
