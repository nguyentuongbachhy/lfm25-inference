#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>

template <int ITEMS_PER_THREAD>
__device__ __forceinline__
void embedding_bf16_body(
    const uint32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    size_t num_tokens,
    size_t vocab_size,
    size_t hidden_size
) {
    const size_t token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const uint32_t token_id = __ldg(token_ids + token_idx);
    if (static_cast<size_t>(token_id) >= vocab_size) return;

    const __nv_bfloat16* __restrict__ src =
        weight + static_cast<size_t>(token_id) * hidden_size;
    __nv_bfloat16* __restrict__ dst =
        out + token_idx * hidden_size;

    const size_t tid = threadIdx.x;
    const size_t block_size = blockDim.x;

    if ((hidden_size & 1) == 0) {
        using VecT = __nv_bfloat162;
        const size_t vec_count = hidden_size >> 1;

        const VecT* __restrict__ src_vec =
            reinterpret_cast<const VecT*>(src);
        VecT* __restrict__ dst_vec =
            reinterpret_cast<VecT*>(dst);

        const size_t full_tiles = vec_count / ITEMS_PER_THREAD;
        const size_t remainder = vec_count % ITEMS_PER_THREAD;

        for (size_t tile = tid; tile < full_tiles; tile += block_size) {
            const size_t base = tile * ITEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                dst_vec[base + i] = src_vec[base + i];
            }
        }

        if (remainder > 0) {
            const size_t base = full_tiles * ITEMS_PER_THREAD;
            for (size_t i = tid; i < remainder; i += block_size) {
                dst_vec[base + i] = src_vec[base + i];
            }
        }
        return;
    }

    for (size_t i = tid; i < hidden_size; i += block_size) {
        dst[i] = __ldg(src + i);
    }
}

extern "C" __global__
__launch_bounds__(256)
void embedding_bf16(
    const uint32_t* token_ids,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    size_t num_tokens,
    size_t vocab_size,
    size_t hidden_size
) {
    embedding_bf16_body<4>(
        token_ids,
        weight,
        out,
        num_tokens,
        vocab_size,
        hidden_size
    );
}