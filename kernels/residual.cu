#include <cuda_bf16.h>

#include <stddef.h>

constexpr int RESIDUAL_MAX_BLOCK_SIZE = 256;
constexpr size_t RESIDUAL_ITEMS_PER_THREAD = 4;

extern "C" __global__
__launch_bounds__(RESIDUAL_MAX_BLOCK_SIZE)
void residual_add_bf16(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    __nv_bfloat16* __restrict__ output,
    size_t numel
) {
    const size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t pair_count = numel >> 1;
    const size_t tile_count = pair_count / RESIDUAL_ITEMS_PER_THREAD;

    const __nv_bfloat162* __restrict__ residual_pairs =
        reinterpret_cast<const __nv_bfloat162*>(residual);
    const __nv_bfloat162* __restrict__ update_pairs =
        reinterpret_cast<const __nv_bfloat162*>(update);
    __nv_bfloat162* __restrict__ output_pairs =
        reinterpret_cast<__nv_bfloat162*>(output);

    for (size_t tile = thread; tile < tile_count; tile += stride) {
        const size_t base = tile * RESIDUAL_ITEMS_PER_THREAD;

        #pragma unroll
        for (size_t item = 0; item < RESIDUAL_ITEMS_PER_THREAD; ++item) {
            const size_t index = base + item;
            output_pairs[index] = __hadd2(residual_pairs[index], update_pairs[index]);
        }
    }

    const size_t remaining_pair_start = tile_count * RESIDUAL_ITEMS_PER_THREAD;
    for (
        size_t pair = remaining_pair_start + thread;
        pair < pair_count;
        pair += stride
    ) {
        output_pairs[pair] = __hadd2(residual_pairs[pair], update_pairs[pair]);
    }

    if ((numel & 1U) != 0U && thread == 0) {
        const size_t tail = numel - 1;
        output[tail] = __hadd(residual[tail], update[tail]);
    }
}
