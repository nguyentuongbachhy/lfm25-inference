#include <cuda_bf16.h>
#include <stddef.h>

namespace {
constexpr size_t Q_WIDTH = 32 * 64;
constexpr size_t KV_WIDTH = 8 * 64;
constexpr size_t PACKED_WIDTH = Q_WIDTH + 2 * KV_WIDTH;
}

extern "C" __global__ void unpack_qkv_bf16(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    size_t num_tokens) {
    const size_t total = num_tokens * PACKED_WIDTH;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t token = index / PACKED_WIDTH;
        const size_t column = index - token * PACKED_WIDTH;
        const __nv_bfloat16 element = packed[index];
        if (column < Q_WIDTH) {
            query[token * Q_WIDTH + column] = element;
        } else if (column < Q_WIDTH + KV_WIDTH) {
            key[token * KV_WIDTH + (column - Q_WIDTH)] = element;
        } else {
            value[token * KV_WIDTH + (column - Q_WIDTH - KV_WIDTH)] = element;
        }
    }
}
