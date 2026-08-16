#include <cuda_bf16.h>

#include <float.h>
#include <stddef.h>
#include <stdint.h>

constexpr int SAMPLING_MAX_BLOCK_SIZE = 256;

extern "C" __global__
__launch_bounds__(SAMPLING_MAX_BLOCK_SIZE)
void argmax_bf16(
    const __nv_bfloat16* __restrict__ input,
    uint32_t* __restrict__ output,
    size_t numel
) {
    __shared__ float maxima[SAMPLING_MAX_BLOCK_SIZE];
    __shared__ uint32_t indices[SAMPLING_MAX_BLOCK_SIZE];

    float local_maximum = -FLT_MAX;
    uint32_t local_index = 0U;

    for (size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const float value = __bfloat162float(input[index]);

        if (value > local_maximum) {
            local_maximum = value;
            local_index = static_cast<uint32_t>(index);
        }
    }

    maxima[threadIdx.x] = local_maximum;
    indices[threadIdx.x] = local_index;
    __syncthreads();

    if (threadIdx.x == 0) {
        float maximum = maxima[0];
        uint32_t index = indices[0];

        for (uint32_t thread = 1U; thread < blockDim.x; ++thread) {
            if (maxima[thread] > maximum) {
                maximum = maxima[thread];
                index = indices[thread];
            }
        }

        output[0] = index;
    }
}

extern "C" __global__
__launch_bounds__(SAMPLING_MAX_BLOCK_SIZE)
void argmax_rows_bf16(
    const __nv_bfloat16* __restrict__ input,
    uint32_t* __restrict__ output,
    size_t rows,
    size_t columns
) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float maxima[SAMPLING_MAX_BLOCK_SIZE];
    __shared__ uint32_t indices[SAMPLING_MAX_BLOCK_SIZE];
    float local_maximum = -FLT_MAX;
    uint32_t local_index = 0U;
    const size_t row_base = row * columns;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = __bfloat162float(input[row_base + column]);
        if (value > local_maximum) {
            local_maximum = value;
            local_index = static_cast<uint32_t>(column);
        }
    }
    maxima[threadIdx.x] = local_maximum;
    indices[threadIdx.x] = local_index;
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = maxima[0];
        uint32_t index = indices[0];
        for (uint32_t thread = 1U; thread < blockDim.x; ++thread) {
            if (maxima[thread] > maximum) {
                maximum = maxima[thread];
                index = indices[thread];
            }
        }
        output[row] = index;
    }
}
