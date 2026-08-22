#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

constexpr int INT8_QUANT_BLOCK = 256;
constexpr int INT8_WARPS_PER_BLOCK = 4;
constexpr int INT8_MAX_M = 8;

__device__ __forceinline__ int clamp_s8(int value) {
    return max(-127, min(127, value));
}

extern "C" __global__
__launch_bounds__(INT8_QUANT_BLOCK)
void quantize_bf16_rows_s8(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    size_t rows,
    size_t cols
) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float maxima[INT8_QUANT_BLOCK];
    float local_max = 0.0f;
    const size_t base = row * cols;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(input[base + col])));
    }
    maxima[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = maxima[0] > 0.0f ? maxima[0] / 127.0f : 1.0f;
    const float inverse_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }

    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = __bfloat162float(input[base + col]);
        output[base + col] = static_cast<int8_t>(clamp_s8(__float2int_rn(value * inverse_scale)));
    }
}

// Decode-specialized W8A8 GEMV/GEMM hybrid for M <= 8. One warp owns one
// output channel and reuses the packed INT8 weight row across every active M
// row. K is consumed four elements at a time through __dp4a. This deliberately
// optimizes the autoregressive tiny-M regime rather than padding M to a tensor-
// core tile and paying for inactive rows.
extern "C" __global__
void int8_tiny_m_dp4a_bf16(
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    __nv_bfloat16* __restrict__ output,
    size_t m,
    size_t n,
    size_t k
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const size_t output_channel = static_cast<size_t>(blockIdx.x) * INT8_WARPS_PER_BLOCK + warp;
    if (output_channel >= n || m == 0 || m > INT8_MAX_M || (k & 3U) != 0U) {
        return;
    }

    const size_t k4 = k >> 2;
    const int32_t* __restrict__ packed_weight =
        reinterpret_cast<const int32_t*>(weight + output_channel * k);
    int32_t accumulators[INT8_MAX_M] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t index = lane; index < k4; index += 32U) {
        const int32_t w4 = packed_weight[index];
        #pragma unroll
        for (uint32_t row = 0; row < INT8_MAX_M; ++row) {
            if (row < m) {
                const int32_t* __restrict__ packed_input =
                    reinterpret_cast<const int32_t*>(input + static_cast<size_t>(row) * k);
                accumulators[row] = __dp4a(packed_input[index], w4, accumulators[row]);
            }
        }
    }

    #pragma unroll
    for (uint32_t row = 0; row < INT8_MAX_M; ++row) {
        if (row < m) {
            int32_t sum = accumulators[row];
            #pragma unroll
            for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                sum += __shfl_down_sync(0xffffffffU, sum, delta);
            }
            if (lane == 0U) {
                const float scale = input_scales[row] * weight_scales[output_channel];
                output[static_cast<size_t>(row) * n + output_channel] =
                    __float2bfloat16_rn(static_cast<float>(sum) * scale);
            }
        }
    }
}
