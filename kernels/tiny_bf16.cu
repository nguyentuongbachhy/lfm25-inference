#include <cuda_bf16.h>

#include <stddef.h>
#include <stdint.h>

constexpr int TINY_BF16_BLOCK_THREADS = 256;
constexpr int TINY_BF16_WARP_SIZE = 32;
constexpr int TINY_BF16_WARPS_PER_BLOCK =
    TINY_BF16_BLOCK_THREADS / TINY_BF16_WARP_SIZE;
constexpr int TINY_BF16_MAX_M = 8;

// Test-only latency-oriented BF16 NT kernel for decode shapes with M <= 8.
// Each warp owns one output channel and reuses that weight row across every
// active input row. The K reduction uses BF16xBF16 products with FP32
// accumulation, then rounds once to BF16 at the output, matching the numerical
// contract of the cuBLASLt reference without quantization.
extern "C" __global__
__launch_bounds__(TINY_BF16_BLOCK_THREADS)
void tiny_bf16_nt_m8(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    size_t m,
    size_t n,
    size_t k
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const size_t output_channel =
        static_cast<size_t>(blockIdx.x) * TINY_BF16_WARPS_PER_BLOCK + warp;
    if (output_channel >= n) {
        return;
    }

    float accum[TINY_BF16_MAX_M];
    #pragma unroll
    for (int row = 0; row < TINY_BF16_MAX_M; ++row) {
        accum[row] = 0.0f;
    }

    const __nv_bfloat162* __restrict__ weight_pairs =
        reinterpret_cast<const __nv_bfloat162*>(weight + output_channel * k);
    const size_t pair_count = k >> 1U;

    for (size_t pair = lane; pair < pair_count; pair += TINY_BF16_WARP_SIZE) {
        const float2 w = __bfloat1622float2(weight_pairs[pair]);
        #pragma unroll
        for (int row = 0; row < TINY_BF16_MAX_M; ++row) {
            if (static_cast<size_t>(row) < m) {
                const __nv_bfloat162* __restrict__ input_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(input + static_cast<size_t>(row) * k);
                const float2 x = __bfloat1622float2(input_pairs[pair]);
                accum[row] = fmaf(w.x, x.x, accum[row]);
                accum[row] = fmaf(w.y, x.y, accum[row]);
            }
        }
    }

    #pragma unroll
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        #pragma unroll
        for (int row = 0; row < TINY_BF16_MAX_M; ++row) {
            accum[row] += __shfl_down_sync(0xffffffffU, accum[row], offset);
        }
    }

    if (lane == 0U) {
        #pragma unroll
        for (int row = 0; row < TINY_BF16_MAX_M; ++row) {
            if (static_cast<size_t>(row) < m) {
                output[static_cast<size_t>(row) * n + output_channel] =
                    __float2bfloat16_rn(accum[row]);
            }
        }
    }
}
