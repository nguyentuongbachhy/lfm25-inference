#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int MXFP8_BLOCK_THREADS = 256;
constexpr int MXFP8_WARP_SIZE = 32;
constexpr float E4M3_MAX_FINITE = 448.0f;

extern "C" __global__
__launch_bounds__(MXFP8_BLOCK_THREADS)
void quantize_bf16_mxfp8_vec32(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    unsigned char* __restrict__ scales,
    size_t outer,
    size_t inner,
    size_t scale_storage_len
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t warps_per_block = blockDim.x >> 5U;
    const size_t logical_block =
        static_cast<size_t>(blockIdx.x) * warps_per_block + warp;
    const size_t inner_blocks = inner >> 5U;
    const size_t total_blocks = outer * inner_blocks;
    if (logical_block >= total_blocks) {
        return;
    }

    const size_t outer_index = logical_block / inner_blocks;
    const size_t inner_block = logical_block - outer_index * inner_blocks;
    const size_t column = inner_block * MXFP8_WARP_SIZE + lane;
    const size_t index = outer_index * inner + column;

    const float value = __bfloat162float(input[index]);
    float amax = fabsf(value);
    #pragma unroll
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffU, amax, offset));
    }

    uint32_t scale_bits = 127U;
    if (lane == 0U && amax > 0.0f) {
        const float raw_scale = amax / E4M3_MAX_FINITE;
        scale_bits = static_cast<uint32_t>(__nv_cvt_float_to_e8m0(
            raw_scale,
            __NV_SATFINITE,
            cudaRoundPosInf
        ));
        if (scale_bits == 0xffU) {
            scale_bits = 0xfeU;
        }
    }
    scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
    const float scale = ldexpf(1.0f, static_cast<int>(scale_bits) - 127);
    output[index] = __nv_cvt_float_to_fp8(
        value / scale,
        __NV_SATFINITE,
        __NV_E4M3
    );

    if (lane == 0U) {
        const size_t outer_tile = outer_index >> 7U;
        const size_t local_outer = outer_index & 127U;
        const size_t inner_tile = inner_block >> 2U;
        const size_t local_inner = inner_block & 3U;
        const size_t inner_tiles = (inner_blocks + 3U) >> 2U;
        const size_t tile = outer_tile * inner_tiles + inner_tile;
        const size_t local_offset =
            (local_outer & 31U) * 16U + (local_outer >> 5U) * 4U + local_inner;
        const size_t scale_offset = tile * 512U + local_offset;
        if (scale_offset < scale_storage_len) {
            scales[scale_offset] = static_cast<unsigned char>(scale_bits);
        }
    }
}
