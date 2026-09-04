#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stddef.h>

constexpr int RESIDUAL_RMS_FP8_BLOCK_SIZE = 256;

__device__ __forceinline__ float residual_rms_fp8_warp_sum(float value) {
    const unsigned mask = __activemask();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int active_lanes = __popc(mask);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(mask, value, offset);
        if (lane + offset < active_lanes) {
            value += other;
        }
    }
    return value;
}

extern "C" __global__
__launch_bounds__(RESIDUAL_RMS_FP8_BLOCK_SIZE)
void residual_rms_norm_bf16_to_e4m3(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual_out,
    unsigned char* __restrict__ normalized_fp8_out,
    size_t rows,
    size_t hidden_size,
    float eps,
    float quant_scale
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const size_t tid = static_cast<size_t>(threadIdx.x);
    const size_t block_size = static_cast<size_t>(blockDim.x);
    const size_t row_offset = row * hidden_size;
    float local_sum = 0.0f;

    if ((hidden_size & 1) == 0) {
        const size_t vec_count = hidden_size >> 1;
        const __nv_bfloat162* __restrict__ residual_vec =
            reinterpret_cast<const __nv_bfloat162*>(residual + row_offset);
        const __nv_bfloat162* __restrict__ update_vec =
            reinterpret_cast<const __nv_bfloat162*>(update + row_offset);
        __nv_bfloat162* __restrict__ output_vec =
            reinterpret_cast<__nv_bfloat162*>(residual_out + row_offset);

        for (size_t index = tid; index < vec_count; index += block_size) {
            const float2 residual_value = __bfloat1622float2(residual_vec[index]);
            const float2 update_value = __bfloat1622float2(update_vec[index]);
            const float sum0 = residual_value.x + update_value.x;
            const float sum1 = residual_value.y + update_value.y;
            output_vec[index] = __floats2bfloat162_rn(sum0, sum1);
            local_sum = fmaf(sum0, sum0, local_sum);
            local_sum = fmaf(sum1, sum1, local_sum);
        }
    } else {
        for (size_t index = tid; index < hidden_size; index += block_size) {
            const size_t offset = row_offset + index;
            const float sum =
                __bfloat162float(residual[offset]) + __bfloat162float(update[offset]);
            residual_out[offset] = __float2bfloat16_rn(sum);
            local_sum = fmaf(sum, sum, local_sum);
        }
    }

    constexpr int MAX_WARPS = RESIDUAL_RMS_FP8_BLOCK_SIZE >> 5;
    __shared__ float warp_sums[MAX_WARPS];
    __shared__ float inv_rms_shared;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;
    const float warp_sum = residual_rms_fp8_warp_sum(local_sum);

    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < num_warps ? warp_sums[lane] : 0.0f;
        block_sum = residual_rms_fp8_warp_sum(block_sum);
        if (lane == 0) {
            const float variance = block_sum / static_cast<float>(hidden_size);
            inv_rms_shared = __frsqrt_rn(variance + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_shared;
    if ((hidden_size & 1) == 0) {
        const size_t vec_count = hidden_size >> 1;
        const __nv_bfloat162* __restrict__ residual_vec =
            reinterpret_cast<const __nv_bfloat162*>(residual_out + row_offset);
        const __nv_bfloat162* __restrict__ weight_vec =
            reinterpret_cast<const __nv_bfloat162*>(weight);
        uchar2* __restrict__ fp8_vec =
            reinterpret_cast<uchar2*>(normalized_fp8_out + row_offset);

        for (size_t index = tid; index < vec_count; index += block_size) {
            const float2 value = __bfloat1622float2(residual_vec[index]);
            const float2 scale = __bfloat1622float2(weight_vec[index]);
            const __nv_bfloat162 normalized = __floats2bfloat162_rn(
                value.x * inv_rms * scale.x,
                value.y * inv_rms * scale.y
            );
            const float2 rounded = __bfloat1622float2(normalized);
            uchar2 quantized;
            quantized.x = __nv_cvt_float_to_fp8(
                rounded.x * quant_scale,
                __NV_SATFINITE,
                __NV_E4M3
            );
            quantized.y = __nv_cvt_float_to_fp8(
                rounded.y * quant_scale,
                __NV_SATFINITE,
                __NV_E4M3
            );
            fp8_vec[index] = quantized;
        }
    } else {
        for (size_t index = tid; index < hidden_size; index += block_size) {
            const size_t offset = row_offset + index;
            const float value = __bfloat162float(residual_out[offset]);
            const float scale = __bfloat162float(weight[index]);
            const __nv_bfloat16 normalized =
                __float2bfloat16_rn(value * inv_rms * scale);
            normalized_fp8_out[offset] = __nv_cvt_float_to_fp8(
                __bfloat162float(normalized) * quant_scale,
                __NV_SATFINITE,
                __NV_E4M3
            );
        }
    }
}
