#include<cuda_bf16.h>
#include<cuda_fp8.h>
#include<stddef.h>
#include<stdint.h>

constexpr int SILU_MUL_BLOCK_SIZE = 256;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ int clamp_s8(int value) {
    return max(-127, min(127, value));
}

extern "C" __global__
__launch_bounds__(SILU_MUL_BLOCK_SIZE)
void silu_mul_packed_bf16(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ out,
    size_t rows,
    size_t intermediate_size
) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    if ((intermediate_size & 1) == 0) {
        const size_t pairs_per_row = intermediate_size >> 1;
        const size_t output_pairs = rows * pairs_per_row;
        const __nv_bfloat162* __restrict__ packed_vec =
            reinterpret_cast<const __nv_bfloat162*>(packed);
        __nv_bfloat162* __restrict__ out_vec =
            reinterpret_cast<__nv_bfloat162*>(out);

        for (size_t pair = tid; pair < output_pairs; pair += stride) {
            const size_t row = pair / pairs_per_row;
            const size_t column = pair % pairs_per_row;
            const size_t packed_row = row * (pairs_per_row << 1);
            const float2 gate = __bfloat1622float2(
                packed_vec[packed_row + column]
            );
            const float2 up = __bfloat1622float2(
                packed_vec[packed_row + pairs_per_row + column]
            );
            out_vec[pair] = __floats2bfloat162_rn(
                silu_f32(gate.x) * up.x,
                silu_f32(gate.y) * up.y
            );
        }
        return;
    }

    const size_t output_elements = rows * intermediate_size;
    for (size_t index = tid; index < output_elements; index += stride) {
        const size_t row = index / intermediate_size;
        const size_t column = index % intermediate_size;
        const size_t packed_row = row * (intermediate_size << 1);
        const float gate = __bfloat162float(packed[packed_row + column]);
        const float up = __bfloat162float(
            packed[packed_row + intermediate_size + column]
        );
        out[index] = __float2bfloat16_rn(silu_f32(gate) * up);
    }
}

// Fuses the exact production sequence
//   silu_mul_packed_bf16 -> quantize_bf16_e4m3
// for FP8 MLP-down sites. The intermediate activation is deliberately rounded
// to BF16 before conversion to E4M3 so this kernel preserves the numerical
// semantics of the unfused path instead of silently changing the PTQ policy.
extern "C" __global__
__launch_bounds__(SILU_MUL_BLOCK_SIZE)
void silu_mul_packed_bf16_to_e4m3(
    const __nv_bfloat16* __restrict__ packed,
    unsigned char* __restrict__ out,
    size_t rows,
    size_t intermediate_size,
    float scale
) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    if ((intermediate_size & 1) == 0) {
        const size_t pairs_per_row = intermediate_size >> 1;
        const size_t output_pairs = rows * pairs_per_row;
        const __nv_bfloat162* __restrict__ packed_vec =
            reinterpret_cast<const __nv_bfloat162*>(packed);
        uchar2* __restrict__ out_vec = reinterpret_cast<uchar2*>(out);

        for (size_t pair = tid; pair < output_pairs; pair += stride) {
            const size_t row = pair / pairs_per_row;
            const size_t column = pair % pairs_per_row;
            const size_t packed_row = row * (pairs_per_row << 1);
            const float2 gate = __bfloat1622float2(
                packed_vec[packed_row + column]
            );
            const float2 up = __bfloat1622float2(
                packed_vec[packed_row + pairs_per_row + column]
            );
            const __nv_bfloat162 rounded = __floats2bfloat162_rn(
                silu_f32(gate.x) * up.x,
                silu_f32(gate.y) * up.y
            );
            const float2 value = __bfloat1622float2(rounded);
            uchar2 quantized;
            quantized.x = __nv_cvt_float_to_fp8(
                value.x * scale,
                __NV_SATFINITE,
                __NV_E4M3
            );
            quantized.y = __nv_cvt_float_to_fp8(
                value.y * scale,
                __NV_SATFINITE,
                __NV_E4M3
            );
            out_vec[pair] = quantized;
        }
        return;
    }

    const size_t output_elements = rows * intermediate_size;
    for (size_t index = tid; index < output_elements; index += stride) {
        const size_t row = index / intermediate_size;
        const size_t column = index % intermediate_size;
        const size_t packed_row = row * (intermediate_size << 1);
        const float gate = __bfloat162float(packed[packed_row + column]);
        const float up = __bfloat162float(
            packed[packed_row + intermediate_size + column]
        );
        const __nv_bfloat16 rounded =
            __float2bfloat16_rn(silu_f32(gate) * up);
        out[index] = __nv_cvt_float_to_fp8(
            __bfloat162float(rounded) * scale,
            __NV_SATFINITE,
            __NV_E4M3
        );
    }
}

// Decode-tiny-M fusion for the custom W8A8 down projection. One CTA owns one
// row so it can compute the exact post-SwiGLU BF16 activation, reduce its amax,
// and quantize to signed INT8 without materializing the BF16 activation in
// global memory. The BF16 rounding before amax/quantization deliberately
// matches silu_mul_packed_bf16 -> quantize_bf16_rows_s8 bit-for-bit.
extern "C" __global__
__launch_bounds__(SILU_MUL_BLOCK_SIZE)
void silu_mul_packed_bf16_to_s8_dynamic(
    const __nv_bfloat16* __restrict__ packed,
    int8_t* __restrict__ out,
    float* __restrict__ scales,
    size_t rows,
    size_t intermediate_size
) {
    const size_t row = blockIdx.x;
    if (row >= rows || intermediate_size == 0) {
        return;
    }

    extern __shared__ __nv_bfloat16 activated[];
    __shared__ float maxima[SILU_MUL_BLOCK_SIZE];
    const size_t packed_row = row * (intermediate_size << 1);

    float local_max = 0.0f;
    for (size_t column = threadIdx.x; column < intermediate_size; column += blockDim.x) {
        const float gate = __bfloat162float(packed[packed_row + column]);
        const float up = __bfloat162float(packed[packed_row + intermediate_size + column]);
        const __nv_bfloat16 rounded = __float2bfloat16_rn(silu_f32(gate) * up);
        activated[column] = rounded;
        local_max = fmaxf(local_max, fabsf(__bfloat162float(rounded)));
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
    __syncthreads();

    const size_t output_base = row * intermediate_size;
    for (size_t column = threadIdx.x; column < intermediate_size; column += blockDim.x) {
        const float value = __bfloat162float(activated[column]);
        out[output_base + column] =
            static_cast<int8_t>(clamp_s8(__float2int_rn(value * inverse_scale)));
    }
}
