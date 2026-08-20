#include<cuda_bf16.h>
#include<cuda_fp8.h>
#include<stddef.h>

constexpr int SILU_MUL_BLOCK_SIZE = 256;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
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
