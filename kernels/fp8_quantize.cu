#include<cuda_bf16.h>
#include<cuda_fp8.h>
#include<stddef.h>

constexpr int FP8_QUANTIZE_BLOCK_SIZE = 256;

extern "C" __global__
__launch_bounds__(FP8_QUANTIZE_BLOCK_SIZE)
void quantize_bf16_e4m3(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    size_t numel,
    float scale
) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t pair_count = numel >> 1;
    const __nv_bfloat162* __restrict__ input_vec =
        reinterpret_cast<const __nv_bfloat162*>(input);
    uchar2* __restrict__ output_vec = reinterpret_cast<uchar2*>(output);

    for (size_t pair = tid; pair < pair_count; pair += stride) {
        const float2 value = __bfloat1622float2(input_vec[pair]);
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
        output_vec[pair] = quantized;
    }

    if ((numel & 1) != 0 && tid == 0) {
        output[numel - 1] = __nv_cvt_float_to_fp8(
            __bfloat162float(input[numel - 1]) * scale,
            __NV_SATFINITE,
            __NV_E4M3
        );
    }
}
