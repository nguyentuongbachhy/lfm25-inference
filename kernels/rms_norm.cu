#include <cuda_bf16.h>
#include <stddef.h>

constexpr int RMS_NORM_BLOCK_SIZE = 256;
constexpr int RMS_NORM_ITEMS_PER_THREAD = 4;

__device__ __forceinline__
float warp_reduce_sum(float value) {
    const unsigned mask = __activemask();

    const int lane =
        static_cast<int>(threadIdx.x) & 31;

    const int active_lanes =
        __popc(mask);

    #pragma unroll
    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        const float other =
            __shfl_down_sync(
                mask,
                value,
                offset
            );

        if (lane + offset < active_lanes) {
            value += other;
        }
    }

    return value;
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__
void rms_norm_bf16_body(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    size_t rows,
    size_t hidden_size,
    float eps
) {
    const size_t row =
        static_cast<size_t>(blockIdx.x);

    if (row >= rows) {
        return;
    }

    const size_t tid =
        static_cast<size_t>(threadIdx.x);

    const size_t block_size =
        static_cast<size_t>(blockDim.x);

    const __nv_bfloat16* __restrict__ row_x =
        x + row * hidden_size;

    __nv_bfloat16* __restrict__ row_out =
        out + row * hidden_size;

    float local_sum = 0.0f;

    if ((hidden_size & 1) == 0) {
        const size_t vec_count =
            hidden_size >> 1;

        const size_t full_tiles =
            vec_count / ITEMS_PER_THREAD;

        const size_t remainder =
            vec_count % ITEMS_PER_THREAD;

        const __nv_bfloat162* __restrict__ x_vec =
            reinterpret_cast<const __nv_bfloat162*>(row_x);

        for (
            size_t tile = tid;
            tile < full_tiles;
            tile += block_size
        ) {
            const size_t base =
                tile * ITEMS_PER_THREAD;

            #pragma unroll
            for (
                int item = 0;
                item < ITEMS_PER_THREAD;
                ++item
            ) {
                const float2 value =
                    __bfloat1622float2(
                        x_vec[base + item]
                    );

                local_sum =
                    fmaf(
                        value.x,
                        value.x,
                        local_sum
                    );

                local_sum =
                    fmaf(
                        value.y,
                        value.y,
                        local_sum
                    );
            }
        }

        if (remainder != 0) {
            const size_t base =
                full_tiles
                * ITEMS_PER_THREAD;

            for (
                size_t item = tid;
                item < remainder;
                item += block_size
            ) {
                const float2 value =
                    __bfloat1622float2(
                        x_vec[base + item]
                    );

                local_sum =
                    fmaf(
                        value.x,
                        value.x,
                        local_sum
                    );

                local_sum =
                    fmaf(
                        value.y,
                        value.y,
                        local_sum
                    );
            }
        }
    } else {
        for (
            size_t i = tid;
            i < hidden_size;
            i += block_size
        ) {
            const float value =
                __bfloat162float(
                    row_x[i]
                );

            local_sum =
                fmaf(
                    value,
                    value,
                    local_sum
                );
        }
    }

    constexpr int MAX_WARPS =
        RMS_NORM_BLOCK_SIZE >> 5;

    __shared__ float warp_sums[MAX_WARPS];
    __shared__ float inv_rms_shared;

    const int lane =
        static_cast<int>(threadIdx.x)
        & 31;

    const int warp_id =
        static_cast<int>(threadIdx.x)
        >> 5;

    const int num_warps =
        (
            static_cast<int>(blockDim.x)
            + 31
        ) >> 5;

    float warp_sum =
        warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_sums[warp_id] =
            warp_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        float block_sum =
            lane < num_warps
                ? warp_sums[lane]
                : 0.0f;

        block_sum =
            warp_reduce_sum(
                block_sum
            );

        if (lane == 0) {
            const float variance =
                block_sum
                / static_cast<float>(
                    hidden_size
                );

            inv_rms_shared =
                __frsqrt_rn(
                    variance + eps
                );
        }
    }

    __syncthreads();

    const float inv_rms =
        inv_rms_shared;

    if ((hidden_size & 1) == 0) {
        const size_t vec_count =
            hidden_size >> 1;

        const size_t full_tiles =
            vec_count
            / ITEMS_PER_THREAD;

        const size_t remainder =
            vec_count
            % ITEMS_PER_THREAD;

        const __nv_bfloat162* __restrict__ x_vec =
            reinterpret_cast<const __nv_bfloat162*>(row_x);
        const __nv_bfloat162* __restrict__ weight_vec =
            reinterpret_cast<const __nv_bfloat162*>(weight);
        __nv_bfloat162* __restrict__ out_vec =
            reinterpret_cast<__nv_bfloat162*>(row_out);

        for (
            size_t tile = tid;
            tile < full_tiles;
            tile += block_size
        ) {
            const size_t base =
                tile * ITEMS_PER_THREAD;

            #pragma unroll
            for (
                int item = 0;
                item < ITEMS_PER_THREAD;
                ++item
            ) {
                const size_t idx =
                    base + item;

                const float2 value =
                    __bfloat1622float2(
                        x_vec[idx]
                    );

                const __nv_bfloat162 normalized =
                    __floats2bfloat162_rn(
                        value.x * inv_rms,
                        value.y * inv_rms
                    );

                out_vec[idx] =
                    __hmul2(
                        normalized,
                        weight_vec[idx]
                    );
            }
        }

        if (remainder != 0) {
            const size_t base =
                full_tiles
                * ITEMS_PER_THREAD;

            for (
                size_t item = tid;
                item < remainder;
                item += block_size
            ) {
                const size_t idx =
                    base + item;

                const float2 value =
                    __bfloat1622float2(
                        x_vec[idx]
                    );

                const __nv_bfloat162 normalized =
                    __floats2bfloat162_rn(
                        value.x * inv_rms,
                        value.y * inv_rms
                    );

                out_vec[idx] =
                    __hmul2(
                        normalized,
                        weight_vec[idx]
                    );
            }
        }

        return;
    }

    for (
        size_t i = tid;
        i < hidden_size;
        i += block_size
    ) {
        const float value =
            __bfloat162float(
                row_x[i]
            );

        const __nv_bfloat16 normalized =
            __float2bfloat16_rn(
                value * inv_rms
            );

        row_out[i] =
            __hmul(
                normalized,
                weight[i]
            );
    }
}

extern "C" __global__
__launch_bounds__(RMS_NORM_BLOCK_SIZE)
void rms_norm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    size_t rows,
    size_t hidden_size,
    float eps
) {
    rms_norm_bf16_body<
        RMS_NORM_ITEMS_PER_THREAD
    >(
        x,
        weight,
        out,
        rows,
        hidden_size,
        eps
    );
}

extern "C" __global__
__launch_bounds__(RMS_NORM_BLOCK_SIZE)
void residual_rms_norm_bf16(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ normalized_out,
    size_t rows,
    size_t hidden_size,
    float eps
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
                __bfloat162float(residual[offset])
                + __bfloat162float(update[offset]);
            residual_out[offset] = __float2bfloat16_rn(sum);
            local_sum = fmaf(sum, sum, local_sum);
        }
    }

    constexpr int MAX_WARPS = RMS_NORM_BLOCK_SIZE >> 5;
    __shared__ float warp_sums[MAX_WARPS];
    __shared__ float inv_rms_shared;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;
    const float warp_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < num_warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
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
        __nv_bfloat162* __restrict__ normalized_vec =
            reinterpret_cast<__nv_bfloat162*>(normalized_out + row_offset);

        for (size_t index = tid; index < vec_count; index += block_size) {
            const float2 value = __bfloat1622float2(residual_vec[index]);
            const float2 scale = __bfloat1622float2(weight_vec[index]);
            normalized_vec[index] = __floats2bfloat162_rn(
                value.x * inv_rms * scale.x,
                value.y * inv_rms * scale.y
            );
        }
    } else {
        for (size_t index = tid; index < hidden_size; index += block_size) {
            const size_t offset = row_offset + index;
            const float value = __bfloat162float(residual_out[offset]);
            const float scale = __bfloat162float(weight[index]);
            normalized_out[offset] = __float2bfloat16_rn(value * inv_rms * scale);
        }
    }
}
