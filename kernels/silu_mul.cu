#include<cuda_bf16.h>
#include<stddef.h>

constexpr int SILU_MUL_BLOCK_SIZE = 256;
constexpr int SILU_MUL_ITEMS_PER_THREAD = 4;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void silu_bf16_body(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    size_t numel
) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    constexpr size_t VEC_WIDTH = 2;
    const size_t vec_count = numel / VEC_WIDTH;

    const size_t full_tiles = vec_count / ITEMS_PER_THREAD;
    const size_t remainder = vec_count % ITEMS_PER_THREAD;

    const __nv_bfloat162* __restrict__ gate_vec =
        reinterpret_cast<const __nv_bfloat162*>(gate);
    const __nv_bfloat162* __restrict__ up_vec =
        reinterpret_cast<const __nv_bfloat162*>(up);
    __nv_bfloat162* __restrict__ out_vec =
        reinterpret_cast<__nv_bfloat162*>(out);

    for (size_t tile = tid; tile < full_tiles; tile += stride) {
        const size_t base = tile * ITEMS_PER_THREAD;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const size_t idx = base + i;

            const float2 g = __bfloat1622float2(gate_vec[idx]);
            const float2 u = __bfloat1622float2(up_vec[idx]);

            const float out_x = silu_f32(g.x) * u.x;
            const float out_y = silu_f32(g.y) * u.y;

            out_vec[idx] = __floats2bfloat162_rn(out_x, out_y);
        }
    }

    if (remainder > 0) {
        const size_t base = full_tiles * ITEMS_PER_THREAD;
        for (size_t i = tid; i < remainder; i += stride) {
            const size_t idx = base + i;

            const float2 g = __bfloat1622float2(gate_vec[idx]);
            const float2 u = __bfloat1622float2(up_vec[idx]);

            const float out_x = silu_f32(g.x) * u.x;
            const float out_y = silu_f32(g.y) * u.y;

            out_vec[idx] = __floats2bfloat162_rn(out_x, out_y);
        }
    }

    const size_t tail_start = vec_count * VEC_WIDTH;
    const size_t tail_count = numel - tail_start;
    for (size_t i = tid; i < tail_count; i += stride) {
        const size_t idx = tail_start + i;

        const float g = __bfloat162float(gate[idx]);
        const float u = __bfloat162float(up[idx]);

        out[idx] = __float2bfloat16_rn(silu_f32(g) * u);
    }
}

extern "C" __global__
__launch_bounds__(SILU_MUL_BLOCK_SIZE)
void silu_mul_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    size_t numel
) {
    silu_bf16_body<SILU_MUL_ITEMS_PER_THREAD>(gate, up, out, numel);
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
