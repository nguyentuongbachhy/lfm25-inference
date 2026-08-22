#include <cuda_bf16.h>

#include <float.h>
#include <stddef.h>
#include <stdint.h>

constexpr int SAMPLING_MAX_BLOCK_SIZE = 256;
constexpr int ARGMAX_ATOMIC_BLOCKS_PER_ROW = 32;

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

// Pack one BF16 logit together with the exact legacy argmax tie priority into
// a sortable 32-bit key. Production vocab is <= 65536, so eight bits are
// sufficient for both the legacy logical lane (column % 256) and the offset
// within that lane (column / 256).
//
// Legacy behavior that must be preserved:
// - each logical lane scans columns in increasing order and replaces only on >;
// - the final serial reduction prefers the smaller logical lane on equal values;
// - NaN and -inf never beat the -FLT_MAX initialization;
// - +0 and -0 compare equal and are therefore resolved only by tie priority.
__device__ __forceinline__ uint32_t argmax_atomic_key(
    const __nv_bfloat16* __restrict__ input,
    size_t index,
    uint32_t column
) {
    uint16_t bits = reinterpret_cast<const uint16_t*>(input)[index];
    const uint16_t magnitude = bits & 0x7fffU;

    // Ignore NaN exactly as the legacy strict-float comparison does.
    if ((magnitude & 0x7f80U) == 0x7f80U && (magnitude & 0x007fU) != 0U) {
        return 0U;
    }
    // -inf never beats the legacy -FLT_MAX initialization.
    if (bits == 0xff80U) {
        return 0U;
    }
    // Signed zero compares equal in the legacy path.
    if (magnitude == 0U) {
        bits = 0U;
    }

    // Monotonic unsigned encoding for non-NaN BF16 values.
    const uint16_t ordered = (bits & 0x8000U)
        ? static_cast<uint16_t>(~bits)
        : static_cast<uint16_t>(bits ^ 0x8000U);
    const uint32_t logical_lane = column & 0xffU;
    const uint32_t lane_offset = column >> 8U;

    // Larger packed key wins atomicMax. Lower logical lane wins equal-value
    // cross-lane ties; lower offset wins equal-value ties inside one lane.
    return (static_cast<uint32_t>(ordered) << 16U)
        | ((0xffU - logical_lane) << 8U)
        | (0xffU - lane_offset);
}

// Scratchless multi-CTA argmax. `output[row]` is used as the atomic accumulator
// after the Rust launcher asynchronously zeroes it. Each CTA performs a local
// reduction first, limiting global contention to 32 atomicMax operations/row.
extern "C" __global__
__launch_bounds__(SAMPLING_MAX_BLOCK_SIZE)
void argmax_rows_bf16_atomic_stage1(
    const __nv_bfloat16* __restrict__ input,
    uint32_t* __restrict__ output,
    size_t rows,
    size_t columns
) {
    const size_t row = blockIdx.x / ARGMAX_ATOMIC_BLOCKS_PER_ROW;
    const uint32_t row_block =
        static_cast<uint32_t>(blockIdx.x % ARGMAX_ATOMIC_BLOCKS_PER_ROW);
    if (row >= rows) {
        return;
    }

    const uint32_t worker = row_block * SAMPLING_MAX_BLOCK_SIZE + threadIdx.x;
    constexpr uint32_t WORKERS_PER_ROW =
        ARGMAX_ATOMIC_BLOCKS_PER_ROW * SAMPLING_MAX_BLOCK_SIZE;
    const size_t row_base = row * columns;
    uint32_t best_key = 0U;

    for (size_t column = worker; column < columns; column += WORKERS_PER_ROW) {
        const uint32_t key = argmax_atomic_key(
            input,
            row_base + column,
            static_cast<uint32_t>(column)
        );
        best_key = max(best_key, key);
    }

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    #pragma unroll
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        best_key = max(best_key, __shfl_down_sync(0xffffffffU, best_key, offset));
    }

    __shared__ uint32_t warp_keys[8];
    if (lane == 0U) {
        warp_keys[warp] = best_key;
    }
    __syncthreads();

    if (warp == 0U) {
        best_key = lane < 8U ? warp_keys[lane] : 0U;
        #pragma unroll
        for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
            best_key = max(best_key, __shfl_down_sync(0xffffffffU, best_key, offset));
        }
        if (lane == 0U) {
            atomicMax(output + row, best_key);
        }
    }
}

// Decode the packed winner in place. Zero means every candidate behaved like
// the legacy initialization (for example an all-NaN/-inf row), whose result is
// index zero.
extern "C" __global__
__launch_bounds__(SAMPLING_MAX_BLOCK_SIZE)
void argmax_rows_bf16_atomic_decode(
    uint32_t* __restrict__ output,
    size_t rows
) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const uint32_t key = output[row];
    if (key == 0U) {
        output[row] = 0U;
        return;
    }

    const uint32_t logical_lane = 0xffU - ((key >> 8U) & 0xffU);
    const uint32_t lane_offset = 0xffU - (key & 0xffU);
    output[row] = (lane_offset << 8U) | logical_lane;
}
