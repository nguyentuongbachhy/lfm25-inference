#include <cuda_bf16.h>

#include <float.h>
#include <stddef.h>
#include <stdint.h>

constexpr int SAMPLING_MAX_BLOCK_SIZE = 256;
constexpr int ARGMAX_LOGICAL_LANES = 256;
constexpr int ARGMAX_WORKERS_PER_LANE = 32;
constexpr int ARGMAX_LANES_PER_STAGE1_BLOCK = 8;
constexpr int ARGMAX_STAGE1_BLOCKS_PER_ROW =
    ARGMAX_LOGICAL_LANES / ARGMAX_LANES_PER_STAGE1_BLOCK;

__device__ __forceinline__ bool argmax_better(
    float candidate_value,
    uint32_t candidate_priority,
    float best_value,
    uint32_t best_priority
) {
    return candidate_value > best_value
        || (candidate_value == best_value && candidate_priority < best_priority);
}

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

// Stage 1 preserves the exact logical work partition of argmax_rows_bf16:
// logical lane L owns columns L, L+256, L+512, ... . Instead of one physical
// thread scanning the entire residue class, one warp cooperatively scans it.
// Eight warps therefore produce eight logical-lane winners per CTA and 32 CTAs
// cover one row. The tie rule inside each logical lane is the earliest column,
// matching the original thread's strictly-increasing scan order.
extern "C" __global__
__launch_bounds__(SAMPLING_MAX_BLOCK_SIZE)
void argmax_rows_bf16_stage1(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ partial_values,
    uint32_t* __restrict__ partial_indices,
    size_t rows,
    size_t columns
) {
    const size_t row = blockIdx.x / ARGMAX_STAGE1_BLOCKS_PER_ROW;
    const uint32_t lane_block =
        static_cast<uint32_t>(blockIdx.x % ARGMAX_STAGE1_BLOCKS_PER_ROW);
    if (row >= rows) {
        return;
    }

    const uint32_t worker = threadIdx.x & 31U;
    const uint32_t local_lane = threadIdx.x >> 5U;
    const uint32_t logical_lane =
        lane_block * ARGMAX_LANES_PER_STAGE1_BLOCK + local_lane;
    const size_t row_base = row * columns;

    float best_value = -FLT_MAX;
    uint32_t best_column = logical_lane;
    for (
        size_t column = static_cast<size_t>(logical_lane)
            + static_cast<size_t>(ARGMAX_LOGICAL_LANES) * worker;
        column < columns;
        column += static_cast<size_t>(ARGMAX_LOGICAL_LANES) * ARGMAX_WORKERS_PER_LANE
    ) {
        const float value = __bfloat162float(input[row_base + column]);
        const uint32_t column_u32 = static_cast<uint32_t>(column);
        if (argmax_better(value, column_u32, best_value, best_column)) {
            best_value = value;
            best_column = column_u32;
        }
    }

    #pragma unroll
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
        const uint32_t other_column =
            __shfl_down_sync(0xffffffffU, best_column, offset);
        if (worker + offset < 32U
            && argmax_better(other_value, other_column, best_value, best_column)) {
            best_value = other_value;
            best_column = other_column;
        }
    }

    if (worker == 0U) {
        const size_t partial = row * ARGMAX_LOGICAL_LANES + logical_lane;
        partial_values[partial] = best_value;
        partial_indices[partial] = best_column;
    }
}

// Stage 2 reproduces the original CTA's final serial tie semantics. The old
// kernel considers logical threads in ascending threadIdx order and only
// replaces the winner for a strictly larger value, so equal maxima prefer the
// smaller logical-lane id even when its absolute token index is larger.
extern "C" __global__
__launch_bounds__(SAMPLING_MAX_BLOCK_SIZE)
void argmax_rows_bf16_stage2(
    const float* __restrict__ partial_values,
    const uint32_t* __restrict__ partial_indices,
    uint32_t* __restrict__ output,
    size_t rows
) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t logical_lane = threadIdx.x;
    const size_t partial = row * ARGMAX_LOGICAL_LANES + logical_lane;

    float best_value = partial_values[partial];
    uint32_t best_priority = logical_lane;
    uint32_t best_index = partial_indices[partial];

    #pragma unroll
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
        const uint32_t other_priority =
            __shfl_down_sync(0xffffffffU, best_priority, offset);
        const uint32_t other_index =
            __shfl_down_sync(0xffffffffU, best_index, offset);
        if (lane + offset < 32U
            && argmax_better(other_value, other_priority, best_value, best_priority)) {
            best_value = other_value;
            best_priority = other_priority;
            best_index = other_index;
        }
    }

    __shared__ float warp_values[8];
    __shared__ uint32_t warp_priorities[8];
    __shared__ uint32_t warp_indices[8];
    if (lane == 0U) {
        warp_values[warp] = best_value;
        warp_priorities[warp] = best_priority;
        warp_indices[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0U) {
        if (lane < 8U) {
            best_value = warp_values[lane];
            best_priority = warp_priorities[lane];
            best_index = warp_indices[lane];
        } else {
            best_value = -FLT_MAX;
            best_priority = UINT32_MAX;
            best_index = 0U;
        }

        #pragma unroll
        for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
            const float other_value =
                __shfl_down_sync(0xffffffffU, best_value, offset);
            const uint32_t other_priority =
                __shfl_down_sync(0xffffffffU, best_priority, offset);
            const uint32_t other_index =
                __shfl_down_sync(0xffffffffU, best_index, offset);
            if (lane + offset < 32U
                && argmax_better(other_value, other_priority, best_value, best_priority)) {
                best_value = other_value;
                best_priority = other_priority;
                best_index = other_index;
            }
        }
        if (lane == 0U) {
            output[row] = best_index;
        }
    }
}
