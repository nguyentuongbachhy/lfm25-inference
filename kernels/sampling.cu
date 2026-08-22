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
constexpr int ARGMAX_ATOMIC_BLOCKS_PER_ROW = 32;

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

// Encode one BF16 logit and the legacy argmax tie priority into a sortable
// 32-bit key. This is possible because production vocab <= 65536: 16 bits hold
// the ordered BF16 value, 8 bits prefer the smaller logical lane (column % 256),
// and 8 bits prefer the earlier element within that lane (column / 256).
//
// The original kernel starts each logical lane at -FLT_MAX and compares with
// strict `>`, so NaNs and -inf never become lane winners. Signed zero compares
// equal in float, therefore both +0 and -0 are normalized to one key.
__device__ __forceinline__ uint32_t argmax_atomic_key(
    const __nv_bfloat16* __restrict__ input,
    size_t index,
    uint32_t column
) {
    uint16_t bits = reinterpret_cast<const uint16_t*>(input)[index];
    const uint16_t magnitude = bits & 0x7fffU;
    if ((magnitude & 0x7f80U) == 0x7f80U && (magnitude & 0x007fU) != 0U) {
        return 0U;
    }
    if (bits == 0xff80U) {
        return 0U;
    }
    if (magnitude == 0U) {
        bits = 0U;
    }

    const uint16_t ordered = (bits & 0x8000U)
        ? static_cast<uint16_t>(~bits)
        : static_cast<uint16_t>(bits ^ 0x8000U);
    const uint32_t logical_lane = column & 0xffU;
    const uint32_t lane_offset = column >> 8U;
    return (static_cast<uint32_t>(ordered) << 16U)
        | ((0xffU - logical_lane) << 8U)
        | (0xffU - lane_offset);
}

// Scratchless multi-CTA argmax. The output row itself is a temporary atomicMax
// accumulator. The Rust launcher zeroes it asynchronously before this kernel.
// Each CTA reduces locally first, limiting global contention to 32 atomics/row.
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

// Convert the packed winner key back to the token column in place. A zero key
// means every candidate behaved like the legacy initial -FLT_MAX state (e.g.
// all NaN/-inf), for which the original kernel returns index 0.
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
