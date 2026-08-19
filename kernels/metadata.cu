#include <stddef.h>
#include <stdint.h>

constexpr int METADATA_BLOCK_SIZE = 256;

__device__ __forceinline__ size_t align_up_8(size_t value) {
    return (value + 7U) & ~size_t(7U);
}

extern "C" __global__
__launch_bounds__(METADATA_BLOCK_SIZE)
void scatter_batch_metadata(
    const uint8_t* __restrict__ packed,
    uint32_t* __restrict__ token_ids,
    uint32_t* __restrict__ positions,
    uint32_t* __restrict__ request_slots,
    int64_t* __restrict__ physical_slots,
    uint32_t* __restrict__ segment_offsets,
    uint32_t* __restrict__ segment_slots,
    uint32_t* __restrict__ output_rows,
    size_t num_tokens,
    size_t num_segments
) {
    const size_t token_bytes = num_tokens * sizeof(uint32_t);
    const size_t token_ids_offset = 0;
    const size_t positions_offset = token_ids_offset + token_bytes;
    const size_t request_slots_offset = positions_offset + token_bytes;
    const size_t physical_slots_offset = align_up_8(request_slots_offset + token_bytes);
    const size_t segment_offsets_offset =
        physical_slots_offset + num_tokens * sizeof(int64_t);
    const size_t segment_slots_offset =
        segment_offsets_offset + (num_segments + 1) * sizeof(uint32_t);
    const size_t output_rows_offset =
        segment_slots_offset + num_segments * sizeof(uint32_t);

    const auto* packed_token_ids =
        reinterpret_cast<const uint32_t*>(packed + token_ids_offset);
    const auto* packed_positions =
        reinterpret_cast<const uint32_t*>(packed + positions_offset);
    const auto* packed_request_slots =
        reinterpret_cast<const uint32_t*>(packed + request_slots_offset);
    const auto* packed_physical_slots =
        reinterpret_cast<const int64_t*>(packed + physical_slots_offset);
    const auto* packed_segment_offsets =
        reinterpret_cast<const uint32_t*>(packed + segment_offsets_offset);
    const auto* packed_segment_slots =
        reinterpret_cast<const uint32_t*>(packed + segment_slots_offset);
    const auto* packed_output_rows =
        reinterpret_cast<const uint32_t*>(packed + output_rows_offset);

    const size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t index = thread; index < num_tokens; index += stride) {
        token_ids[index] = packed_token_ids[index];
        positions[index] = packed_positions[index];
        request_slots[index] = packed_request_slots[index];
        physical_slots[index] = packed_physical_slots[index];
    }

    for (size_t index = thread; index < num_segments; index += stride) {
        segment_slots[index] = packed_segment_slots[index];
        output_rows[index] = packed_output_rows[index];
    }

    for (size_t index = thread; index <= num_segments; index += stride) {
        segment_offsets[index] = packed_segment_offsets[index];
    }
}
