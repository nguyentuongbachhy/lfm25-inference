#include <cuda_bf16.h>

#include <stddef.h>
#include <stdint.h>

constexpr int GATHER_MAX_BLOCK_SIZE = 256;

extern "C" __global__
__launch_bounds__(GATHER_MAX_BLOCK_SIZE)
void gather_rows_bf16(
    const __nv_bfloat16* __restrict__ input,
    const uint32_t* __restrict__ row_indices,
    __nv_bfloat16* __restrict__ output,
    size_t output_rows,
    size_t input_rows,
    size_t columns
) {
    const size_t row = blockIdx.x;
    if (row >= output_rows) {
        return;
    }
    const size_t source_row = row_indices[row];
    if (source_row >= input_rows) {
        return;
    }
    const size_t source_base = source_row * columns;
    const size_t output_base = row * columns;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        output[output_base + column] = input[source_base + column];
    }
}
