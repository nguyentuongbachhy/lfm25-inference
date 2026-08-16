#include <cuda_bf16.h>

#include <stddef.h>
#include <stdint.h>

constexpr int SHORT_CONV_MAX_BLOCK_SIZE = 256;

extern "C" __global__
__launch_bounds__(SHORT_CONV_MAX_BLOCK_SIZE)
void short_conv_lfm2_bf16(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t hidden_size
) {
    const size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t channel = thread; channel < hidden_size; channel += stride) {
        float state0 = __bfloat162float(state[channel * 2]);
        float state1 = __bfloat162float(state[channel * 2 + 1]);

        const float weight0 = __bfloat162float(weight[channel * 3]);
        const float weight1 = __bfloat162float(weight[channel * 3 + 1]);
        const float weight2 = __bfloat162float(weight[channel * 3 + 2]);

        for (size_t token = 0; token < num_tokens; ++token) {
            const size_t base = token * hidden_size * 3;
            const float b = __bfloat162float(projected[base + channel]);
            const float c = __bfloat162float(projected[base + hidden_size + channel]);
            const float x = __bfloat162float(projected[base + hidden_size * 2 + channel]);
            const float gated = b * x;
            const float convolved =
                weight0 * state0 + weight1 * state1 + weight2 * gated;

            output[token * hidden_size + channel] =
                __float2bfloat16_rn(c * convolved);

            state0 = state1;
            state1 = gated;
        }

        state[channel * 2] = __float2bfloat16_rn(state0);
        state[channel * 2 + 1] = __float2bfloat16_rn(state1);
    }
}

extern "C" __global__
__launch_bounds__(SHORT_CONV_MAX_BLOCK_SIZE)
void short_conv_ragged_lfm2_bf16(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ states,
    const uint32_t* __restrict__ request_slots,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t hidden_size,
    size_t num_request_slots
) {
    const size_t work_items = num_tokens * hidden_size;
    const size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t work = thread; work < work_items; work += stride) {
        const size_t token = work / hidden_size;
        const size_t channel = work % hidden_size;
        const size_t request_slot = request_slots[token];
        if (request_slot >= num_request_slots) {
            continue;
        }
        const size_t state_base =
            (request_slot * hidden_size + channel) * 2;
        const size_t projected_base = token * hidden_size * 3;
        const float state0 = __bfloat162float(states[state_base]);
        const float state1 = __bfloat162float(states[state_base + 1]);
        const float weight0 = __bfloat162float(weight[channel * 3]);
        const float weight1 = __bfloat162float(weight[channel * 3 + 1]);
        const float weight2 = __bfloat162float(weight[channel * 3 + 2]);
        const float b = __bfloat162float(projected[projected_base + channel]);
        const float c = __bfloat162float(
            projected[projected_base + hidden_size + channel]
        );
        const float x = __bfloat162float(
            projected[projected_base + hidden_size * 2 + channel]
        );
        const float gated = b * x;
        const float convolved =
            weight0 * state0 + weight1 * state1 + weight2 * gated;
        output[token * hidden_size + channel] =
            __float2bfloat16_rn(c * convolved);
        states[state_base] = __float2bfloat16_rn(state1);
        states[state_base + 1] = __float2bfloat16_rn(gated);
    }
}

extern "C" __global__
__launch_bounds__(SHORT_CONV_MAX_BLOCK_SIZE)
void short_conv_segmented_lfm2_bf16(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ states,
    const uint32_t* __restrict__ segment_offsets,
    const uint32_t* __restrict__ segment_slots,
    __nv_bfloat16* __restrict__ output,
    size_t num_segments,
    size_t hidden_size,
    size_t num_request_slots
) {
    const size_t work_items = num_segments * hidden_size;
    const size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t work = thread; work < work_items; work += stride) {
        const size_t segment = work / hidden_size;
        const size_t channel = work % hidden_size;
        const size_t request_slot = segment_slots[segment];
        if (request_slot >= num_request_slots) {
            continue;
        }
        const size_t state_base = (request_slot * hidden_size + channel) * 2;
        float state0 = __bfloat162float(states[state_base]);
        float state1 = __bfloat162float(states[state_base + 1]);
        const float weight0 = __bfloat162float(weight[channel * 3]);
        const float weight1 = __bfloat162float(weight[channel * 3 + 1]);
        const float weight2 = __bfloat162float(weight[channel * 3 + 2]);
        const size_t token_start = segment_offsets[segment];
        const size_t token_end = segment_offsets[segment + 1];
        for (size_t token = token_start; token < token_end; ++token) {
            const size_t base = token * hidden_size * 3;
            const float b = __bfloat162float(projected[base + channel]);
            const float c = __bfloat162float(projected[base + hidden_size + channel]);
            const float x = __bfloat162float(projected[base + hidden_size * 2 + channel]);
            const float gated = b * x;
            const float convolved = weight0 * state0 + weight1 * state1 + weight2 * gated;
            output[token * hidden_size + channel] = __float2bfloat16_rn(c * convolved);
            state0 = state1;
            state1 = gated;
        }
        states[state_base] = __float2bfloat16_rn(state0);
        states[state_base + 1] = __float2bfloat16_rn(state1);
    }
}
