#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int BLOCK_SIZE = 256;
constexpr uint32_t NUM_Q_HEADS = 32U;
constexpr uint32_t NUM_KV_HEADS = 8U;
constexpr uint32_t HEAD_DIM = 64U;
constexpr uint32_t Q_PER_KV = 4U;
constexpr int PAGE_SIZE = 16;
constexpr size_t PAGE_HEAD_ELEMENTS = PAGE_SIZE * HEAD_DIM;
constexpr size_t PAGE_HEAD_PAIRS = PAGE_HEAD_ELEMENTS / 2U;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float ATTN_SCALE = 0.125f;

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(src));
#else
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

extern "C" __global__
__launch_bounds__(BLOCK_SIZE)
void quantize_paged_kv_lfm2_e4m3_ps16(
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    unsigned char* __restrict__ key_fp8,
    unsigned char* __restrict__ value_fp8,
    float* __restrict__ key_scales,
    float* __restrict__ value_scales,
    size_t num_pages
) {
    const size_t page_head = blockIdx.x;
    const size_t total_page_heads = num_pages * NUM_KV_HEADS;
    if (page_head >= total_page_heads) {
        return;
    }
    const size_t base = page_head * PAGE_HEAD_ELEMENTS;
    float local_key_max = 0.0f;
    float local_value_max = 0.0f;
    for (size_t i = threadIdx.x; i < PAGE_HEAD_ELEMENTS; i += blockDim.x) {
        local_key_max = fmaxf(local_key_max, fabsf(__bfloat162float(key_cache[base + i])));
        local_value_max = fmaxf(local_value_max, fabsf(__bfloat162float(value_cache[base + i])));
    }
    __shared__ float key_reduce[BLOCK_SIZE];
    __shared__ float value_reduce[BLOCK_SIZE];
    key_reduce[threadIdx.x] = local_key_max;
    value_reduce[threadIdx.x] = local_value_max;
    __syncthreads();
    for (uint32_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            key_reduce[threadIdx.x] = fmaxf(key_reduce[threadIdx.x], key_reduce[threadIdx.x + stride]);
            value_reduce[threadIdx.x] = fmaxf(value_reduce[threadIdx.x], value_reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float key_scale = key_reduce[0] > 0.0f ? key_reduce[0] / FP8_E4M3_MAX : 1.0f;
    const float value_scale = value_reduce[0] > 0.0f ? value_reduce[0] / FP8_E4M3_MAX : 1.0f;
    if (threadIdx.x == 0) {
        key_scales[page_head] = key_scale;
        value_scales[page_head] = value_scale;
    }
    const float key_inv_scale = 1.0f / key_scale;
    const float value_inv_scale = 1.0f / value_scale;
    for (size_t i = threadIdx.x; i < PAGE_HEAD_ELEMENTS; i += blockDim.x) {
        key_fp8[base + i] = __nv_cvt_float_to_fp8(
            __bfloat162float(key_cache[base + i]) * key_inv_scale,
            __NV_SATFINITE,
            __NV_E4M3
        );
        value_fp8[base + i] = __nv_cvt_float_to_fp8(
            __bfloat162float(value_cache[base + i]) * value_inv_scale,
            __NV_SATFINITE,
            __NV_E4M3
        );
    }
}

__device__ __forceinline__ void stage_page_async(
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    size_t physical_page,
    uint32_t kv_head,
    unsigned char* __restrict__ key_stage,
    unsigned char* __restrict__ value_stage
) {
    constexpr size_t COPIES = PAGE_HEAD_ELEMENTS / 16;
    const size_t base = (physical_page * NUM_KV_HEADS + kv_head) * PAGE_HEAD_ELEMENTS;
    for (size_t copy = threadIdx.x; copy < COPIES; copy += blockDim.x) {
        const size_t offset = copy * 16;
        cp_async_16(key_stage + offset, key_cache + base + offset);
        cp_async_16(value_stage + offset, value_cache + base + offset);
    }
    cp_async_commit();
}

__device__ __forceinline__ void decode_staged_page_fp8x2(
    const unsigned char* __restrict__ key_fp8_stage,
    const unsigned char* __restrict__ value_fp8_stage,
    __half* __restrict__ key_decoded_stage,
    __half* __restrict__ value_decoded_stage
) {
    for (size_t pair = threadIdx.x; pair < PAGE_HEAD_PAIRS; pair += blockDim.x) {
        const size_t offset = pair * 2U;
        const auto key_packed = *reinterpret_cast<const __nv_fp8x2_storage_t*>(key_fp8_stage + offset);
        const auto value_packed = *reinterpret_cast<const __nv_fp8x2_storage_t*>(value_fp8_stage + offset);
        const __half2 key_half2(__nv_cvt_fp8x2_to_halfraw2(key_packed, __NV_E4M3));
        const __half2 value_half2(__nv_cvt_fp8x2_to_halfraw2(value_packed, __NV_E4M3));
        key_decoded_stage[offset] = key_half2.x;
        key_decoded_stage[offset + 1U] = key_half2.y;
        value_decoded_stage[offset] = value_half2.x;
        value_decoded_stage[offset + 1U] = value_half2.y;
    }
}

extern "C" __global__
__launch_bounds__(BLOCK_SIZE)
void paged_gqa_lfm2_fp8_kv_ps16(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const float* __restrict__ key_scales,
    const float* __restrict__ value_scales,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length
) {
    const size_t token = blockIdx.x / NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % NUM_KV_HEADS;
    if (token >= num_tokens || block_table_length == 0) {
        return;
    }
    const uint32_t position = position_ids[token];
    const size_t last_page = static_cast<size_t>(position) / PAGE_SIZE;
    if (last_page >= block_table_length) {
        return;
    }
    const size_t first_physical_page = static_cast<size_t>(block_table[0]);
    if (first_physical_page >= num_pages) {
        return;
    }

    __shared__ __align__(16) unsigned char key_fp8_stage[2][PAGE_HEAD_ELEMENTS];
    __shared__ __align__(16) unsigned char value_fp8_stage[2][PAGE_HEAD_ELEMENTS];
    __shared__ __align__(16) __half key_decoded_stage[2][PAGE_HEAD_ELEMENTS];
    __shared__ __align__(16) __half value_decoded_stage[2][PAGE_HEAD_ELEMENTS];
    __shared__ float key_scale_stage[2];
    __shared__ float value_scale_stage[2];

    if (threadIdx.x == 0) {
        const size_t scale_index = first_physical_page * NUM_KV_HEADS + kv_head;
        key_scale_stage[0] = key_scales[scale_index];
        value_scale_stage[0] = value_scales[scale_index];
    }
    stage_page_async(
        key_cache,
        value_cache,
        first_physical_page,
        kv_head,
        key_fp8_stage[0],
        value_fp8_stage[0]
    );
    cp_async_wait_all();
    __syncthreads();
    decode_staged_page_fp8x2(
        key_fp8_stage[0],
        value_fp8_stage[0],
        key_decoded_stage[0],
        value_decoded_stage[0]
    );
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const bool active = warp < Q_PER_KV;
    const uint32_t q_head = kv_head * Q_PER_KV + warp;
    const size_t query_base = (token * NUM_Q_HEADS + q_head) * HEAD_DIM;
    const size_t dim0 = lane;
    const size_t dim1 = lane + 32U;
    const float q0 = active ? __bfloat162float(query[query_base + dim0]) : 0.0f;
    const float q1 = active ? __bfloat162float(query[query_base + dim1]) : 0.0f;

    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    for (size_t page = 0; page <= last_page; ++page) {
        const uint32_t stage = static_cast<uint32_t>(page & 1U);
        const bool has_next = page < last_page;
        if (has_next) {
            const size_t next_physical_page = static_cast<size_t>(block_table[page + 1]);
            if (next_physical_page >= num_pages) {
                return;
            }
            if (threadIdx.x == 0) {
                const size_t scale_index = next_physical_page * NUM_KV_HEADS + kv_head;
                key_scale_stage[stage ^ 1U] = key_scales[scale_index];
                value_scale_stage[stage ^ 1U] = value_scales[scale_index];
            }
            stage_page_async(
                key_cache,
                value_cache,
                next_physical_page,
                kv_head,
                key_fp8_stage[stage ^ 1U],
                value_fp8_stage[stage ^ 1U]
            );
        }

        if (active) {
            const size_t page_start = page * PAGE_SIZE;
            const size_t remaining = static_cast<size_t>(position) + 1U - page_start;
            const size_t page_tokens = remaining < PAGE_SIZE ? remaining : PAGE_SIZE;
            const float key_scale = key_scale_stage[stage];
            const float value_scale = value_scale_stage[stage];
            for (size_t key_offset = 0; key_offset < page_tokens; ++key_offset) {
                const size_t key_base = key_offset * HEAD_DIM;
                const size_t index0 = key_base + dim0;
                const size_t index1 = key_base + dim1;
                float dot = q0 * __half2float(key_decoded_stage[stage][index0]) * key_scale
                    + q1 * __half2float(key_decoded_stage[stage][index1]) * key_scale;
                #pragma unroll
                for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                    dot += __shfl_down_sync(0xffffffffU, dot, delta);
                }
                dot = __shfl_sync(0xffffffffU, dot, 0) * ATTN_SCALE;
                const float next_maximum = fmaxf(maximum, dot);
                const float old_scale = __expf(maximum - next_maximum);
                const float new_scale = __expf(dot - next_maximum);
                const float value0 = __half2float(value_decoded_stage[stage][index0]) * value_scale;
                const float value1 = __half2float(value_decoded_stage[stage][index1]) * value_scale;
                accumulator0 = accumulator0 * old_scale + value0 * new_scale;
                accumulator1 = accumulator1 * old_scale + value1 * new_scale;
                denominator = denominator * old_scale + new_scale;
                maximum = next_maximum;
            }
        }
        if (has_next) {
            cp_async_wait_all();
            __syncthreads();
            decode_staged_page_fp8x2(
                key_fp8_stage[stage ^ 1U],
                value_fp8_stage[stage ^ 1U],
                key_decoded_stage[stage ^ 1U],
                value_decoded_stage[stage ^ 1U]
            );
            __syncthreads();
        }
    }

    if (active) {
        output[query_base + dim0] = __float2bfloat16_rn(accumulator0 / denominator);
        output[query_base + dim1] = __float2bfloat16_rn(accumulator1 / denominator);
    }
}
