#include <cuda_bf16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int ATTENTION_ASYNC_BLOCK_SIZE = 256;
constexpr uint32_t LFM2_NUM_Q_HEADS = 32U;
constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_Q_PER_KV = 4U;
constexpr float LFM2_ATTN_SCALE = 0.125f;

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem), "l"(src)
    );
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

template <int PAGE_SIZE>
__device__ __forceinline__ size_t cache_index(
    size_t page,
    size_t kv_head,
    size_t offset,
    size_t dim
) {
    return (((page * LFM2_NUM_KV_HEADS + kv_head) * PAGE_SIZE + offset)
        * LFM2_HEAD_DIM + dim);
}

template <int PAGE_SIZE>
__device__ __forceinline__ void stage_page_async(
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    size_t physical_page,
    uint32_t kv_head,
    __nv_bfloat16* __restrict__ key_stage,
    __nv_bfloat16* __restrict__ value_stage
) {
    constexpr size_t ELEMENTS = PAGE_SIZE * LFM2_HEAD_DIM;
    constexpr size_t ELEMENTS_PER_COPY = 16 / sizeof(__nv_bfloat16);
    constexpr size_t COPIES = ELEMENTS / ELEMENTS_PER_COPY;
    const size_t cache_base = cache_index<PAGE_SIZE>(physical_page, kv_head, 0, 0);

    for (size_t copy = threadIdx.x; copy < COPIES; copy += blockDim.x) {
        const size_t element = copy * ELEMENTS_PER_COPY;
        cp_async_16(key_stage + element, key_cache + cache_base + element);
        cp_async_16(value_stage + element, value_cache + cache_base + element);
    }
    cp_async_commit();
}

template <int PAGE_SIZE, bool RAGGED>
__device__ __forceinline__ void paged_gqa_lfm2_bf16_async_body(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t block_table_rows
) {
    const size_t token = blockIdx.x / LFM2_NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % LFM2_NUM_KV_HEADS;
    if (token >= num_tokens || block_table_stride == 0) {
        return;
    }

    size_t table_row = 0;
    if (RAGGED) {
        table_row = static_cast<size_t>(request_slots[token]);
        if (table_row >= block_table_rows) {
            return;
        }
    }
    const uint32_t* __restrict__ block_table =
        block_tables + table_row * block_table_stride;

    const uint32_t position = position_ids[token];
    if (static_cast<size_t>(position) >= block_table_stride * PAGE_SIZE) {
        return;
    }

    const size_t last_page = static_cast<size_t>(position) / PAGE_SIZE;
    const size_t first_physical_page = static_cast<size_t>(block_table[0]);
    if (first_physical_page >= num_pages) {
        return;
    }

    __shared__ __align__(16) __nv_bfloat16 key_stage[2][PAGE_SIZE * LFM2_HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 value_stage[2][PAGE_SIZE * LFM2_HEAD_DIM];

    stage_page_async<PAGE_SIZE>(
        key_cache,
        value_cache,
        first_physical_page,
        kv_head,
        key_stage[0],
        value_stage[0]
    );
    cp_async_wait_all();
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const bool active = warp < LFM2_Q_PER_KV;
    const uint32_t q_head = kv_head * LFM2_Q_PER_KV + warp;
    const size_t query_base =
        (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;
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
            stage_page_async<PAGE_SIZE>(
                key_cache,
                value_cache,
                next_physical_page,
                kv_head,
                key_stage[stage ^ 1U],
                value_stage[stage ^ 1U]
            );
        }

        if (active) {
            const size_t page_start = page * PAGE_SIZE;
            const size_t remaining = static_cast<size_t>(position) + 1 - page_start;
            const size_t page_tokens = remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

            for (size_t key_offset = 0; key_offset < page_tokens; ++key_offset) {
                const size_t key_base = key_offset * LFM2_HEAD_DIM;
                const size_t index0 = key_base + dim0;
                const size_t index1 = key_base + dim1;

                float dot =
                    q0 * __bfloat162float(key_stage[stage][index0])
                    + q1 * __bfloat162float(key_stage[stage][index1]);

                #pragma unroll
                for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                    dot += __shfl_down_sync(0xffffffffU, dot, delta);
                }

                dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;
                const float next_maximum = fmaxf(maximum, dot);
                const float old_scale = expf(maximum - next_maximum);
                const float new_scale = expf(dot - next_maximum);
                const float value0 = __bfloat162float(value_stage[stage][index0]);
                const float value1 = __bfloat162float(value_stage[stage][index1]);

                accumulator0 = accumulator0 * old_scale + value0 * new_scale;
                accumulator1 = accumulator1 * old_scale + value1 * new_scale;
                denominator = denominator * old_scale + new_scale;
                maximum = next_maximum;
            }
        }

        if (has_next) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    if (active) {
        output[query_base + dim0] = __float2bfloat16_rn(accumulator0 / denominator);
        output[query_base + dim1] = __float2bfloat16_rn(accumulator1 / denominator);
    }
}

extern "C" __global__
__launch_bounds__(ATTENTION_ASYNC_BLOCK_SIZE)
void paged_gqa_lfm2_bf16_async_ps16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length
) {
    paged_gqa_lfm2_bf16_async_body<16, false>(
        query,
        key_cache,
        value_cache,
        block_table,
        nullptr,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_length,
        1
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_ASYNC_BLOCK_SIZE)
void paged_gqa_lfm2_bf16_async_ps32(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length
) {
    paged_gqa_lfm2_bf16_async_body<32, false>(
        query,
        key_cache,
        value_cache,
        block_table,
        nullptr,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_length,
        1
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_ASYNC_BLOCK_SIZE)
void paged_ragged_gqa_lfm2_bf16_async_ps16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t block_table_rows
) {
    paged_gqa_lfm2_bf16_async_body<16, true>(
        query,
        key_cache,
        value_cache,
        block_tables,
        request_slots,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_stride,
        block_table_rows
    );
}

extern "C" __global__
__launch_bounds__(ATTENTION_ASYNC_BLOCK_SIZE)
void paged_ragged_gqa_lfm2_bf16_async_ps32(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t block_table_rows
) {
    paged_gqa_lfm2_bf16_async_body<32, true>(
        query,
        key_cache,
        value_cache,
        block_tables,
        request_slots,
        position_ids,
        output,
        num_tokens,
        num_pages,
        block_table_stride,
        block_table_rows
    );
}
