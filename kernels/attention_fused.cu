#include <cuda_bf16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

constexpr int FUSED_ATTENTION_BLOCK_SIZE = 256;
constexpr uint32_t LFM2_NUM_Q_HEADS = 32U;
constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_HALF_DIM = 32U;
constexpr uint32_t LFM2_Q_PER_KV = 4U;
constexpr float LFM2_ATTN_SCALE = 0.125f;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
        value += __shfl_down_sync(0xffffffffU, value, delta);
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float rms_weight_bf16(
    float value,
    float inv_rms,
    __nv_bfloat16 weight
) {
    const __nv_bfloat16 normalized = __float2bfloat16_rn(value * inv_rms);
    return __bfloat162float(__hmul(normalized, weight));
}

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
    uint32_t kv_head,
    size_t offset,
    uint32_t dim
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
__device__ __forceinline__ void fused_decode_attention_body(
    const __nv_bfloat16* __restrict__ query_raw,
    const __nv_bfloat16* __restrict__ key_raw,
    const __nv_bfloat16* __restrict__ value_raw,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t block_table_rows,
    float eps
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
    const size_t logical_page_offset = static_cast<size_t>(position) % PAGE_SIZE;
    const size_t first_physical_page = static_cast<size_t>(block_table[0]);
    const size_t current_physical_page = static_cast<size_t>(block_table[last_page]);
    if (first_physical_page >= num_pages || current_physical_page >= num_pages) {
        return;
    }

    const int64_t slot_value = slot_mapping[token];
    if (slot_value < 0) {
        return;
    }
    const size_t physical_slot = static_cast<size_t>(slot_value);
    const size_t slot_page = physical_slot / PAGE_SIZE;
    const size_t slot_offset = physical_slot % PAGE_SIZE;
    if (slot_page != current_physical_page || slot_offset != logical_page_offset) {
        return;
    }

    __shared__ __align__(16) __nv_bfloat16 key_stage[2][PAGE_SIZE * LFM2_HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 value_stage[2][PAGE_SIZE * LFM2_HEAD_DIM];
    __shared__ float sin_cache[LFM2_HALF_DIM];
    __shared__ float cos_cache[LFM2_HALF_DIM];

    // Start the first page transfer before Q/K postprocess. For a one-page
    // context the current cache slot may still contain stale data; it is patched
    // in shared memory after the async copy completes.
    stage_page_async<PAGE_SIZE>(
        key_cache,
        value_cache,
        first_physical_page,
        kv_head,
        key_stage[0],
        value_stage[0]
    );

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    if (warp == 0U) {
        sincosf(
            static_cast<float>(position) * inv_freq[lane],
            &sin_cache[lane],
            &cos_cache[lane]
        );
    }
    __syncthreads();

    const float sin_value = sin_cache[lane];
    const float cos_value = cos_cache[lane];
    const bool active = warp < LFM2_Q_PER_KV;
    const uint32_t q_head = kv_head * LFM2_Q_PER_KV + warp;

    float q0 = 0.0f;
    float q1 = 0.0f;
    if (active) {
        const size_t query_base =
            (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;
        const size_t index0 = query_base + lane;
        const size_t index1 = index0 + LFM2_HALF_DIM;
        const float raw0 = __bfloat162float(query_raw[index0]);
        const float raw1 = __bfloat162float(query_raw[index1]);
        const float sum = warp_sum(fmaf(raw0, raw0, raw1 * raw1));
        const float inv_rms = __frsqrt_rn(sum * (1.0f / 64.0f) + eps);
        const float x0 = rms_weight_bf16(raw0, inv_rms, query_norm[lane]);
        const float x1 = rms_weight_bf16(
            raw1,
            inv_rms,
            query_norm[lane + LFM2_HALF_DIM]
        );
        // Preserve the exact global-Q BF16 rounding boundary of the two-kernel
        // path while keeping the rounded values in registers for attention.
        const __nv_bfloat16 rounded0 =
            __float2bfloat16_rn(fmaf(-x1, sin_value, x0 * cos_value));
        const __nv_bfloat16 rounded1 =
            __float2bfloat16_rn(fmaf(x0, sin_value, x1 * cos_value));
        q0 = __bfloat162float(rounded0);
        q1 = __bfloat162float(rounded1);
    }

    __nv_bfloat16 current_key0 = __float2bfloat16_rn(0.0f);
    __nv_bfloat16 current_key1 = __float2bfloat16_rn(0.0f);
    if (warp == 4U) {
        const size_t base =
            (token * LFM2_NUM_KV_HEADS + kv_head) * LFM2_HEAD_DIM;
        const size_t index0 = base + lane;
        const size_t index1 = index0 + LFM2_HALF_DIM;
        const float raw0 = __bfloat162float(key_raw[index0]);
        const float raw1 = __bfloat162float(key_raw[index1]);
        const float sum = warp_sum(fmaf(raw0, raw0, raw1 * raw1));
        const float inv_rms = __frsqrt_rn(sum * (1.0f / 64.0f) + eps);
        const float x0 = rms_weight_bf16(raw0, inv_rms, key_norm[lane]);
        const float x1 = rms_weight_bf16(
            raw1,
            inv_rms,
            key_norm[lane + LFM2_HALF_DIM]
        );
        current_key0 = __float2bfloat16_rn(fmaf(-x1, sin_value, x0 * cos_value));
        current_key1 = __float2bfloat16_rn(fmaf(x0, sin_value, x1 * cos_value));
        key_cache[cache_index<PAGE_SIZE>(slot_page, kv_head, slot_offset, lane)] =
            current_key0;
        key_cache[cache_index<PAGE_SIZE>(
            slot_page,
            kv_head,
            slot_offset,
            lane + LFM2_HALF_DIM
        )] = current_key1;
    }

    __nv_bfloat16 current_value0 = __float2bfloat16_rn(0.0f);
    __nv_bfloat16 current_value1 = __float2bfloat16_rn(0.0f);
    if (warp == 5U) {
        const size_t base =
            (token * LFM2_NUM_KV_HEADS + kv_head) * LFM2_HEAD_DIM;
        const size_t index0 = base + lane;
        const size_t index1 = index0 + LFM2_HALF_DIM;
        current_value0 = value_raw[index0];
        current_value1 = value_raw[index1];
        value_cache[cache_index<PAGE_SIZE>(slot_page, kv_head, slot_offset, lane)] =
            current_value0;
        value_cache[cache_index<PAGE_SIZE>(
            slot_page,
            kv_head,
            slot_offset,
            lane + LFM2_HALF_DIM
        )] = current_value1;
    }

    cp_async_wait_all();
    __syncthreads();

    if (last_page == 0) {
        const size_t stage_base = logical_page_offset * LFM2_HEAD_DIM;
        if (warp == 4U) {
            key_stage[0][stage_base + lane] = current_key0;
            key_stage[0][stage_base + lane + LFM2_HALF_DIM] = current_key1;
        } else if (warp == 5U) {
            value_stage[0][stage_base + lane] = current_value0;
            value_stage[0][stage_base + lane + LFM2_HALF_DIM] = current_value1;
        }
        __syncthreads();
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    for (size_t page = 0; page <= last_page; ++page) {
        const uint32_t stage = static_cast<uint32_t>(page & 1U);
        const bool has_next = page < last_page;

        if (has_next) {
            const size_t next_logical_page = page + 1;
            const size_t next_physical_page = next_logical_page == last_page
                ? current_physical_page
                : static_cast<size_t>(block_table[next_logical_page]);
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
                const size_t index0 = key_base + lane;
                const size_t index1 = index0 + LFM2_HALF_DIM;

                float dot =
                    q0 * __bfloat162float(key_stage[stage][index0])
                    + q1 * __bfloat162float(key_stage[stage][index1]);
                #pragma unroll
                for (uint32_t delta = 16U; delta > 0U; delta >>= 1U) {
                    dot += __shfl_down_sync(0xffffffffU, dot, delta);
                }
                dot = __shfl_sync(0xffffffffU, dot, 0) * LFM2_ATTN_SCALE;

                const float value0 = __bfloat162float(value_stage[stage][index0]);
                const float value1 = __bfloat162float(value_stage[stage][index1]);
                if (dot > maximum) {
                    const float old_scale = __expf(maximum - dot);
                    accumulator0 = accumulator0 * old_scale + value0;
                    accumulator1 = accumulator1 * old_scale + value1;
                    denominator = denominator * old_scale + 1.0f;
                    maximum = dot;
                } else {
                    const float new_scale = __expf(dot - maximum);
                    accumulator0 = fmaf(value0, new_scale, accumulator0);
                    accumulator1 = fmaf(value1, new_scale, accumulator1);
                    denominator += new_scale;
                }
            }
        }

        if (has_next) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    if (active) {
        const size_t output_base =
            (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM;
        output[output_base + lane] = __float2bfloat16_rn(accumulator0 / denominator);
        output[output_base + lane + LFM2_HALF_DIM] =
            __float2bfloat16_rn(accumulator1 / denominator);
    }
}

extern "C" __global__
__launch_bounds__(FUSED_ATTENTION_BLOCK_SIZE)
void fused_decode_attention_lfm2_bf16_ps16(
    const __nv_bfloat16* __restrict__ query_raw,
    const __nv_bfloat16* __restrict__ key_raw,
    const __nv_bfloat16* __restrict__ value_raw,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length,
    float eps
) {
    fused_decode_attention_body<16, false>(
        query_raw, key_raw, value_raw, query_norm, key_norm, inv_freq,
        key_cache, value_cache, block_table, nullptr, position_ids, slot_mapping,
        output, num_tokens, num_pages, block_table_length, 1, eps
    );
}

extern "C" __global__
__launch_bounds__(FUSED_ATTENTION_BLOCK_SIZE)
void fused_decode_attention_lfm2_bf16_ps32(
    const __nv_bfloat16* __restrict__ query_raw,
    const __nv_bfloat16* __restrict__ key_raw,
    const __nv_bfloat16* __restrict__ value_raw,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_length,
    float eps
) {
    fused_decode_attention_body<32, false>(
        query_raw, key_raw, value_raw, query_norm, key_norm, inv_freq,
        key_cache, value_cache, block_table, nullptr, position_ids, slot_mapping,
        output, num_tokens, num_pages, block_table_length, 1, eps
    );
}

extern "C" __global__
__launch_bounds__(FUSED_ATTENTION_BLOCK_SIZE)
void fused_ragged_decode_attention_lfm2_bf16_ps16(
    const __nv_bfloat16* __restrict__ query_raw,
    const __nv_bfloat16* __restrict__ key_raw,
    const __nv_bfloat16* __restrict__ value_raw,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t block_table_rows,
    float eps
) {
    fused_decode_attention_body<16, true>(
        query_raw, key_raw, value_raw, query_norm, key_norm, inv_freq,
        key_cache, value_cache, block_tables, request_slots, position_ids,
        slot_mapping, output, num_tokens, num_pages, block_table_stride,
        block_table_rows, eps
    );
}

extern "C" __global__
__launch_bounds__(FUSED_ATTENTION_BLOCK_SIZE)
void fused_ragged_decode_attention_lfm2_bf16_ps32(
    const __nv_bfloat16* __restrict__ query_raw,
    const __nv_bfloat16* __restrict__ key_raw,
    const __nv_bfloat16* __restrict__ value_raw,
    const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ request_slots,
    const uint32_t* __restrict__ position_ids,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens,
    size_t num_pages,
    size_t block_table_stride,
    size_t block_table_rows,
    float eps
) {
    fused_decode_attention_body<32, true>(
        query_raw, key_raw, value_raw, query_norm, key_norm, inv_freq,
        key_cache, value_cache, block_tables, request_slots, position_ids,
        slot_mapping, output, num_tokens, num_pages, block_table_stride,
        block_table_rows, eps
    );
}
