#include <cuda_bf16.h>

#include <stddef.h>
#include <stdint.h>

constexpr int KV_CACHE_LFM2_BLOCK_SIZE = 256;

constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_PAIRS_PER_HEAD = LFM2_HEAD_DIM >> 1;
constexpr uint32_t LFM2_PAIRS_PER_TOKEN =
    LFM2_NUM_KV_HEADS * LFM2_PAIRS_PER_HEAD;

template <int PAGE_SIZE>
struct PageTraits;

template <>
struct PageTraits<16> {
    static constexpr uint32_t SHIFT = 4U;
    static constexpr size_t MASK = 15U;
};

template <>
struct PageTraits<32> {
    static constexpr uint32_t SHIFT = 5U;
    static constexpr size_t MASK = 31U;
};

template <int PAGE_SIZE>
__device__ __forceinline__ void kv_cache_write_lfm2_bf16_body(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    size_t num_tokens,
    size_t num_pages
) {
    const size_t token = blockIdx.x;

    if (token >= num_tokens) {
        return;
    }

    const int64_t slot = slot_mapping[token];

    if (slot < 0) {
        return;
    }

    const size_t physical_slot =
        static_cast<size_t>(slot);

    const size_t page =
        physical_slot >> PageTraits<PAGE_SIZE>::SHIFT;

    if (page >= num_pages) {
        return;
    }

    const size_t page_offset =
        physical_slot & PageTraits<PAGE_SIZE>::MASK;

    constexpr size_t cache_head_stride =
        PAGE_SIZE * LFM2_PAIRS_PER_HEAD;

    constexpr size_t cache_page_stride =
        LFM2_NUM_KV_HEADS * cache_head_stride;

    const size_t source_base =
        token * LFM2_PAIRS_PER_TOKEN;

    const size_t cache_page_base =
        page * cache_page_stride;

    const size_t cache_offset_base =
        page_offset * LFM2_PAIRS_PER_HEAD;

    const __nv_bfloat162* __restrict__ key_vec =
        reinterpret_cast<const __nv_bfloat162*>(key);

    const __nv_bfloat162* __restrict__ value_vec =
        reinterpret_cast<const __nv_bfloat162*>(value);

    __nv_bfloat162* __restrict__ key_cache_vec =
        reinterpret_cast<__nv_bfloat162*>(key_cache);

    __nv_bfloat162* __restrict__ value_cache_vec =
        reinterpret_cast<__nv_bfloat162*>(value_cache);

    for (
        uint32_t work = threadIdx.x;
        work < LFM2_PAIRS_PER_TOKEN;
        work += blockDim.x
    ) {
        const uint32_t head = work >> 5;
        const uint32_t pair = work & 31U;

        const size_t source_index =
            source_base + work;

        const size_t cache_index =
            cache_page_base
            + head * cache_head_stride
            + cache_offset_base
            + pair;

        const __nv_bfloat162 key_value =
            key_vec[source_index];

        const __nv_bfloat162 value_value =
            value_vec[source_index];

        key_cache_vec[cache_index] =
            key_value;

        value_cache_vec[cache_index] =
            value_value;
    }
}

extern "C" __global__
__launch_bounds__(KV_CACHE_LFM2_BLOCK_SIZE)
void kv_cache_write_lfm2_bf16_ps16(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    size_t num_tokens,
    size_t num_pages
) {
    kv_cache_write_lfm2_bf16_body<16>(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        num_pages
    );
}

extern "C" __global__
__launch_bounds__(KV_CACHE_LFM2_BLOCK_SIZE)
void kv_cache_write_lfm2_bf16_ps32(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    size_t num_tokens,
    size_t num_pages
) {
    kv_cache_write_lfm2_bf16_body<32>(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        num_pages
    );
}