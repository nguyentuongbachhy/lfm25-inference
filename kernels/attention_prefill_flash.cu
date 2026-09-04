#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

using namespace nvcuda::wmma;

constexpr int ATTENTION_FLASH_BLOCK_SIZE = 128;
constexpr uint32_t LFM2_NUM_Q_HEADS = 32U;
constexpr uint32_t LFM2_NUM_KV_HEADS = 8U;
constexpr uint32_t LFM2_HEAD_DIM = 64U;
constexpr uint32_t LFM2_Q_PER_KV = 4U;
constexpr float LFM2_ATTN_SCALE = 0.125f;
constexpr uint32_t TILE_Q = 16U;
constexpr uint32_t TILE_K = 16U;

// Research-only FlashAttention-style tiled contiguous prefill kernel.
// Uses Blackwell Tensor Cores (wmma) for QK^T and PV tile matrix multiplications,
// completely eliminating per-key scalar warp-shuffle latency serialization.
extern "C" __global__
__launch_bounds__(ATTENTION_FLASH_BLOCK_SIZE)
void prefill_gqa_lfm2_bf16_flash(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output,
    size_t num_tokens
) {
    const size_t q_tile_idx = blockIdx.x / LFM2_NUM_KV_HEADS;
    const uint32_t kv_head = blockIdx.x % LFM2_NUM_KV_HEADS;
    const size_t q_start = q_tile_idx * TILE_Q;

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U; // 0..3 (4 warps = 128 threads)

    __shared__ __align__(16) __nv_bfloat16 s_Q[4][TILE_Q * LFM2_HEAD_DIM]; // 4 * 16 * 64
    __shared__ __align__(16) __nv_bfloat16 s_K[TILE_K * LFM2_HEAD_DIM];     // 16 * 64
    __shared__ __align__(16) __nv_bfloat16 s_V[TILE_K * LFM2_HEAD_DIM];     // 16 * 64

    __shared__ __align__(16) float s_S[4][TILE_Q * TILE_K];                 // 4 * 16 * 16 floats
    __shared__ __align__(16) __nv_bfloat16 s_P[4][TILE_Q * TILE_K];         // 4 * 16 * 16 bf16
    __shared__ __align__(16) float s_O_tile[4][TILE_Q * 16];                // 4 * 16 * 16 floats

    // Load Q into shared memory: 4 heads * 16 tokens * 64 dims = 4096 elements
    for (size_t elem = threadIdx.x; elem < 4 * TILE_Q * LFM2_HEAD_DIM; elem += blockDim.x) {
        const size_t h = elem / (TILE_Q * LFM2_HEAD_DIM);
        const size_t rem = elem % (TILE_Q * LFM2_HEAD_DIM);
        const size_t t = rem / LFM2_HEAD_DIM;
        const size_t d = rem % LFM2_HEAD_DIM;
        const size_t token = q_start + t;
        const uint32_t q_head = kv_head * LFM2_Q_PER_KV + h;
        if (token < num_tokens) {
            const size_t src = (token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM + d;
            s_Q[h][t * LFM2_HEAD_DIM + d] = query[src];
        } else {
            s_Q[h][t * LFM2_HEAD_DIM + d] = __float2bfloat16_rn(0.0f);
        }
    }

    // Running state per thread:
    // Thread lane in warp handles row r = lane / 2.
    // Dims 0..31 if (lane & 1) == 0, dims 32..63 if (lane & 1) == 1.
    const uint32_t r = lane >> 1U;
    const uint32_t d_half = lane & 1U;
    const uint32_t d_base = d_half * 32U;

    float O_acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        O_acc[i] = 0.0f;
    }
    float M_run = -INFINITY;
    float L_run = 0.0f;

    __syncthreads();

    // Loop over Key/Value tiles
    const size_t num_k_tiles = (q_start + TILE_Q - 1 < num_tokens ? q_start + TILE_Q : num_tokens);
    const size_t k_tiles_end = (num_k_tiles + TILE_K - 1) / TILE_K;

    for (size_t k_tile = 0; k_tile < k_tiles_end; ++k_tile) {
        const size_t k_start = k_tile * TILE_K;

        // Load K and V for this tile: 16 * 64 = 1024 elements
        for (size_t elem = threadIdx.x; elem < TILE_K * LFM2_HEAD_DIM; elem += blockDim.x) {
            const size_t t = elem / LFM2_HEAD_DIM;
            const size_t d = elem % LFM2_HEAD_DIM;
            const size_t token = k_start + t;
            if (token < num_tokens) {
                const size_t src = (token * LFM2_NUM_KV_HEADS + kv_head) * LFM2_HEAD_DIM + d;
                s_K[elem] = key[src];
                s_V[elem] = value[src];
            } else {
                s_K[elem] = __float2bfloat16_rn(0.0f);
                s_V[elem] = __float2bfloat16_rn(0.0f);
            }
        }
        __syncthreads();

        // Warp computes Q * K^T
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> q_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> k_frag;
        fragment<accumulator, 16, 16, 16, float> s_frag;
        fill_fragment(s_frag, 0.0f);

        #pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            load_matrix_sync(q_frag, &s_Q[warp][k_step * 16], LFM2_HEAD_DIM);
            load_matrix_sync(k_frag, &s_K[k_step * 16], LFM2_HEAD_DIM);
            mma_sync(s_frag, q_frag, k_frag, s_frag);
        }
        store_matrix_sync(&s_S[warp][0], s_frag, 16, mem_row_major);
        __syncwarp();

        // Softmax reduction for row r
        float m_curr = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const size_t k_token = k_start + j;
            const size_t q_token = q_start + r;
            if (k_token <= q_token && k_token < num_tokens && q_token < num_tokens) {
                const float score = s_S[warp][r * 16 + j] * LFM2_ATTN_SCALE;
                if (score > m_curr) m_curr = score;
            }
        }

        const float m_new = fmaxf(M_run, m_curr);
        const float alpha = (M_run == -INFINITY) ? 0.0f : __expf(M_run - m_new);
        float l_curr = 0.0f;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const size_t k_token = k_start + j;
            const size_t q_token = q_start + r;
            float p = 0.0f;
            if (k_token <= q_token && k_token < num_tokens && q_token < num_tokens && m_curr != -INFINITY) {
                p = __expf(s_S[warp][r * 16 + j] * LFM2_ATTN_SCALE - m_new);
            }
            if (d_half == 0) {
                s_P[warp][r * 16 + j] = __float2bfloat16_rn(p);
            }
            l_curr += p;
        }

        L_run = L_run * alpha + l_curr;
        M_run = m_new;

        // Scale running accumulator by alpha ONCE per tile
        #pragma unroll
        for (int d = 0; d < 32; ++d) {
            O_acc[d] *= alpha;
        }

        __syncwarp();

        // Compute P * V
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_frag;
        load_matrix_sync(p_frag, &s_P[warp][0], 16);

        #pragma unroll
        for (int d_tile = 0; d_tile < 4; ++d_tile) {
            fragment<accumulator, 16, 16, 16, float> o_frag;
            fill_fragment(o_frag, 0.0f);
            load_matrix_sync(v_frag, &s_V[d_tile * 16], LFM2_HEAD_DIM);
            mma_sync(o_frag, p_frag, v_frag, o_frag);
            store_matrix_sync(&s_O_tile[warp][0], o_frag, 16, mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int d = 0; d < 16; ++d) {
                const int global_d = d_tile * 16 + d;
                if (global_d >= (int)d_base && global_d < (int)d_base + 32) {
                    const int local_idx = global_d - (int)d_base;
                    O_acc[local_idx] += s_O_tile[warp][r * 16 + d];
                }
            }
            __syncwarp();
        }

        __syncthreads();
    }

    // Write final normalized output to global memory
    if (q_start + r < num_tokens) {
        const size_t q_token = q_start + r;
        const uint32_t q_head = kv_head * LFM2_Q_PER_KV + warp;
        const size_t out_base = (q_token * LFM2_NUM_Q_HEADS + q_head) * LFM2_HEAD_DIM + d_base;
        const float inv_L = (L_run > 0.0f) ? (1.0f / L_run) : 0.0f;
        #pragma unroll
        for (int d = 0; d < 32; ++d) {
            output[out_base + d] = __float2bfloat16_rn(O_acc[d] * inv_L);
        }
    }
}

