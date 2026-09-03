#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr unsigned int CLUSTER_BLOCKS = 8;

extern "C" __global__ __cluster_dims__(CLUSTER_BLOCKS, 1, 1)
void dsm_handoff_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    unsigned int elements) {
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned int rank = cluster.block_rank();

    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto* local = reinterpret_cast<__nv_bfloat16*>(shared_bytes);

    if (rank == 0) {
        for (unsigned int index = threadIdx.x; index < elements; index += blockDim.x) {
            local[index] = input[index];
        }
    }

    cluster.sync();

    auto* producer = cluster.map_shared_rank(local, 0);
    if (rank > 0) {
        const unsigned int output_base = (rank - 1) * elements;
        for (unsigned int index = threadIdx.x; index < elements; index += blockDim.x) {
            output[output_base + index] = producer[index];
        }
    }

    cluster.sync();
}

extern "C" __global__ __cluster_dims__(CLUSTER_BLOCKS, 1, 1)
void global_handoff_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* scratch,
    __nv_bfloat16* output,
    unsigned int elements) {
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned int rank = cluster.block_rank();

    if (rank == 0) {
        for (unsigned int index = threadIdx.x; index < elements; index += blockDim.x) {
            scratch[index] = input[index];
        }
    }

    cluster.sync();

    if (rank > 0) {
        const unsigned int output_base = (rank - 1) * elements;
        for (unsigned int index = threadIdx.x; index < elements; index += blockDim.x) {
            output[output_base + index] = scratch[index];
        }
    }

    cluster.sync();
}
