#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifndef NVFP4_TILE_N
#define NVFP4_TILE_N 8
#endif

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Scale = typename ElementA::ScaleFactorType;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using Accumulator = float;
using Arch = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using Tile = Shape<_128, Int<NVFP4_TILE_N>, _128>;
using Cluster = Shape<_1, _1, _1>;

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentC = 8;
constexpr int kAlignmentD = 8;
constexpr int kScaleVector = 16;
constexpr size_t kL2FlushBytes = 128ull * 1024ull * 1024ull;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch,
    OperatorClass,
    Tile,
    Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator,
    Accumulator,
    ElementC,
    cutlass::layout::ColumnMajor,
    kAlignmentC,
    ElementD,
    cutlass::layout::ColumnMajor,
    kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch,
    OperatorClass,
    ElementA,
    cutlass::layout::RowMajor,
    kAlignmentA,
    ElementB,
    cutlass::layout::ColumnMajor,
    kAlignmentB,
    Accumulator,
    Tile,
    Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    Mainloop,
    Epilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t status_ = (expr);                                                 \
    if (status_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(status_));                                 \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

#define CUTLASS_CHECK(expr)                                                      \
  do {                                                                           \
    cutlass::Status status_ = (expr);                                             \
    if (status_ != cutlass::Status::kSuccess) {                                   \
      std::fprintf(stderr, "CUTLASS error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cutlassGetStatusString(status_));                              \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

struct DeviceBuffers {
  __nv_bfloat16* weight_bf16 = nullptr;
  __nv_bfloat16* input_bf16 = nullptr;
  uint8_t* weight_fp4 = nullptr;
  uint8_t* input_fp4 = nullptr;
  Scale* weight_scale = nullptr;
  Scale* input_scale = nullptr;
  ElementC* c = nullptr;
  ElementD* d = nullptr;
  uint8_t* workspace = nullptr;
  uint32_t* flush = nullptr;

  ~DeviceBuffers() {
    cudaFree(weight_bf16);
    cudaFree(input_bf16);
    cudaFree(weight_fp4);
    cudaFree(input_fp4);
    cudaFree(weight_scale);
    cudaFree(input_scale);
    cudaFree(c);
    cudaFree(d);
    cudaFree(workspace);
    cudaFree(flush);
  }
};

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__global__ void fill_bf16(__nv_bfloat16* dst, size_t count, uint32_t seed) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  uint32_t bits = mix32(static_cast<uint32_t>(i) + seed);
  float unit = static_cast<float>(bits & 0xffffU) * (1.0f / 65535.0f);
  dst[i] = __float2bfloat16_rn((unit - 0.5f) * 1.5f);
}

__global__ void touch_flush(uint32_t* data, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = data[i] * 1664525U + static_cast<uint32_t>(i) + 1013904223U;
  }
}

__device__ __forceinline__ size_t scale_offset(
    size_t row,
    size_t block_k,
    size_t blocks_k) {
  size_t row_tile = row >> 7U;
  size_t local_row = row & 127U;
  size_t k_tile = block_k >> 2U;
  size_t local_k = block_k & 3U;
  size_t k_tiles = (blocks_k + 3U) >> 2U;
  size_t tile = row_tile * k_tiles + k_tile;
  size_t local = (local_row & 31U) * 16U + (local_row >> 5U) * 4U + local_k;
  return tile * 512U + local;
}

__global__ void quantize_bf16_nvfp4(
    const __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst,
    Scale* __restrict__ scales,
    size_t rows,
    size_t k) {
  size_t blocks_k = k / kScaleVector;
  size_t logical_block = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_blocks = rows * blocks_k;
  if (logical_block >= total_blocks) {
    return;
  }

  size_t row = logical_block / blocks_k;
  size_t block_k = logical_block - row * blocks_k;
  size_t base = row * k + block_k * kScaleVector;

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kScaleVector; ++i) {
    amax = fmaxf(amax, fabsf(__bfloat162float(src[base + static_cast<size_t>(i)])));
  }

  Scale scale = amax == 0.0f ? Scale(1.0f) : Scale(amax / 6.0f);
  float scale_f = static_cast<float>(scale);
  if (!(scale_f > 0.0f) || !isfinite(scale_f)) {
    scale = Scale(1.0f);
    scale_f = 1.0f;
  }
  scales[scale_offset(row, block_k, blocks_k)] = scale;

  float inv = 1.0f / scale_f;
  size_t packed_base = row * (k / 2U) + block_k * 8U;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    float v0 = __bfloat162float(src[base + static_cast<size_t>(pair * 2)]) * inv;
    float v1 = __bfloat162float(src[base + static_cast<size_t>(pair * 2 + 1)]) * inv;
    cutlass::float_e2m1_t q0(v0);
    cutlass::float_e2m1_t q1(v1);
    uint8_t lo = static_cast<uint8_t>(q0.raw()) & 0x0fU;
    uint8_t hi = static_cast<uint8_t>(q1.raw()) & 0x0fU;
    dst[packed_base + static_cast<size_t>(pair)] = static_cast<uint8_t>(lo | (hi << 4U));
  }
}

size_t scale_storage_elements(size_t rows, size_t k) {
  size_t row_tiles = (rows + 127U) / 128U;
  size_t blocks_k = k / kScaleVector;
  size_t k_tiles = (blocks_k + 3U) / 4U;
  return row_tiles * k_tiles * 512U;
}

void launch_fill(__nv_bfloat16* ptr, size_t count, uint32_t seed) {
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1U) / threads);
  fill_bf16<<<blocks, threads>>>(ptr, count, seed);
  CUDA_CHECK(cudaGetLastError());
}

void launch_quantize(
    const __nv_bfloat16* src,
    uint8_t* dst,
    Scale* scales,
    size_t rows,
    size_t k,
    cudaStream_t stream = nullptr) {
  size_t logical_blocks = rows * (k / kScaleVector);
  constexpr int threads = 256;
  int blocks = static_cast<int>((logical_blocks + threads - 1U) / threads);
  quantize_bf16_nvfp4<<<blocks, threads, 0, stream>>>(src, dst, scales, rows, k);
  CUDA_CHECK(cudaGetLastError());
}

float timed_hot(int iterations, const std::function<void()>& fn) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms * 1000.0f / static_cast<float>(iterations);
}

float timed_cold(
    int iterations,
    uint32_t* flush,
    size_t flush_count,
    const std::function<void()>& fn) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  constexpr int threads = 256;
  int flush_blocks = static_cast<int>((flush_count + threads - 1U) / threads);
  double total_us = 0.0;
  for (int i = 0; i < iterations; ++i) {
    touch_flush<<<flush_blocks, threads>>>(flush, flush_count);
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_us += static_cast<double>(elapsed_ms) * 1000.0;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<float>(total_us / static_cast<double>(iterations));
}

struct Options {
  std::string site = "unknown";
  int m = 1;
  int n = 2048;
  int k = 8192;
  int iterations = 100;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value = [&](const char* key) -> const char* {
      size_t len = std::strlen(key);
      return arg.rfind(key, 0) == 0 ? arg.c_str() + len : nullptr;
    };
    if (const char* v = value("--site=")) {
      options.site = v;
    } else if (const char* v = value("--m=")) {
      options.m = std::atoi(v);
    } else if (const char* v = value("--n=")) {
      options.n = std::atoi(v);
    } else if (const char* v = value("--k=")) {
      options.k = std::atoi(v);
    } else if (const char* v = value("--iterations=")) {
      options.iterations = std::atoi(v);
    }
  }
  return options;
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (options.m != 1 || options.n <= 0 || options.k <= 0 || options.k % 128 != 0 ||
      options.n % 128 != 0 || options.iterations <= 0) {
    std::fprintf(stderr, "phase1 requires M=1, N%%128=0, K%%128=0\n");
    return 2;
  }

  int cutlass_m = options.n;
  int cutlass_n = options.m;
  int k = options.k;
  size_t weight_elements = static_cast<size_t>(options.n) * k;
  size_t input_elements = static_cast<size_t>(options.m) * k;
  size_t output_elements = static_cast<size_t>(options.m) * options.n;
  size_t weight_scale_elements = scale_storage_elements(options.n, k);
  size_t input_scale_elements = scale_storage_elements(options.m, k);

  DeviceBuffers buffers;
  CUDA_CHECK(cudaMalloc(&buffers.weight_bf16, weight_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers.input_bf16, input_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers.weight_fp4, weight_elements / 2U));
  CUDA_CHECK(cudaMalloc(&buffers.input_fp4, input_elements / 2U));
  CUDA_CHECK(cudaMalloc(&buffers.weight_scale, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&buffers.input_scale, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&buffers.c, output_elements * sizeof(ElementC)));
  CUDA_CHECK(cudaMalloc(&buffers.d, output_elements * sizeof(ElementD)));
  size_t flush_count = kL2FlushBytes / sizeof(uint32_t);
  CUDA_CHECK(cudaMalloc(&buffers.flush, kL2FlushBytes));
  CUDA_CHECK(cudaMemset(buffers.c, 0, output_elements * sizeof(ElementC)));
  CUDA_CHECK(cudaMemset(buffers.weight_scale, 0, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(buffers.input_scale, 0, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(buffers.flush, 0x5a, kL2FlushBytes));

  launch_fill(buffers.weight_bf16, weight_elements, 0x9abcU);
  launch_fill(buffers.input_bf16, input_elements, 0x1234U);
  launch_quantize(
      buffers.weight_bf16,
      buffers.weight_fp4,
      buffers.weight_scale,
      options.n,
      k);
  launch_quantize(
      buffers.input_bf16,
      buffers.input_fp4,
      buffers.input_scale,
      options.m,
      k);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {cutlass_m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {cutlass_n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {cutlass_m, cutlass_n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {cutlass_m, cutlass_n, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(cutlass_m, cutlass_n, k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(cutlass_m, cutlass_n, k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {cutlass_m, cutlass_n, k, 1},
      {reinterpret_cast<cutlass::float_e2m1_t*>(buffers.weight_fp4),
       stride_a,
       reinterpret_cast<cutlass::float_e2m1_t*>(buffers.input_fp4),
       stride_b,
       buffers.weight_scale,
       layout_sfa,
       buffers.input_scale,
       layout_sfb},
      {{1.0f, 0.0f}, buffers.c, stride_c, buffers.d, stride_d}};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&buffers.workspace, workspace_size));
  }
  CUTLASS_CHECK(gemm.initialize(arguments, buffers.workspace));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < 10; ++i) {
    launch_quantize(
        buffers.input_bf16,
        buffers.input_fp4,
        buffers.input_scale,
        options.m,
        k);
    CUTLASS_CHECK(gemm.run());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  float quant_hot_us = timed_hot(options.iterations, [&]() {
    launch_quantize(
        buffers.input_bf16,
        buffers.input_fp4,
        buffers.input_scale,
        options.m,
        k);
  });
  float gemm_hot_us = timed_hot(options.iterations, [&]() { CUTLASS_CHECK(gemm.run()); });
  float e2e_hot_us = timed_hot(options.iterations, [&]() {
    launch_quantize(
        buffers.input_bf16,
        buffers.input_fp4,
        buffers.input_scale,
        options.m,
        k);
    CUTLASS_CHECK(gemm.run());
  });
  float e2e_cold_us = timed_cold(
      options.iterations,
      buffers.flush,
      flush_count,
      [&]() {
        launch_quantize(
            buffers.input_bf16,
            buffers.input_fp4,
            buffers.input_scale,
            options.m,
            k);
        CUTLASS_CHECK(gemm.run());
      });

  std::printf(
      "nvfp4_phase1 site=%s M=%d N=%d K=%d tileN=%d quant_hot_us=%.3f "
      "gemm_hot_us=%.3f e2e_hot_us=%.3f e2e_cold_us=%.3f\n",
      options.site.c_str(),
      options.m,
      options.n,
      options.k,
      NVFP4_TILE_N,
      quant_hot_us,
      gemm_hot_us,
      e2e_hot_us,
      e2e_cold_us);
  return 0;
}
