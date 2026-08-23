#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Scale = typename ElementA::ScaleFactorType;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using Accumulator = float;
using Arch = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = cute::Shape<cute::_128, cute::Int<NVFP4_TILE_N>, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentC = 8;
constexpr int kAlignmentD = 8;
constexpr int kScaleVector = 16;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
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
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
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

struct Options {
  std::string site = "unknown";
  std::string weight;
  std::string input;
  std::string output;
  int n = 0;
  int k = 0;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value = [&](const char* key) -> const char* {
      const size_t len = std::strlen(key);
      return arg.rfind(key, 0) == 0 ? arg.c_str() + len : nullptr;
    };
    if (const char* v = value("--site=")) {
      options.site = v;
    } else if (const char* v = value("--weight=")) {
      options.weight = v;
    } else if (const char* v = value("--input=")) {
      options.input = v;
    } else if (const char* v = value("--output=")) {
      options.output = v;
    } else if (const char* v = value("--n=")) {
      options.n = std::atoi(v);
    } else if (const char* v = value("--k=")) {
      options.k = std::atoi(v);
    }
  }
  return options;
}

std::vector<uint16_t> read_bf16_bits(const std::string& path, size_t expected) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "failed to open %s\n", path.c_str());
    std::exit(2);
  }
  const std::streamsize bytes = file.tellg();
  if (bytes != static_cast<std::streamsize>(expected * sizeof(uint16_t))) {
    std::fprintf(stderr, "unexpected BF16 file size for %s: got %lld expected %zu\n",
                 path.c_str(), static_cast<long long>(bytes), expected * sizeof(uint16_t));
    std::exit(2);
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint16_t> values(expected);
  if (!file.read(reinterpret_cast<char*>(values.data()), bytes)) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    std::exit(2);
  }
  return values;
}

void write_bf16_bits(const std::string& path, const std::vector<uint16_t>& values) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    std::fprintf(stderr, "failed to create %s\n", path.c_str());
    std::exit(2);
  }
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(uint16_t)));
  if (!file) {
    std::fprintf(stderr, "failed to write %s\n", path.c_str());
    std::exit(2);
  }
}

__device__ __forceinline__ size_t scale_offset(
    size_t row,
    size_t block_k,
    size_t blocks_k) {
  const size_t row_tile = row >> 7U;
  const size_t local_row = row & 127U;
  const size_t k_tile = block_k >> 2U;
  const size_t local_k = block_k & 3U;
  const size_t k_tiles = (blocks_k + 3U) >> 2U;
  const size_t tile = row_tile * k_tiles + k_tile;
  const size_t local = (local_row & 31U) * 16U + (local_row >> 5U) * 4U + local_k;
  return tile * 512U + local;
}

__global__ void quantize_bf16_nvfp4_nearest(
    const __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst,
    Scale* __restrict__ scales,
    size_t rows,
    size_t k) {
  const size_t blocks_k = k / kScaleVector;
  const size_t logical_block = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total_blocks = rows * blocks_k;
  if (logical_block >= total_blocks) {
    return;
  }

  const size_t row = logical_block / blocks_k;
  const size_t block_k = logical_block - row * blocks_k;
  const size_t base = row * k + block_k * kScaleVector;

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

  const float inv = 1.0f / scale_f;
  const size_t packed_base = row * (k / 2U) + block_k * 8U;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float v0 = __bfloat162float(src[base + static_cast<size_t>(pair * 2)]) * inv;
    const float v1 = __bfloat162float(src[base + static_cast<size_t>(pair * 2 + 1)]) * inv;
    cutlass::float_e2m1_t q0(v0);
    cutlass::float_e2m1_t q1(v1);
    const uint8_t lo = static_cast<uint8_t>(q0.raw()) & 0x0fU;
    const uint8_t hi = static_cast<uint8_t>(q1.raw()) & 0x0fU;
    dst[packed_base + static_cast<size_t>(pair)] = static_cast<uint8_t>(lo | (hi << 4U));
  }
}

size_t scale_storage_elements(size_t rows, size_t k) {
  const size_t row_tiles = (rows + 127U) / 128U;
  const size_t blocks_k = k / kScaleVector;
  const size_t k_tiles = (blocks_k + 3U) / 4U;
  return row_tiles * k_tiles * 512U;
}

void launch_quantize(
    const __nv_bfloat16* src,
    uint8_t* dst,
    Scale* scales,
    size_t rows,
    size_t k) {
  const size_t logical_blocks = rows * (k / kScaleVector);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((logical_blocks + threads - 1U) / threads);
  quantize_bf16_nvfp4_nearest<<<blocks, threads>>>(src, dst, scales, rows, k);
  CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.n <= 0 || options.k <= 0 || options.n % 128 != 0 || options.k % 128 != 0 ||
      options.weight.empty() || options.input.empty() || options.output.empty()) {
    std::fprintf(stderr, "usage: nvfp4_replay_site --site=NAME --n=N --k=K --weight=PATH --input=PATH --output=PATH\n");
    return 2;
  }

  constexpr int rows = 1;
  const int cutlass_m = options.n;
  const int cutlass_n = rows;
  const int k = options.k;
  const size_t weight_elements = static_cast<size_t>(options.n) * k;
  const size_t input_elements = static_cast<size_t>(k);
  const size_t output_elements = static_cast<size_t>(options.n);
  const size_t weight_scale_elements = scale_storage_elements(options.n, k);
  const size_t input_scale_elements = scale_storage_elements(rows, k);

  const auto weight_host = read_bf16_bits(options.weight, weight_elements);
  const auto input_host = read_bf16_bits(options.input, input_elements);

  __nv_bfloat16* weight_bf16 = nullptr;
  __nv_bfloat16* input_bf16 = nullptr;
  uint8_t* weight_fp4 = nullptr;
  uint8_t* input_fp4 = nullptr;
  Scale* weight_scale = nullptr;
  Scale* input_scale = nullptr;
  ElementC* c = nullptr;
  ElementD* d = nullptr;
  uint8_t* workspace = nullptr;

  CUDA_CHECK(cudaMalloc(&weight_bf16, weight_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&input_bf16, input_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&weight_fp4, weight_elements / 2U));
  CUDA_CHECK(cudaMalloc(&input_fp4, input_elements / 2U));
  CUDA_CHECK(cudaMalloc(&weight_scale, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&input_scale, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&c, output_elements * sizeof(ElementC)));
  CUDA_CHECK(cudaMalloc(&d, output_elements * sizeof(ElementD)));
  CUDA_CHECK(cudaMemcpy(weight_bf16, weight_host.data(), weight_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(input_bf16, input_host.data(), input_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(weight_scale, 0, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(input_scale, 0, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(c, 0, output_elements * sizeof(ElementC)));

  launch_quantize(weight_bf16, weight_fp4, weight_scale, options.n, k);
  launch_quantize(input_bf16, input_fp4, input_scale, rows, k);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {cutlass_m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {cutlass_n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {cutlass_m, cutlass_n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {cutlass_m, cutlass_n, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(cutlass_m, cutlass_n, k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(cutlass_m, cutlass_n, k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {cutlass_m, cutlass_n, k, 1},
      {reinterpret_cast<cutlass::float_e2m1_t*>(weight_fp4),
       stride_a,
       reinterpret_cast<cutlass::float_e2m1_t*>(input_fp4),
       stride_b,
       weight_scale,
       layout_sfa,
       input_scale,
       layout_sfb},
      {{1.0f, 0.0f}, c, stride_c, d, stride_d}};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }
  CUTLASS_CHECK(gemm.initialize(arguments, workspace));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint16_t> output_host(output_elements);
  CUDA_CHECK(cudaMemcpy(output_host.data(), d, output_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  write_bf16_bits(options.output, output_host);

  cudaFree(workspace);
  cudaFree(d);
  cudaFree(c);
  cudaFree(input_scale);
  cudaFree(weight_scale);
  cudaFree(input_fp4);
  cudaFree(weight_fp4);
  cudaFree(input_bf16);
  cudaFree(weight_bf16);

  std::printf("nvfp4_replay site=%s N=%d K=%d tileN=%d output=%s\n",
              options.site.c_str(), options.n, options.k, NVFP4_TILE_N, options.output.c_str());
  return 0;
}
