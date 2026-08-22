#define main nvfp4_phase1_perf_main
#include "nvfp4_phase1.cu"
#undef main

#include <algorithm>
#include <limits>
#include <numeric>

__global__ void bf16_gemv_reference(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
    int n,
    int k) {
  int row = blockIdx.x;
  if (row >= n) {
    return;
  }

  float sum = 0.0f;
  const __nv_bfloat16* row_ptr = weight + static_cast<size_t>(row) * k;
  for (int col = threadIdx.x; col < k; col += blockDim.x) {
    sum = fmaf(__bfloat162float(row_ptr[col]), __bfloat162float(input[col]), sum);
  }

  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

float bf16_bits_to_float(uint16_t bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0f;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

std::vector<int> top_indices(const std::vector<float>& values, size_t count) {
  count = std::min(count, values.size());
  std::vector<int> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(),
      indices.begin() + static_cast<std::ptrdiff_t>(count),
      indices.end(),
      [&](int lhs, int rhs) { return values[lhs] > values[rhs]; });
  indices.resize(count);
  return indices;
}

int rank_of(const std::vector<float>& values, int index) {
  float target = values[static_cast<size_t>(index)];
  int rank = 1;
  for (float value : values) {
    if (value > target) {
      ++rank;
    }
  }
  return rank;
}

int overlap_count(
    const std::vector<int>& lhs,
    const std::vector<int>& rhs,
    size_t count) {
  count = std::min(count, std::min(lhs.size(), rhs.size()));
  int overlap = 0;
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < count; ++j) {
      if (lhs[i] == rhs[j]) {
        ++overlap;
        break;
      }
    }
  }
  return overlap;
}

struct AccuracyMetrics {
  double rel_l2 = 0.0;
  double cosine = 0.0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  int reference_top1 = -1;
  int nvfp4_top1 = -1;
  int reference_top1_rank_in_nvfp4 = -1;
  int nvfp4_top1_rank_in_reference = -1;
  int top5_overlap = 0;
  int top10_overlap = 0;
  float reference_top1_value = 0.0f;
  float reference_top1_nvfp4_value = 0.0f;
  float nvfp4_top1_reference_value = 0.0f;
  float nvfp4_top1_value = 0.0f;
  float reference_margin = 0.0f;
  float nvfp4_margin = 0.0f;
  bool finite = true;
};

AccuracyMetrics compute_metrics(
    const std::vector<float>& reference,
    const std::vector<uint16_t>& nvfp4) {
  AccuracyMetrics metrics;
  double diff_sq = 0.0;
  double reference_sq = 0.0;
  double nvfp4_sq = 0.0;
  double dot = 0.0;
  double abs_sum = 0.0;
  std::vector<float> nvfp4_values(reference.size());

  for (size_t i = 0; i < reference.size(); ++i) {
    float ref = reference[i];
    float got = bf16_bits_to_float(nvfp4[i]);
    nvfp4_values[i] = got;
    if (!std::isfinite(ref) || !std::isfinite(got)) {
      metrics.finite = false;
    }

    double diff = static_cast<double>(got) - static_cast<double>(ref);
    diff_sq += diff * diff;
    reference_sq += static_cast<double>(ref) * ref;
    nvfp4_sq += static_cast<double>(got) * got;
    dot += static_cast<double>(ref) * got;
    double abs_diff = std::abs(diff);
    abs_sum += abs_diff;
    metrics.max_abs = std::max(metrics.max_abs, abs_diff);
  }

  metrics.rel_l2 = std::sqrt(diff_sq / std::max(reference_sq, 1.0e-30));
  metrics.cosine = dot / std::sqrt(std::max(reference_sq * nvfp4_sq, 1.0e-30));
  metrics.mean_abs = abs_sum / static_cast<double>(reference.size());

  auto reference_top = top_indices(reference, 10);
  auto nvfp4_top = top_indices(nvfp4_values, 10);
  metrics.reference_top1 = reference_top.front();
  metrics.nvfp4_top1 = nvfp4_top.front();
  metrics.reference_top1_rank_in_nvfp4 = rank_of(nvfp4_values, metrics.reference_top1);
  metrics.nvfp4_top1_rank_in_reference = rank_of(reference, metrics.nvfp4_top1);
  metrics.top5_overlap = overlap_count(reference_top, nvfp4_top, 5);
  metrics.top10_overlap = overlap_count(reference_top, nvfp4_top, 10);
  metrics.reference_top1_value = reference[static_cast<size_t>(metrics.reference_top1)];
  metrics.reference_top1_nvfp4_value =
      nvfp4_values[static_cast<size_t>(metrics.reference_top1)];
  metrics.nvfp4_top1_reference_value = reference[static_cast<size_t>(metrics.nvfp4_top1)];
  metrics.nvfp4_top1_value = nvfp4_values[static_cast<size_t>(metrics.nvfp4_top1)];
  if (reference_top.size() > 1) {
    metrics.reference_margin =
        reference[static_cast<size_t>(reference_top[0])] -
        reference[static_cast<size_t>(reference_top[1])];
  }
  if (nvfp4_top.size() > 1) {
    metrics.nvfp4_margin =
        nvfp4_values[static_cast<size_t>(nvfp4_top[0])] -
        nvfp4_values[static_cast<size_t>(nvfp4_top[1])];
  }
  return metrics;
}

uint32_t parse_seed(int argc, char** argv, const char* key, uint32_t fallback) {
  size_t key_len = std::strlen(key);
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(key, 0) == 0) {
      return static_cast<uint32_t>(std::strtoul(arg.c_str() + key_len, nullptr, 0));
    }
  }
  return fallback;
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (options.m != 1 || options.n <= 0 || options.k <= 0 || options.k % 128 != 0 ||
      options.n % 128 != 0) {
    std::fprintf(stderr, "phase1 accuracy requires M=1, N%%128=0, K%%128=0\n");
    return 2;
  }

  uint32_t weight_seed = parse_seed(argc, argv, "--weight-seed=", 0x9abcU);
  uint32_t input_seed = parse_seed(argc, argv, "--input-seed=", 0x1234U);
  int cutlass_m = options.n;
  int cutlass_n = options.m;
  int k = options.k;
  size_t weight_elements = static_cast<size_t>(options.n) * k;
  size_t input_elements = static_cast<size_t>(options.m) * k;
  size_t output_elements = static_cast<size_t>(options.m) * options.n;
  size_t weight_scale_elements = scale_storage_elements(options.n, k);
  size_t input_scale_elements = scale_storage_elements(options.m, k);

  DeviceBuffers buffers;
  float* reference_device = nullptr;
  CUDA_CHECK(cudaMalloc(&buffers.weight_bf16, weight_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers.input_bf16, input_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers.weight_fp4, weight_elements / 2U));
  CUDA_CHECK(cudaMalloc(&buffers.input_fp4, input_elements / 2U));
  CUDA_CHECK(cudaMalloc(&buffers.weight_scale, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&buffers.input_scale, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&buffers.c, output_elements * sizeof(ElementC)));
  CUDA_CHECK(cudaMalloc(&buffers.d, output_elements * sizeof(ElementD)));
  CUDA_CHECK(cudaMalloc(&reference_device, output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemset(buffers.c, 0, output_elements * sizeof(ElementC)));
  CUDA_CHECK(cudaMemset(buffers.weight_scale, 0, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(buffers.input_scale, 0, input_scale_elements * sizeof(Scale)));

  launch_fill(buffers.weight_bf16, weight_elements, weight_seed);
  launch_fill(buffers.input_bf16, input_elements, input_seed);
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

  constexpr int reference_threads = 256;
  bf16_gemv_reference<<<options.n, reference_threads>>>(
      buffers.weight_bf16,
      buffers.input_bf16,
      reference_device,
      options.n,
      k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> reference(output_elements);
  std::vector<uint16_t> nvfp4(output_elements);
  CUDA_CHECK(cudaMemcpy(
      reference.data(),
      reference_device,
      output_elements * sizeof(float),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      nvfp4.data(),
      buffers.d,
      output_elements * sizeof(uint16_t),
      cudaMemcpyDeviceToHost));
  cudaFree(reference_device);

  AccuracyMetrics metrics = compute_metrics(reference, nvfp4);
  bool top1_agreement = metrics.reference_top1 == metrics.nvfp4_top1;
  std::printf(
      "nvfp4_accuracy site=%s M=%d N=%d K=%d tileN=%d weight_seed=%u input_seed=%u "
      "rel_l2=%.8f cosine=%.8f max_abs=%.8f mean_abs=%.8f finite=%s "
      "reference_top1=%d nvfp4_top1=%d top1_agreement=%s "
      "reference_margin=%.8f nvfp4_margin=%.8f "
      "reference_top1_rank_in_nvfp4=%d nvfp4_top1_rank_in_reference=%d "
      "top5_overlap=%d top10_overlap=%d "
      "reference_top1_value=%.8f reference_top1_nvfp4_value=%.8f "
      "nvfp4_top1_reference_value=%.8f nvfp4_top1_value=%.8f\n",
      options.site.c_str(),
      options.m,
      options.n,
      options.k,
      NVFP4_TILE_N,
      weight_seed,
      input_seed,
      metrics.rel_l2,
      metrics.cosine,
      metrics.max_abs,
      metrics.mean_abs,
      metrics.finite ? "true" : "false",
      metrics.reference_top1,
      metrics.nvfp4_top1,
      top1_agreement ? "true" : "false",
      metrics.reference_margin,
      metrics.nvfp4_margin,
      metrics.reference_top1_rank_in_nvfp4,
      metrics.nvfp4_top1_rank_in_reference,
      metrics.top5_overlap,
      metrics.top10_overlap,
      metrics.reference_top1_value,
      metrics.reference_top1_nvfp4_value,
      metrics.nvfp4_top1_reference_value,
      metrics.nvfp4_top1_value);

  if (!metrics.finite || metrics.rel_l2 > 0.30 || metrics.cosine < 0.95) {
    return 3;
  }
  return 0;
}
