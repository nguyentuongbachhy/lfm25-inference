#define main nvfp4_phase1_perf_main
#include "nvfp4_phase1.cu"
#undef main

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

struct CheckpointOptions {
  std::string site;
  std::string weight_path;
  std::string input_path;
  std::string reference_path;
  int rows = 0;
  int n = 0;
  int k = 0;
};

CheckpointOptions parse_checkpoint_options(int argc, char** argv) {
  CheckpointOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value = [&](const char* key) -> const char* {
      size_t len = std::strlen(key);
      return arg.rfind(key, 0) == 0 ? arg.c_str() + len : nullptr;
    };
    if (const char* v = value("--site=")) {
      options.site = v;
    } else if (const char* v = value("--weight=")) {
      options.weight_path = v;
    } else if (const char* v = value("--input=")) {
      options.input_path = v;
    } else if (const char* v = value("--reference=")) {
      options.reference_path = v;
    } else if (const char* v = value("--rows=")) {
      options.rows = std::atoi(v);
    } else if (const char* v = value("--n=")) {
      options.n = std::atoi(v);
    } else if (const char* v = value("--k=")) {
      options.k = std::atoi(v);
    }
  }
  return options;
}

std::vector<uint16_t> read_bf16_file(const std::string& path, size_t elements) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::fprintf(stderr, "failed to open %s\n", path.c_str());
    std::exit(2);
  }
  std::streamsize bytes = input.tellg();
  size_t expected = elements * sizeof(uint16_t);
  if (bytes < 0 || static_cast<size_t>(bytes) != expected) {
    std::fprintf(stderr, "unexpected BF16 file size for %s: got %lld expected %zu\n",
                 path.c_str(), static_cast<long long>(bytes), expected);
    std::exit(2);
  }
  input.seekg(0);
  std::vector<uint16_t> values(elements);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected));
  if (!input) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    std::exit(2);
  }
  return values;
}

float bf16_bits_to_float_checkpoint(uint16_t bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0f;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

std::vector<int> top_indices(const uint16_t* values, int n, int count) {
  std::vector<int> indices(static_cast<size_t>(n));
  std::iota(indices.begin(), indices.end(), 0);
  count = std::min(count, n);
  std::partial_sort(
      indices.begin(),
      indices.begin() + count,
      indices.end(),
      [&](int left, int right) {
        return bf16_bits_to_float_checkpoint(values[left]) >
               bf16_bits_to_float_checkpoint(values[right]);
      });
  indices.resize(static_cast<size_t>(count));
  return indices;
}

int overlap_count(const std::vector<int>& left, const std::vector<int>& right, int k) {
  k = std::min<int>(k, std::min(left.size(), right.size()));
  int overlap = 0;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      overlap += left[static_cast<size_t>(i)] == right[static_cast<size_t>(j)];
    }
  }
  return overlap;
}

double row_kl(const uint16_t* reference, const uint16_t* candidate, int n) {
  float ref_max = -INFINITY;
  float cand_max = -INFINITY;
  for (int i = 0; i < n; ++i) {
    ref_max = std::max(ref_max, bf16_bits_to_float_checkpoint(reference[i]));
    cand_max = std::max(cand_max, bf16_bits_to_float_checkpoint(candidate[i]));
  }
  double ref_sum = 0.0;
  double cand_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    ref_sum += std::exp(static_cast<double>(bf16_bits_to_float_checkpoint(reference[i]) - ref_max));
    cand_sum += std::exp(static_cast<double>(bf16_bits_to_float_checkpoint(candidate[i]) - cand_max));
  }
  double log_ref_z = static_cast<double>(ref_max) + std::log(ref_sum);
  double log_cand_z = static_cast<double>(cand_max) + std::log(cand_sum);
  double kl = 0.0;
  for (int i = 0; i < n; ++i) {
    double ref_value = bf16_bits_to_float_checkpoint(reference[i]);
    double cand_value = bf16_bits_to_float_checkpoint(candidate[i]);
    double log_p = ref_value - log_ref_z;
    double log_q = cand_value - log_cand_z;
    kl += std::exp(log_p) * (log_p - log_q);
  }
  return kl;
}

int main(int argc, char** argv) {
  CheckpointOptions options = parse_checkpoint_options(argc, argv);
  if (options.site.empty() || options.rows <= 0 || options.rows > 64 || options.n <= 0 ||
      options.k <= 0 || options.n % 128 != 0 || options.k % 128 != 0 ||
      options.weight_path.empty() || options.input_path.empty() ||
      options.reference_path.empty()) {
    std::fprintf(stderr,
                 "checkpoint bridge requires site, rows in [1,64], N%%128=0, K%%128=0 and three BF16 files\n");
    return 2;
  }

  size_t weight_elements = static_cast<size_t>(options.n) * options.k;
  size_t input_elements = static_cast<size_t>(options.rows) * options.k;
  size_t output_elements = static_cast<size_t>(options.rows) * options.n;
  std::vector<uint16_t> weight_host = read_bf16_file(options.weight_path, weight_elements);
  std::vector<uint16_t> input_host = read_bf16_file(options.input_path, input_elements);
  std::vector<uint16_t> reference = read_bf16_file(options.reference_path, output_elements);

  DeviceBuffers buffers;
  CUDA_CHECK(cudaMalloc(&buffers.weight_bf16, weight_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers.input_bf16, input_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers.weight_fp4, weight_elements / 2U));
  CUDA_CHECK(cudaMalloc(&buffers.input_fp4, input_elements / 2U));
  size_t weight_scale_elements = scale_storage_elements(options.n, options.k);
  size_t input_scale_elements = scale_storage_elements(options.rows, options.k);
  CUDA_CHECK(cudaMalloc(&buffers.weight_scale, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&buffers.input_scale, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMalloc(&buffers.c, output_elements * sizeof(ElementC)));
  CUDA_CHECK(cudaMalloc(&buffers.d, output_elements * sizeof(ElementD)));
  CUDA_CHECK(cudaMemcpy(buffers.weight_bf16, weight_host.data(),
                        weight_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.input_bf16, input_host.data(),
                        input_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(buffers.weight_scale, 0, weight_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(buffers.input_scale, 0, input_scale_elements * sizeof(Scale)));
  CUDA_CHECK(cudaMemset(buffers.c, 0, output_elements * sizeof(ElementC)));

  launch_quantize(buffers.weight_bf16, buffers.weight_fp4, buffers.weight_scale,
                  options.n, options.k);
  launch_quantize(buffers.input_bf16, buffers.input_fp4, buffers.input_scale,
                  options.rows, options.k);

  int cutlass_m = options.n;
  int cutlass_n = options.rows;
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {cutlass_m, options.k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {cutlass_n, options.k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {cutlass_m, cutlass_n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {cutlass_m, cutlass_n, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(cutlass_m, cutlass_n, options.k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(cutlass_m, cutlass_n, options.k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {cutlass_m, cutlass_n, options.k, 1},
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

  std::vector<uint16_t> candidate(output_elements);
  CUDA_CHECK(cudaMemcpy(candidate.data(), buffers.d, output_elements * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));

  double diff_sq = 0.0;
  double ref_sq = 0.0;
  double cand_sq = 0.0;
  double dot = 0.0;
  double abs_sum = 0.0;
  double max_abs = 0.0;
  size_t non_finite = 0;
  for (size_t i = 0; i < output_elements; ++i) {
    double ref = bf16_bits_to_float_checkpoint(reference[i]);
    double got = bf16_bits_to_float_checkpoint(candidate[i]);
    if (!std::isfinite(ref) || !std::isfinite(got)) {
      ++non_finite;
      continue;
    }
    double diff = got - ref;
    diff_sq += diff * diff;
    ref_sq += ref * ref;
    cand_sq += got * got;
    dot += ref * got;
    abs_sum += std::abs(diff);
    max_abs = std::max(max_abs, std::abs(diff));
  }

  double nrmse = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : 0.0;
  double cosine = ref_sq > 0.0 && cand_sq > 0.0 ? dot / std::sqrt(ref_sq * cand_sq) : 1.0;
  double mean_abs = abs_sum / static_cast<double>(output_elements);
  double output_rms_ratio = ref_sq > 0.0 ? std::sqrt(cand_sq / ref_sq) : 1.0;

  int top1_matches = 0;
  double top5_overlap = 0.0;
  double top10_overlap = 0.0;
  double mean_kl = 0.0;
  for (int row = 0; row < options.rows; ++row) {
    const uint16_t* ref_row = reference.data() + static_cast<size_t>(row) * options.n;
    const uint16_t* cand_row = candidate.data() + static_cast<size_t>(row) * options.n;
    std::vector<int> ref_top = top_indices(ref_row, options.n, 10);
    std::vector<int> cand_top = top_indices(cand_row, options.n, 10);
    top1_matches += ref_top[0] == cand_top[0];
    top5_overlap += static_cast<double>(overlap_count(ref_top, cand_top, 5)) / 5.0;
    top10_overlap += static_cast<double>(overlap_count(ref_top, cand_top, 10)) / 10.0;
    if (options.site == "lm_head") {
      mean_kl += row_kl(ref_row, cand_row, options.n);
    }
  }

  double rows = static_cast<double>(options.rows);
  double top1_agreement = static_cast<double>(top1_matches) / rows;
  top5_overlap /= rows;
  top10_overlap /= rows;
  if (options.site == "lm_head") {
    mean_kl /= rows;
  } else {
    mean_kl = -1.0;
  }

  std::printf(
      "nvfp4_checkpoint site=%s rows=%d N=%d K=%d tileN=%d nrmse=%.8f cosine=%.8f "
      "max_abs=%.8f mean_abs=%.8f output_rms_ratio=%.8f non_finite=%zu "
      "top1_agreement=%.8f top5_overlap=%.8f top10_overlap=%.8f mean_kl=%.8f\n",
      options.site.c_str(), options.rows, options.n, options.k, NVFP4_TILE_N,
      nrmse, cosine, max_abs, mean_abs, output_rms_ratio, non_finite,
      top1_agreement, top5_overlap, top10_overlap, mean_kl);
  return non_finite == 0 ? 0 : 3;
}
