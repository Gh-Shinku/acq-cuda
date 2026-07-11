#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kAbsTolerance = 1.0e-2;
constexpr double kRelTolerance = 1.0e-2;
constexpr double kTf32AbsTolerance = 5.0e-2;
constexpr double kTf32RelTolerance = 2.0e-2;

struct Options {
  std::vector<int> sizes{64,  96,   128,  256,  512,  768,
                         1024, 1536, 2048, 3072, 4096};
  std::vector<std::string> impl_filter;
  gemm::CublasMathMode cublas_math_mode = gemm::CublasMathMode::kFp32;
  int warmup = 10;
  int repeat = 50;
  bool repeat_overridden = false;
  int device = 0;
  std::string csv_path = "benchmark_results/gemm_benchmark_results.csv";
};

struct Metrics {
  int size = 0;
  std::string impl;
  int m = 0;
  int n = 0;
  int k = 0;
  float avg_time_ms = 0.0f;
  double tflops = 0.0;
  double bandwidth_gbs = 0.0;
  double speedup_vs_cublas = 1.0;
  bool valid = true;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    GEMM_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const { return ptr_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

std::vector<std::string> parse_csv_strings(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
}

std::vector<int> parse_sizes(const std::string& value) {
  std::vector<int> sizes;
  for (const std::string& item : parse_csv_strings(value)) {
    int size = std::stoi(item);
    if (size <= 0) {
      throw std::invalid_argument("matrix sizes must be positive");
    }
    sizes.push_back(size);
  }
  if (sizes.empty()) {
    throw std::invalid_argument("--sizes must contain at least one size");
  }
  return sizes;
}

gemm::CublasMathMode parse_cublas_math_mode(const std::string& value) {
  if (value == "fp32") {
    return gemm::CublasMathMode::kFp32;
  }
  if (value == "default") {
    return gemm::CublasMathMode::kDefault;
  }
  throw std::invalid_argument("--cublas-math must be 'fp32' or 'default'");
}

Options parse_args(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "--sizes") {
      options.sizes = parse_sizes(require_value(arg));
    } else if (arg == "--impl") {
      options.impl_filter = parse_csv_strings(require_value(arg));
      if (options.impl_filter.empty()) {
        throw std::invalid_argument("--impl must contain at least one name");
      }
    } else if (arg == "--cublas-math") {
      options.cublas_math_mode = parse_cublas_math_mode(require_value(arg));
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value(arg));
    } else if (arg == "--repeat") {
      options.repeat = std::stoi(require_value(arg));
      options.repeat_overridden = true;
    } else if (arg == "--device") {
      options.device = std::stoi(require_value(arg));
    } else if (arg == "--csv") {
      options.csv_path = require_value(arg);
    } else if (arg == "--help") {
      std::cout << "Usage: gemm_benchmark [--sizes 64,128] "
                   "[--impl cuBLAS,CUTLASS,CUTLASS TensorOp TF32,"
                   "CUDA Naive,CUDA SMEM,CUDA Thread Tiling,"
                   "CUDA Warp Tiling,CUDA Async Tiling,CUDA WMMA TF32] "
                   "[--cublas-math fp32|default] [--warmup N] "
                   "[--repeat N] "
                   "[--device ID] [--csv PATH]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (options.warmup < 0 || options.repeat <= 0) {
    throw std::invalid_argument("--warmup must be >= 0 and --repeat must be > 0");
  }

  return options;
}

bool contains_impl(const std::vector<std::string>& impl_filter,
                   const char* name) {
  return impl_filter.empty() ||
         std::find(impl_filter.begin(), impl_filter.end(), name) !=
             impl_filter.end();
}

std::vector<gemm::SgemmImplementation> selected_implementations(
    const Options& options) {
  std::vector<gemm::SgemmImplementation> selected;
  for (const gemm::SgemmImplementation& impl :
       gemm::get_sgemm_implementations()) {
    if (contains_impl(options.impl_filter, impl.name)) {
      selected.push_back(impl);
    }
  }

  if (selected.empty()) {
    throw std::invalid_argument("no GEMM implementations selected");
  }

  auto baseline =
      std::find_if(selected.begin(), selected.end(),
                   [](const gemm::SgemmImplementation& impl) {
                     return impl.is_baseline;
                   });
  if (baseline == selected.end()) {
    throw std::invalid_argument("selected implementations must include cuBLAS");
  }
  std::rotate(selected.begin(), baseline, baseline + 1);

  return selected;
}

std::vector<float> make_matrix(int rows, int cols, int seed) {
  std::vector<float> values(static_cast<size_t>(rows) * cols);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : values) {
    value = dist(gen);
  }

  return values;
}

int repeat_for_size(int size, const Options& options) {
  if (options.repeat_overridden) {
    return options.repeat;
  }
  if (size >= 4096) {
    return 10;
  }
  if (size >= 2048) {
    return 20;
  }
  return options.repeat;
}

template <typename Fn>
float time_kernel(Fn&& fn, int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  GEMM_CUDA_CHECK(cudaEventCreate(&start));
  GEMM_CUDA_CHECK(cudaEventCreate(&stop));

  GEMM_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    fn();
  }
  GEMM_CUDA_CHECK(cudaEventRecord(stop));
  GEMM_CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  GEMM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  GEMM_CUDA_CHECK(cudaEventDestroy(start));
  GEMM_CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_ms / static_cast<float>(repeat);
}

void compare_results(const std::vector<float>& expected,
                     const std::vector<float>& actual, double abs_tolerance,
                     double rel_tolerance, bool* passed) {
  *passed = true;
  for (size_t i = 0; i < expected.size(); ++i) {
    double abs_error =
        std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    double denom = std::max(1.0, std::abs(static_cast<double>(expected[i])));
    double rel_error = abs_error / denom;

    if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
      *passed = false;
      return;
    }
  }
}

bool uses_approximate_math(const gemm::SgemmImplementation& impl,
                           const Options& options) {
  return impl.accuracy == gemm::SgemmAccuracy::kTf32Approx ||
         options.cublas_math_mode == gemm::CublasMathMode::kDefault;
}

Metrics make_metrics(const std::string& impl, int m, int n, int k,
                     float avg_time_ms) {
  double time_sec = static_cast<double>(avg_time_ms) / 1000.0;
  double flops = 2.0 * static_cast<double>(m) * n * k;
  double bytes = sizeof(float) *
                 (static_cast<double>(m) * k + static_cast<double>(k) * n +
                  static_cast<double>(m) * n);

  Metrics metrics;
  metrics.size = m;
  metrics.impl = impl;
  metrics.m = m;
  metrics.n = n;
  metrics.k = k;
  metrics.avg_time_ms = avg_time_ms;
  metrics.tflops = flops / time_sec / 1.0e12;
  metrics.bandwidth_gbs = bytes / time_sec / 1.0e9;
  return metrics;
}

std::vector<Metrics> benchmark_size(
    int size, const Options& options,
    const std::vector<gemm::SgemmImplementation>& implementations) {
  gemm::SgemmProblem problem{size, size, size, 1.0f, 0.0f};
  int repeat = repeat_for_size(size, options);
  size_t a_count = static_cast<size_t>(problem.m) * problem.k;
  size_t b_count = static_cast<size_t>(problem.k) * problem.n;
  size_t c_count = static_cast<size_t>(problem.m) * problem.n;

  std::vector<float> host_a = make_matrix(problem.m, problem.k, 42);
  std::vector<float> host_b = make_matrix(problem.k, problem.n, 43);
  std::vector<float> host_c(c_count, 0.0f);
  std::vector<float> host_baseline(c_count);

  DeviceBuffer<float> dev_a(a_count);
  DeviceBuffer<float> dev_b(b_count);
  DeviceBuffer<float> dev_c(c_count);
  DeviceBuffer<float> dev_d(c_count);

  GEMM_CUDA_CHECK(
      cudaMemcpy(dev_a.get(), host_a.data(), dev_a.bytes(), cudaMemcpyHostToDevice));
  GEMM_CUDA_CHECK(
      cudaMemcpy(dev_b.get(), host_b.data(), dev_b.bytes(), cudaMemcpyHostToDevice));
  GEMM_CUDA_CHECK(
      cudaMemcpy(dev_c.get(), host_c.data(), dev_c.bytes(), cudaMemcpyHostToDevice));

  std::vector<Metrics> metrics;
  float baseline_ms = 0.0f;

  for (const gemm::SgemmImplementation& impl : implementations) {
    GEMM_CUDA_CHECK(cudaMemset(dev_d.get(), 0, dev_d.bytes()));

    float avg_ms = time_kernel(
        [&] {
          impl.launcher(problem, dev_a.get(), dev_b.get(), dev_c.get(),
                        dev_d.get(), nullptr);
        },
        options.warmup, repeat);

    std::vector<float> host_output(c_count);
    GEMM_CUDA_CHECK(cudaMemcpy(host_output.data(), dev_d.get(), dev_d.bytes(),
                               cudaMemcpyDeviceToHost));

    Metrics impl_metrics =
        make_metrics(impl.name, problem.m, problem.n, problem.k, avg_ms);

    if (impl.is_baseline) {
      baseline_ms = avg_ms;
      host_baseline = std::move(host_output);
      impl_metrics.speedup_vs_cublas = 1.0;
      impl_metrics.valid = true;
    } else {
      if (baseline_ms == 0.0f) {
        throw std::runtime_error("cuBLAS baseline must run before other kernels");
      }
      impl_metrics.speedup_vs_cublas = baseline_ms / avg_ms;
      double const abs_tolerance =
          uses_approximate_math(impl, options) ? kTf32AbsTolerance
                                               : kAbsTolerance;
      double const rel_tolerance =
          uses_approximate_math(impl, options) ? kTf32RelTolerance
                                               : kRelTolerance;
      compare_results(host_baseline, host_output, abs_tolerance, rel_tolerance,
                      &impl_metrics.valid);
    }

    metrics.push_back(impl_metrics);
  }

  return metrics;
}

void write_header(std::ostream& os) {
  os << "size,impl,M,N,K,avg_time_ms,tflops,bandwidth_gbs,"
        "speedup_vs_cublas,valid\n";
}

void write_metric(std::ostream& os, const Metrics& metric) {
  os << metric.size << ',' << metric.impl << ',' << metric.m << ',' << metric.n
     << ',' << metric.k << ',' << std::fixed << std::setprecision(6)
     << metric.avg_time_ms << ',' << metric.tflops << ','
     << metric.bandwidth_gbs << ',' << metric.speedup_vs_cublas << ','
     << (metric.valid ? "true" : "false") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    GEMM_CUDA_CHECK(cudaSetDevice(options.device));
    gemm::set_cublas_math_mode(options.cublas_math_mode);

    std::vector<gemm::SgemmImplementation> implementations =
        selected_implementations(options);

    std::ofstream csv(options.csv_path);
    if (!csv) {
      throw std::runtime_error("failed to open CSV output: " + options.csv_path);
    }

    write_header(csv);
    write_header(std::cout);

    bool all_passed = true;
    for (int size : options.sizes) {
      auto metrics = benchmark_size(size, options, implementations);
      for (const Metrics& metric : metrics) {
        write_metric(csv, metric);
        write_metric(std::cout, metric);
        all_passed = all_passed && metric.valid;
      }
    }

    return all_passed ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "gemm_benchmark: " << e.what() << '\n';
    return 1;
  }
}
