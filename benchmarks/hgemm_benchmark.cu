#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gemm::fp16_4096::Implementation;
using gemm::fp16_4096::kDimension;

constexpr double kAbsTolerance = 0.10;
constexpr double kRelTolerance = 0.01;

struct Options {
  int device = 0;
  int warmup = 50;
  int samples = 30;
  int launches_per_sample = 10;
  std::vector<std::string> impl_keys;
  std::string csv_path = "benchmark_results/hgemm_fp16_4096.csv";
  bool validate = true;
};

struct Validation {
  bool passed = true;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
};

struct Metrics {
  const char* impl = "";
  float median_ms = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  double median_tflops = 0.0;
  double speedup_vs_cublas = 1.0;
  Validation validation;
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
  T* data() const { return ptr_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

int parse_positive_int(const std::string& option, const char* value) {
  size_t consumed = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(option + " requires a positive integer");
  }
  if (value[0] == '\0' || consumed != std::string(value).size() || parsed <= 0) {
    throw std::invalid_argument(option + " requires a positive integer");
  }
  return parsed;
}

int parse_nonnegative_int(const std::string& option, const char* value) {
  size_t consumed = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(option + " requires a non-negative integer");
  }
  if (value[0] == '\0' || consumed != std::string(value).size() || parsed < 0) {
    throw std::invalid_argument(option + " requires a non-negative integer");
  }
  return parsed;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program
      << " [--device ID] [--warmup N] [--samples N]"
         " [--launches-per-sample N] [--impl cublas,custom] [--csv PATH]"
         " [--no-validate]\n\n"
         "Fixed workload: row-major D=A*B, M=N=K=4096; FP16 A/B/D; FP32 "
         "accumulation.\n"
         "Defaults: 50 warmups, 30 samples, 10 launches per sample.\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const std::string& option) -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value");
      }
      return argv[++i];
    };
    if (arg == "--device") {
      options.device = parse_nonnegative_int(arg, value(arg));
    } else if (arg == "--warmup") {
      options.warmup = parse_nonnegative_int(arg, value(arg));
    } else if (arg == "--samples") {
      options.samples = parse_positive_int(arg, value(arg));
    } else if (arg == "--launches-per-sample") {
      options.launches_per_sample = parse_positive_int(arg, value(arg));
    } else if (arg == "--impl") {
      options.impl_keys = split_csv(value(arg));
      if (options.impl_keys.empty()) {
        throw std::invalid_argument("--impl must select cublas and/or custom");
      }
    } else if (arg == "--csv") {
      options.csv_path = value(arg);
    } else if (arg == "--no-validate") {
      options.validate = false;
    } else if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return options;
}

const char* implementation_key(const Implementation& implementation) {
  return implementation.is_baseline ? "cublas" : "custom";
}

std::vector<Implementation> select_implementations(const Options& options) {
  for (const std::string& key : options.impl_keys) {
    if (key != "cublas" && key != "custom") {
      throw std::invalid_argument("unknown implementation key: " + key);
    }
  }
  std::vector<Implementation> selected;
  for (const Implementation& implementation : gemm::fp16_4096::implementations()) {
    if (options.impl_keys.empty() ||
        std::find(options.impl_keys.begin(), options.impl_keys.end(),
                  implementation_key(implementation)) != options.impl_keys.end()) {
      selected.push_back(implementation);
    }
  }
  if (selected.empty()) {
    throw std::invalid_argument("--impl must select cublas and/or custom");
  }
  const auto baseline = std::find_if(selected.begin(), selected.end(),
                                     [](const Implementation& value) {
                                       return value.is_baseline;
                                     });
  if (baseline == selected.end()) {
    throw std::invalid_argument("--impl must include cublas as the baseline");
  }
  std::rotate(selected.begin(), baseline, baseline + 1);
  return selected;
}

std::vector<__half> make_matrix(int seed) {
  const size_t count = static_cast<size_t>(kDimension) * kDimension;
  std::vector<__half> values(count);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (__half& value : values) {
    value = __float2half_rn(distribution(generator));
  }
  return values;
}

template <typename Fn>
float time_sample(Fn&& fn, int launches) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  GEMM_CUDA_CHECK(cudaEventCreate(&start));
  GEMM_CUDA_CHECK(cudaEventCreate(&stop));
  GEMM_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) {
    fn();
  }
  GEMM_CUDA_CHECK(cudaEventRecord(stop));
  GEMM_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  GEMM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  GEMM_CUDA_CHECK(cudaEventDestroy(start));
  GEMM_CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(launches);
}

Validation compare_results(const std::vector<__half>& expected,
                           const std::vector<__half>& actual) {
  Validation result;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double reference = __half2float(expected[i]);
    const double observed = __half2float(actual[i]);
    const double abs_error = std::abs(reference - observed);
    const double rel_error = abs_error / std::max(1.0, std::abs(reference));
    result.max_abs_error = std::max(result.max_abs_error, abs_error);
    result.max_rel_error = std::max(result.max_rel_error, rel_error);
    if (!std::isfinite(observed) ||
        (abs_error > kAbsTolerance && rel_error > kRelTolerance)) {
      result.passed = false;
    }
  }
  return result;
}

double tflops(float milliseconds) {
  constexpr double operations =
      2.0 * static_cast<double>(kDimension) * kDimension * kDimension;
  return operations / (static_cast<double>(milliseconds) / 1000.0) / 1.0e12;
}

Metrics run_implementation(const Implementation& implementation,
                           const Options& options, const __half* a,
                           const __half* b, __half* d,
                           const std::vector<__half>* baseline) {
  for (int i = 0; i < options.warmup; ++i) {
    implementation.launcher(a, b, d, nullptr);
  }
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> timings;
  timings.reserve(options.samples);
  for (int sample = 0; sample < options.samples; ++sample) {
    timings.push_back(time_sample(
        [&] { implementation.launcher(a, b, d, nullptr); },
        options.launches_per_sample));
  }
  std::sort(timings.begin(), timings.end());

  Metrics metrics;
  metrics.impl = implementation.name;
  metrics.min_ms = timings.front();
  metrics.max_ms = timings.back();
  const size_t middle = timings.size() / 2;
  metrics.median_ms = timings.size() % 2 == 0
                          ? (timings[middle - 1] + timings[middle]) / 2.0f
                          : timings[middle];
  metrics.median_tflops = tflops(metrics.median_ms);

  if (baseline != nullptr && options.validate) {
    std::vector<__half> output(baseline->size());
    GEMM_CUDA_CHECK(cudaMemcpy(output.data(), d, output.size() * sizeof(__half),
                               cudaMemcpyDeviceToHost));
    metrics.validation = compare_results(*baseline, output);
  }
  return metrics;
}

void csv_string(std::ostream& output, const char* value) {
  output << '"';
  for (const char* cursor = value; *cursor != '\0'; ++cursor) {
    if (*cursor == '"') {
      output << '"';
    }
    output << *cursor;
  }
  output << '"';
}

void write_header(std::ostream& output) {
  output << "gpu,compute_capability,impl,M,N,K,fp16_inputs,fp16_output,"
            "fp32_accumulation,median_ms,min_ms,max_ms,median_tflops,"
            "speedup_vs_cublas,valid,max_abs_error,max_rel_error\n";
}

void write_metrics(std::ostream& output, const cudaDeviceProp& device,
                   const Metrics& metrics) {
  csv_string(output, device.name);
  output << ',' << device.major << '.' << device.minor << ',';
  csv_string(output, metrics.impl);
  output << ',' << kDimension << ',' << kDimension << ',' << kDimension
         << ",true,true,true," << std::fixed << std::setprecision(6)
         << metrics.median_ms << ',' << metrics.min_ms << ',' << metrics.max_ms
         << ',' << metrics.median_tflops << ',' << metrics.speedup_vs_cublas
         << ',' << (metrics.validation.passed ? "true" : "false") << ','
         << metrics.validation.max_abs_error << ','
         << metrics.validation.max_rel_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    GEMM_CUDA_CHECK(cudaSetDevice(options.device));
    cudaDeviceProp device{};
    GEMM_CUDA_CHECK(cudaGetDeviceProperties(&device, options.device));
    if (device.major < 12) {
      throw std::runtime_error(
          "hgemm_benchmark requires a Blackwell GPU (compute capability >= 12.0)");
    }

    const std::vector<Implementation> implementations =
        select_implementations(options);
    const size_t elements = static_cast<size_t>(kDimension) * kDimension;
    const std::vector<__half> host_a = make_matrix(42);
    const std::vector<__half> host_b = make_matrix(43);
    DeviceBuffer<__half> dev_a(elements);
    DeviceBuffer<__half> dev_b(elements);
    DeviceBuffer<__half> dev_d(elements);
    GEMM_CUDA_CHECK(cudaMemcpy(dev_a.data(), host_a.data(), dev_a.bytes(),
                               cudaMemcpyHostToDevice));
    GEMM_CUDA_CHECK(cudaMemcpy(dev_b.data(), host_b.data(), dev_b.bytes(),
                               cudaMemcpyHostToDevice));

    std::filesystem::path csv_path(options.csv_path);
    if (csv_path.has_parent_path()) {
      std::filesystem::create_directories(csv_path.parent_path());
    }
    std::ofstream csv(options.csv_path);
    if (!csv) {
      throw std::runtime_error("failed to open CSV output: " + options.csv_path);
    }
    std::cout << "hgemm_benchmark: gpu=\"" << device.name << "\" cc="
              << device.major << '.' << device.minor << " workload=4096x4096x4096"
              << " FP16->FP16, FP32 accumulate\n";
    write_header(std::cout);
    write_header(csv);

    std::vector<__half> baseline;
    float baseline_ms = 0.0f;
    bool all_valid = true;
    for (const Implementation& implementation : implementations) {
      const std::vector<__half>* reference =
          implementation.is_baseline ? nullptr : &baseline;
      Metrics metrics = run_implementation(implementation, options, dev_a.data(),
                                            dev_b.data(), dev_d.data(), reference);
      if (implementation.is_baseline) {
        baseline.resize(elements);
        GEMM_CUDA_CHECK(cudaMemcpy(baseline.data(), dev_d.data(), dev_d.bytes(),
                                   cudaMemcpyDeviceToHost));
        baseline_ms = metrics.median_ms;
      } else {
        metrics.speedup_vs_cublas = baseline_ms / metrics.median_ms;
      }
      all_valid = all_valid && metrics.validation.passed;
      write_metrics(std::cout, device, metrics);
      write_metrics(csv, device, metrics);
    }
    return all_valid ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "hgemm_benchmark: " << error.what() << '\n';
    return 1;
  }
}
