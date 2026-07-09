#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::string impl = "CUDA SMEM";
  gemm::CublasMathMode cublas_math_mode = gemm::CublasMathMode::kFp32;
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 10;
  int repeat = 1;
  int device = 0;
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

void print_usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--impl NAME] [--size N | --m M --n N --k K]"
               " [--cublas-math fp32|default]"
               " [--warmup N] [--repeat N] [--device ID]\n"
            << "Implementations: cuBLAS, CUTLASS, CUDA Naive, CUDA SMEM, "
               "CUDA Thread Tiling\n";
}

int parse_int(const std::string& name, const char* value) {
  std::string_view text(value);
  if (text.empty()) {
    throw std::invalid_argument(name + " requires an integer value");
  }
  size_t parsed_chars = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(std::string(text), &parsed_chars);
  } catch (const std::exception&) {
    throw std::invalid_argument(name + " requires an integer value, got '" +
                                std::string(text) + "'");
  }
  if (parsed_chars != text.size()) {
    throw std::invalid_argument(name + " requires an integer value, got '" +
                                std::string(text) + "'");
  }
  return parsed;
}

int parse_positive_int(const std::string& name, const char* value) {
  int parsed = parse_int(name, value);
  if (parsed <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return parsed;
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
    auto require_value = [&](const std::string& name) -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "--impl") {
      options.impl = require_value(arg);
    } else if (arg == "--cublas-math") {
      options.cublas_math_mode = parse_cublas_math_mode(require_value(arg));
    } else if (arg == "--size") {
      int size = parse_positive_int(arg, require_value(arg));
      options.m = size;
      options.n = size;
      options.k = size;
    } else if (arg == "--m") {
      options.m = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--n") {
      options.n = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--k") {
      options.k = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--warmup") {
      options.warmup = parse_int(arg, require_value(arg));
    } else if (arg == "--repeat") {
      options.repeat = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--device") {
      options.device = parse_int(arg, require_value(arg));
    } else if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (options.warmup < 0) {
    throw std::invalid_argument("--warmup must be >= 0");
  }
  if (options.device < 0) {
    throw std::invalid_argument("--device must be >= 0");
  }

  return options;
}

const gemm::SgemmImplementation& find_implementation(
    const std::string& name) {
  for (const gemm::SgemmImplementation& impl :
       gemm::get_sgemm_implementations()) {
    if (name == impl.name) {
      return impl;
    }
  }

  std::string message = "unknown implementation: " + name + "\navailable:";
  for (const gemm::SgemmImplementation& impl :
       gemm::get_sgemm_implementations()) {
    message += " ";
    message += impl.name;
  }
  throw std::invalid_argument(message);
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

void run_profile(const Options& options,
                 const gemm::SgemmImplementation& impl) {
  gemm::SgemmProblem problem{options.m, options.n, options.k, 1.0f, 0.0f};

  size_t a_count = static_cast<size_t>(problem.m) * problem.k;
  size_t b_count = static_cast<size_t>(problem.k) * problem.n;
  size_t c_count = static_cast<size_t>(problem.m) * problem.n;

  std::vector<float> host_a = make_matrix(problem.m, problem.k, 42);
  std::vector<float> host_b = make_matrix(problem.k, problem.n, 43);
  std::vector<float> host_c(c_count, 0.0f);

  DeviceBuffer<float> dev_a(a_count);
  DeviceBuffer<float> dev_b(b_count);
  DeviceBuffer<float> dev_c(c_count);
  DeviceBuffer<float> dev_d(c_count);

  GEMM_CUDA_CHECK(cudaMemcpy(dev_a.get(), host_a.data(), dev_a.bytes(),
                             cudaMemcpyHostToDevice));
  GEMM_CUDA_CHECK(cudaMemcpy(dev_b.get(), host_b.data(), dev_b.bytes(),
                             cudaMemcpyHostToDevice));
  GEMM_CUDA_CHECK(cudaMemcpy(dev_c.get(), host_c.data(), dev_c.bytes(),
                             cudaMemcpyHostToDevice));
  GEMM_CUDA_CHECK(cudaMemset(dev_d.get(), 0, dev_d.bytes()));

  for (int i = 0; i < options.warmup; ++i) {
    impl.launcher(problem, dev_a.get(), dev_b.get(), dev_c.get(), dev_d.get(),
                  nullptr);
  }
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());

  GEMM_CUDA_CHECK(cudaProfilerStart());
  for (int i = 0; i < options.repeat; ++i) {
    impl.launcher(problem, dev_a.get(), dev_b.get(), dev_c.get(), dev_d.get(),
                  nullptr);
  }
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());
  GEMM_CUDA_CHECK(cudaProfilerStop());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    GEMM_CUDA_CHECK(cudaSetDevice(options.device));
    gemm::set_cublas_math_mode(options.cublas_math_mode);
    const gemm::SgemmImplementation& impl = find_implementation(options.impl);

    std::cout << "gemm_profile: impl=\"" << impl.name << "\""
              << " m=" << options.m << " n=" << options.n
              << " k=" << options.k << " warmup=" << options.warmup
              << " repeat=" << options.repeat
              << " device=" << options.device << "\n";

    run_profile(options, impl);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "gemm_profile: " << e.what() << "\n";
    return 1;
  }
}
