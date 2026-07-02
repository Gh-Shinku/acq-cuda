#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      std::ostringstream oss;                                                  \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "          \
          << cudaGetErrorString(status);                                       \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

namespace {

struct Options {
  std::vector<int> sizes{256, 512, 1024, 2048, 4096};
  int warmup = 5;
  int repeat = 20;
  int device = 0;
  std::string csv_path = "benchmark_results/gemm_benchmark.csv";
};

struct Metrics {
  std::string kernel;
  int m = 0;
  int n = 0;
  int k = 0;
  int warmup = 0;
  int repeat = 0;
  float time_ms = 0.0f;
  double tflops = 0.0;
  double bandwidth_gbps = 0.0;
  double speedup_vs_naive = 1.0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  bool passed = true;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
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

__global__ void sgemm_naive_kernel(int m, int n, int k, const float* a,
                                   const float* b, float* c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    sum += a[row * k + inner] * b[inner * n + col];
  }

  c[row * n + col] = sum;
}

void launch_naive_gemm(int m, int n, int k, const float* a, const float* b,
                       float* c) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  sgemm_naive_kernel<<<grid, block>>>(m, n, k, a, b, c);
  CUDA_CHECK(cudaGetLastError());
}

void launch_cutlass_gemm(int m, int n, int k, const float* a, const float* b,
                         float* c) {
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float,
                                  RowMajor>;

  CutlassGemm gemm;
  typename CutlassGemm::Arguments args({m, n, k}, {a, k}, {b, n}, {c, n},
                                       {c, n}, {1.0f, 0.0f});

  cutlass::Status status = gemm(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM launch failed");
  }
  CUDA_CHECK(cudaGetLastError());
}

std::vector<int> parse_sizes(const std::string& value) {
  std::vector<int> sizes;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
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
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value(arg));
    } else if (arg == "--repeat") {
      options.repeat = std::stoi(require_value(arg));
    } else if (arg == "--device") {
      options.device = std::stoi(require_value(arg));
    } else if (arg == "--csv") {
      options.csv_path = require_value(arg);
    } else if (arg == "--help") {
      std::cout << "Usage: gemm_benchmark [--sizes 256,512] [--warmup N] "
                   "[--repeat N] [--device ID] [--csv PATH]\n";
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

std::vector<float> make_matrix(int rows, int cols, int seed) {
  std::vector<float> values(static_cast<size_t>(rows) * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int mixed = (i * 131 + j * 17 + seed * 29) % 23;
      values[static_cast<size_t>(i) * cols + j] =
          (static_cast<float>(mixed) - 11.0f) / 11.0f;
    }
  }
  return values;
}

template <typename Fn>
float time_kernel(Fn&& fn, int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_ms / static_cast<float>(repeat);
}

void compare_results(const std::vector<float>& expected,
                     const std::vector<float>& actual,
                     double* max_abs_error,
                     double* max_rel_error,
                     bool* passed) {
  constexpr double kAbsTolerance = 1.0e-2;
  constexpr double kRelTolerance = 1.0e-4;

  *max_abs_error = 0.0;
  *max_rel_error = 0.0;
  *passed = true;

  for (size_t i = 0; i < expected.size(); ++i) {
    double abs_error =
        std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    double denom = std::max(1.0, std::abs(static_cast<double>(expected[i])));
    double rel_error = abs_error / denom;

    *max_abs_error = std::max(*max_abs_error, abs_error);
    *max_rel_error = std::max(*max_rel_error, rel_error);

    if (abs_error > kAbsTolerance && rel_error > kRelTolerance) {
      *passed = false;
    }
  }
}

Metrics make_metrics(const std::string& kernel, int m, int n, int k, int warmup,
                     int repeat, float time_ms) {
  double time_sec = static_cast<double>(time_ms) / 1000.0;
  double flops = 2.0 * static_cast<double>(m) * n * k;
  double bytes = sizeof(float) *
                 (static_cast<double>(m) * k + static_cast<double>(k) * n +
                  static_cast<double>(m) * n);

  Metrics metrics;
  metrics.kernel = kernel;
  metrics.m = m;
  metrics.n = n;
  metrics.k = k;
  metrics.warmup = warmup;
  metrics.repeat = repeat;
  metrics.time_ms = time_ms;
  metrics.tflops = flops / time_sec / 1.0e12;
  metrics.bandwidth_gbps = bytes / time_sec / 1.0e9;
  return metrics;
}

std::vector<Metrics> benchmark_size(int size, int warmup, int repeat) {
  int m = size;
  int n = size;
  int k = size;
  size_t a_count = static_cast<size_t>(m) * k;
  size_t b_count = static_cast<size_t>(k) * n;
  size_t c_count = static_cast<size_t>(m) * n;

  std::vector<float> host_a = make_matrix(m, k, 1);
  std::vector<float> host_b = make_matrix(k, n, 2);
  std::vector<float> host_naive(c_count);
  std::vector<float> host_cutlass(c_count);

  DeviceBuffer<float> dev_a(a_count);
  DeviceBuffer<float> dev_b(b_count);
  DeviceBuffer<float> dev_naive(c_count);
  DeviceBuffer<float> dev_cutlass(c_count);

  CUDA_CHECK(cudaMemcpy(dev_a.get(), host_a.data(), dev_a.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b.get(), host_b.data(), dev_b.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dev_naive.get(), 0, dev_naive.bytes()));
  CUDA_CHECK(cudaMemset(dev_cutlass.get(), 0, dev_cutlass.bytes()));

  float naive_ms = time_kernel(
      [&] { launch_naive_gemm(m, n, k, dev_a.get(), dev_b.get(), dev_naive.get()); },
      warmup, repeat);
  CUDA_CHECK(cudaMemcpy(host_naive.data(), dev_naive.get(), dev_naive.bytes(),
                        cudaMemcpyDeviceToHost));

  float cutlass_ms = time_kernel(
      [&] {
        launch_cutlass_gemm(m, n, k, dev_a.get(), dev_b.get(), dev_cutlass.get());
      },
      warmup, repeat);
  CUDA_CHECK(cudaMemcpy(host_cutlass.data(), dev_cutlass.get(), dev_cutlass.bytes(),
                        cudaMemcpyDeviceToHost));

  Metrics naive = make_metrics("naive", m, n, k, warmup, repeat, naive_ms);
  Metrics cutlass = make_metrics("cutlass", m, n, k, warmup, repeat, cutlass_ms);
  cutlass.speedup_vs_naive = naive_ms / cutlass_ms;

  compare_results(host_naive, host_cutlass, &cutlass.max_abs_error,
                  &cutlass.max_rel_error, &cutlass.passed);

  return {naive, cutlass};
}

void write_header(std::ostream& os) {
  os << "kernel,m,n,k,warmup,repeat,time_ms,tflops,effective_bandwidth_gbps,"
        "speedup_vs_naive,max_abs_error,max_rel_error,passed\n";
}

void write_metric(std::ostream& os, const Metrics& metric) {
  os << metric.kernel << ',' << metric.m << ',' << metric.n << ',' << metric.k
     << ',' << metric.warmup << ',' << metric.repeat << ',' << std::fixed
     << std::setprecision(6) << metric.time_ms << ',' << metric.tflops << ','
     << metric.bandwidth_gbps << ',' << metric.speedup_vs_naive << ','
     << metric.max_abs_error << ',' << metric.max_rel_error << ','
     << (metric.passed ? "true" : "false") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.device));

    std::ofstream csv(options.csv_path);
    if (!csv) {
      throw std::runtime_error("failed to open CSV output: " + options.csv_path);
    }

    write_header(csv);
    write_header(std::cout);

    bool all_passed = true;
    for (int size : options.sizes) {
      auto metrics = benchmark_size(size, options.warmup, options.repeat);
      for (const Metrics& metric : metrics) {
        write_metric(csv, metric);
        write_metric(std::cout, metric);
        all_passed = all_passed && metric.passed;
      }
    }

    return all_passed ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "gemm_benchmark: " << e.what() << '\n';
    return 1;
  }
}
