#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double kAbsTolerance = 1.0e-3;
constexpr double kRelTolerance = 1.0e-3;
constexpr double kTf32AbsTolerance = 5.0e-2;
constexpr double kTf32RelTolerance = 2.0e-2;

struct Options {
  bool full = false;
  gemm::CublasMathMode cublas_math_mode = gemm::CublasMathMode::kFp32;
  int device = 0;
};

struct TestCase {
  int m = 0;
  int n = 0;
  int k = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  const char* name = "";
};

struct Comparison {
  bool passed = true;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t mismatch_index = 0;
  float expected = 0.0f;
  float actual = 0.0f;
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
  std::cout << "Usage: " << program << " [--quick | --full] [--device ID]\n"
            << "\n"
            << "Options:\n"
            << "  --quick      Run small and boundary correctness cases (default)\n"
            << "  --full       Run all correctness cases\n"
            << "  --cublas-math fp32|default\n"
            << "  --device ID  CUDA device id (default: 0)\n"
            << "  --help       Show this help\n";
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

int parse_nonnegative_int(const std::string& name, const char* value) {
  int parsed = parse_int(name, value);
  if (parsed < 0) {
    throw std::invalid_argument(name + " must be non-negative");
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

    if (arg == "--quick") {
      options.full = false;
    } else if (arg == "--full") {
      options.full = true;
    } else if (arg == "--cublas-math") {
      options.cublas_math_mode = parse_cublas_math_mode(require_value(arg));
    } else if (arg == "--device") {
      options.device = parse_nonnegative_int(arg, require_value(arg));
    } else if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return options;
}

std::vector<TestCase> make_test_cases(bool full) {
  std::vector<TestCase> cases = {
      {1, 1, 1, 1.0f, 0.0f, "one_element"},
      {2, 3, 4, 1.0f, 0.0f, "tiny_rect"},
      {16, 16, 16, 1.0f, 0.0f, "tile_exact_16"},
      {17, 17, 17, 1.0f, 0.0f, "tile_plus_one_17"},
      {31, 33, 17, 1.0f, 0.0f, "non_square_boundary"},
      {32, 32, 32, 0.0f, 1.0f, "beta_only"},
      {64, 64, 64, 1.0f, 0.0f, "wmma_tf32_aligned"},
      {33, 31, 17, 1.25f, -0.5f, "alpha_beta_mixed"},
  };

  if (full) {
    cases.push_back({3, 2, 5, 1.0f, 0.0f, "tiny_transposed_rect"});
    cases.push_back({7, 9, 5, 1.0f, 0.0f, "small_rect"});
    cases.push_back({64, 17, 31, 1.0f, 0.0f, "wide_k_tail"});
    cases.push_back({127, 65, 33, 1.0f, 0.0f, "large_rect_tail"});
    cases.push_back({32, 33, 31, 1.25f, -0.5f, "alpha_beta_tail"});
  }

  return cases;
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

std::vector<float> reference_sgemm(const TestCase& test,
                                   const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   const std::vector<float>& c) {
  std::vector<float> d(static_cast<size_t>(test.m) * test.n);
  for (int row = 0; row < test.m; ++row) {
    for (int col = 0; col < test.n; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < test.k; ++inner) {
        sum += static_cast<double>(a[row * test.k + inner]) *
               static_cast<double>(b[inner * test.n + col]);
      }
      const size_t idx = static_cast<size_t>(row) * test.n + col;
      double previous = static_cast<double>(c[idx]);
      d[idx] = static_cast<float>(static_cast<double>(test.alpha) * sum +
                                  static_cast<double>(test.beta) * previous);
    }
  }
  return d;
}

Comparison compare_results(const std::vector<float>& expected,
                           const std::vector<float>& actual,
                           double abs_tolerance,
                           double rel_tolerance) {
  Comparison result;
  for (size_t i = 0; i < expected.size(); ++i) {
    double abs_error =
        std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    double denom = std::max(1.0, std::abs(static_cast<double>(expected[i])));
    double rel_error = abs_error / denom;

    result.max_abs_error = std::max(result.max_abs_error, abs_error);
    result.max_rel_error = std::max(result.max_rel_error, rel_error);

    if (result.passed && abs_error > abs_tolerance &&
        rel_error > rel_tolerance) {
      result.passed = false;
      result.mismatch_index = i;
      result.expected = expected[i];
      result.actual = actual[i];
    }
  }
  return result;
}

bool uses_approximate_math(const gemm::SgemmImplementation& impl) {
  return impl.accuracy == gemm::SgemmAccuracy::kTf32Approx ||
         (impl.is_baseline &&
          gemm::get_cublas_math_mode() == gemm::CublasMathMode::kDefault);
}

Comparison run_case(const gemm::SgemmImplementation& impl,
                    const TestCase& test) {
  size_t a_count = static_cast<size_t>(test.m) * test.k;
  size_t b_count = static_cast<size_t>(test.k) * test.n;
  size_t c_count = static_cast<size_t>(test.m) * test.n;

  std::vector<float> host_a = make_matrix(test.m, test.k, 100 + test.m);
  std::vector<float> host_b = make_matrix(test.k, test.n, 200 + test.n);
  std::vector<float> host_c = make_matrix(test.m, test.n, 300 + test.k);
  std::vector<float> expected = reference_sgemm(test, host_a, host_b, host_c);

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
  GEMM_CUDA_CHECK(cudaMemset(dev_d.get(), 0, dev_d.bytes()));

  gemm::SgemmProblem problem{test.m, test.n, test.k, test.alpha, test.beta};
  impl.launcher(problem, dev_a.get(), dev_b.get(), dev_c.get(), dev_d.get(),
                nullptr);
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> actual(c_count);
  GEMM_CUDA_CHECK(cudaMemcpy(actual.data(), dev_d.get(), dev_d.bytes(),
                             cudaMemcpyDeviceToHost));

  double const abs_tolerance =
      uses_approximate_math(impl) ? kTf32AbsTolerance : kAbsTolerance;
  double const rel_tolerance =
      uses_approximate_math(impl) ? kTf32RelTolerance : kRelTolerance;
  return compare_results(expected, actual, abs_tolerance, rel_tolerance);
}

void print_failure(const gemm::SgemmImplementation& impl, const TestCase& test,
                   const Comparison& comparison) {
  int row = static_cast<int>(comparison.mismatch_index / test.n);
  int col = static_cast<int>(comparison.mismatch_index % test.n);
  std::cerr << std::setprecision(8) << "FAIL impl='" << impl.name
            << "' case='" << test.name << "' shape=" << test.m << "x"
            << test.n << "x" << test.k << " alpha=" << test.alpha
            << " beta=" << test.beta << " mismatch=(" << row << "," << col
            << ") expected=" << comparison.expected
            << " actual=" << comparison.actual
            << " max_abs_error=" << comparison.max_abs_error
            << " max_rel_error=" << comparison.max_rel_error << "\n";
}

int run_correctness(const Options& options) {
  GEMM_CUDA_CHECK(cudaSetDevice(options.device));
  gemm::set_cublas_math_mode(options.cublas_math_mode);

  std::vector<TestCase> cases = make_test_cases(options.full);
  int failures = 0;
  int total = 0;

  for (const gemm::SgemmImplementation& impl :
       gemm::get_sgemm_implementations()) {
    for (const TestCase& test : cases) {
      ++total;
      Comparison comparison = run_case(impl, test);
      if (!comparison.passed) {
        ++failures;
        print_failure(impl, test, comparison);
      }
    }
  }

  if (failures == 0) {
    std::cout << "GEMM correctness passed: " << total << " cases across "
              << gemm::get_sgemm_implementations().size()
              << " implementations\n";
  } else {
    std::cerr << "GEMM correctness failed: " << failures << " / " << total
              << " cases\n";
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    return run_correctness(options);
  } catch (const std::exception& ex) {
    std::cerr << "gemm_correctness_test: " << ex.what() << "\n";
    return 1;
  }
}
