#define GEMM_ENABLE_CUBLAS_CHECK
#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace gemm {
namespace {

CublasMathMode g_cublas_math_mode = CublasMathMode::kFp32;

cublasHandle_t get_handle() {
  static thread_local cublasHandle_t handle = [] {
    cublasHandle_t created = nullptr;
    GEMM_CUBLAS_CHECK(cublasCreate(&created));
    return created;
  }();
  return handle;
}

cublasMath_t to_cublas_math_mode(CublasMathMode mode) {
  switch (mode) {
    case CublasMathMode::kFp32:
      return CUBLAS_PEDANTIC_MATH;
    case CublasMathMode::kDefault:
      return CUBLAS_DEFAULT_MATH;
  }
  return CUBLAS_PEDANTIC_MATH;
}

}  // namespace

void set_cublas_math_mode(CublasMathMode mode) {
  g_cublas_math_mode = mode;
}

CublasMathMode get_cublas_math_mode() {
  return g_cublas_math_mode;
}

void launch_sgemm_cublas(const SgemmProblem& problem, const float* a,
                         const float* b, const float* c, float* d,
                         cudaStream_t stream) {
  if (problem.beta != 0.0f && c == nullptr) {
    throw std::invalid_argument("cuBLAS SGEMM requires C when beta is non-zero");
  }
  if (problem.beta != 0.0f && c != d) {
    size_t bytes = static_cast<size_t>(problem.m) * problem.n * sizeof(float);
    GEMM_CUDA_CHECK(cudaMemcpyAsync(d, c, bytes, cudaMemcpyDeviceToDevice, stream));
  }

  cublasHandle_t handle = get_handle();
  GEMM_CUBLAS_CHECK(cublasSetStream(handle, stream));
  GEMM_CUBLAS_CHECK(
      cublasSetMathMode(handle, to_cublas_math_mode(g_cublas_math_mode)));

  const float beta = problem.beta;
  GEMM_CUBLAS_CHECK(cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, problem.n, problem.m, problem.k,
      &problem.alpha, b, problem.n, a, problem.k, &beta, d, problem.n));
}

}  // namespace gemm
