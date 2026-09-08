#define GEMM_ENABLE_CUBLAS_CHECK
#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cublas_v2.h>

namespace gemm::fp16_4096 {
namespace {

cublasHandle_t handle() {
  static thread_local cublasHandle_t value = [] {
    cublasHandle_t created = nullptr;
    GEMM_CUBLAS_CHECK(cublasCreate(&created));
    return created;
  }();
  return value;
}

}  // namespace

void launch_cublas(const __half* a, const __half* b, __half* d,
                   cudaStream_t stream) {
  constexpr int n = kDimension;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  // cuBLAS is column-major. (A * B)^T = B^T * A^T lets the row-major
  // buffers be passed without copies or transposes.
  GEMM_CUBLAS_CHECK(cublasSetStream(handle(), stream));
  GEMM_CUBLAS_CHECK(cublasGemmEx(
      handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b, CUDA_R_16F, n,
      a, CUDA_R_16F, n, &beta, d, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace gemm::fp16_4096
