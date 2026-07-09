#ifndef GEMM_CHECK_HPP
#define GEMM_CHECK_HPP

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

#if defined(GEMM_ENABLE_CUTLASS_CHECK)
#include <cutlass/cutlass.h>
#endif

#if defined(GEMM_ENABLE_CUBLAS_CHECK)
#include <cublas_v2.h>
#endif

namespace gemm {

inline void check_cuda(cudaError_t status, const char* expr, const char* file,
                       int line) {
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " while running "
        << expr << ": " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

#if defined(GEMM_ENABLE_CUTLASS_CHECK)
inline void check_cutlass(cutlass::Status status, const char* expr,
                          const char* file, int line) {
  if (status != cutlass::Status::kSuccess) {
    std::ostringstream oss;
    oss << "CUTLASS error at " << file << ":" << line << " while running "
        << expr << ": " << cutlassGetStatusString(status);
    throw std::runtime_error(oss.str());
  }
}
#endif

#if defined(GEMM_ENABLE_CUBLAS_CHECK)
inline const char* cublas_status_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}

inline void check_cublas(cublasStatus_t status, const char* expr,
                         const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "cuBLAS error at " << file << ":" << line << " while running "
        << expr << ": " << cublas_status_string(status);
    throw std::runtime_error(oss.str());
  }
}
#endif

}  // namespace gemm

#define GEMM_CUDA_CHECK(expr) ::gemm::check_cuda((expr), #expr, __FILE__, __LINE__)

#if defined(GEMM_ENABLE_CUTLASS_CHECK)
#define GEMM_CUTLASS_CHECK(expr) \
  ::gemm::check_cutlass((expr), #expr, __FILE__, __LINE__)
#endif

#if defined(GEMM_ENABLE_CUBLAS_CHECK)
#define GEMM_CUBLAS_CHECK(expr) \
  ::gemm::check_cublas((expr), #expr, __FILE__, __LINE__)
#endif

#endif  // GEMM_CHECK_HPP
