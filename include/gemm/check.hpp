#ifndef GEMM_CHECK_HPP
#define GEMM_CHECK_HPP

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

#if defined(GEMM_ENABLE_CUTLASS_CHECK)
#include <cutlass/cutlass.h>
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

}  // namespace gemm

#define GEMM_CUDA_CHECK(expr) ::gemm::check_cuda((expr), #expr, __FILE__, __LINE__)

#if defined(GEMM_ENABLE_CUTLASS_CHECK)
#define GEMM_CUTLASS_CHECK(expr) \
  ::gemm::check_cutlass((expr), #expr, __FILE__, __LINE__)
#endif

#endif  // GEMM_CHECK_HPP
