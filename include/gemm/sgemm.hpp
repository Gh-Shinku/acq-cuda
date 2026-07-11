#ifndef GEMM_SGEMM_HPP
#define GEMM_SGEMM_HPP

#include <cuda_runtime_api.h>

#include <vector>

namespace gemm {

struct SgemmProblem {
  int m = 0;
  int n = 0;
  int k = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
};

enum class CublasMathMode {
  kFp32,
  kDefault,
};

using SgemmLauncher = void (*)(const SgemmProblem& problem,
                               const float* a,
                               const float* b,
                               const float* c,
                               float* d,
                               cudaStream_t stream);

struct SgemmImplementation {
  const char* name;
  SgemmLauncher launcher;
  bool is_baseline;
};

void launch_sgemm_naive(const SgemmProblem& problem,
                        const float* a,
                        const float* b,
                        const float* c,
                        float* d,
                        cudaStream_t stream);

void launch_sgemm_smem(const SgemmProblem& problem,
                       const float* a,
                       const float* b,
                       const float* c,
                       float* d,
                       cudaStream_t stream);

void launch_sgemm_thread_tiling(const SgemmProblem& problem,
                                const float* a,
                                const float* b,
                                const float* c,
                                float* d,
                                cudaStream_t stream);

void launch_sgemm_warp_tiling(const SgemmProblem& problem,
                              const float* a,
                              const float* b,
                              const float* c,
                              float* d,
                              cudaStream_t stream);

void launch_sgemm_cutlass(const SgemmProblem& problem,
                          const float* a,
                          const float* b,
                          const float* c,
                          float* d,
                          cudaStream_t stream);

void launch_sgemm_cublas(const SgemmProblem& problem,
                         const float* a,
                         const float* b,
                         const float* c,
                         float* d,
                         cudaStream_t stream);

void set_cublas_math_mode(CublasMathMode mode);
CublasMathMode get_cublas_math_mode();

const std::vector<SgemmImplementation>& get_sgemm_implementations();

}  // namespace gemm

#endif  // GEMM_SGEMM_HPP
