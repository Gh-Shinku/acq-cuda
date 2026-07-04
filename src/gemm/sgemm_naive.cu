#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

namespace gemm {
namespace {

__global__ void sgemm_naive_kernel(SgemmProblem problem, const float* a,
                                   const float* b, const float* c, float* d) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= problem.m || col >= problem.n) {
    return;
  }

  float sum = 0.0f;
  for (int inner = 0; inner < problem.k; ++inner) {
    sum += a[row * problem.k + inner] * b[inner * problem.n + col];
  }

  int idx = row * problem.n + col;
  float previous = (problem.beta != 0.0f && c != nullptr) ? c[idx] : 0.0f;
  d[idx] = problem.alpha * sum + problem.beta * previous;
}

}  // namespace

void launch_sgemm_naive(const SgemmProblem& problem, const float* a,
                        const float* b, const float* c, float* d,
                        cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((problem.n + block.x - 1) / block.x,
            (problem.m + block.y - 1) / block.y);

  sgemm_naive_kernel<<<grid, block, 0, stream>>>(problem, a, b, c, d);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm
