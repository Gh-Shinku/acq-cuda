#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace gemm {

namespace {

template <typename T, size_t BLOCK_SIZE_X, size_t BLOCK_SIZE_Y>
__global__ void sgemm_naive(size_t m, size_t n, size_t k, T const* A,
                            T const* B, T const* C, T* D, T alpha, T beta) {
  size_t const row{blockIdx.y * BLOCK_SIZE_Y + threadIdx.y};
  size_t const col{blockIdx.x * BLOCK_SIZE_X + threadIdx.x};

  if (row >= m || col >= n) {
    return;
  }

  T sum{};
  for (size_t inner{}; inner < k; ++inner) {
    sum += A[row * k + inner] * B[inner * n + col];
  }

  size_t const idx{row * n + col};
  T const previous{(beta != T{} && C != nullptr) ? C[idx] : T{}};
  D[idx] = alpha * sum + beta * previous;
}

} // namespace

void launch_sgemm_naive(const SgemmProblem &problem, const float *a,
                        const float *b, const float *c, float *d,
                        cudaStream_t stream) {
  constexpr size_t BLOCK_SIZE_X{16};
  constexpr size_t BLOCK_SIZE_Y{16};
  dim3 const block_dim{static_cast<unsigned int>(BLOCK_SIZE_X),
                       static_cast<unsigned int>(BLOCK_SIZE_Y), 1};
  dim3 const grid_dim{
      static_cast<unsigned int>((problem.n + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X),
      static_cast<unsigned int>((problem.m + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y),
      1};

  sgemm_naive<float, BLOCK_SIZE_X, BLOCK_SIZE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(
          static_cast<size_t>(problem.m), static_cast<size_t>(problem.n),
          static_cast<size_t>(problem.k), a, b, c, d, problem.alpha,
          problem.beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

} // namespace gemm
