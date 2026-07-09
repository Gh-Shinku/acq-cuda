#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace gemm {

namespace {

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void sgemm_smem(size_t m, size_t n, size_t k, T const* A,
                           T const* B, T const* C, T* D, T alpha, T beta) {
  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const row{blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.y};
  size_t const col{blockIdx.x * BLOCK_TILE_SIZE_X + threadIdx.x};

  T sum{};

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};
  for (size_t thread_block_tile_idx{}; thread_block_tile_idx <
                                      num_thread_block_tiles;
       ++thread_block_tile_idx) {
    size_t const A_row_idx{row};
    size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           threadIdx.x};
    if (A_row_idx < m && A_col_idx < k) {
      A_thread_block_tile[threadIdx.y][threadIdx.x] =
          A[A_row_idx * k + A_col_idx];
    } else {
      A_thread_block_tile[threadIdx.y][threadIdx.x] = T{};
    }

    size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           threadIdx.y};
    size_t const B_col_idx{col};
    if (B_row_idx < k && B_col_idx < n) {
      B_thread_block_tile[threadIdx.y][threadIdx.x] =
          B[B_row_idx * n + B_col_idx];
    } else {
      B_thread_block_tile[threadIdx.y][threadIdx.x] = T{};
    }
    __syncthreads();

    for (size_t k_i{}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
      sum += A_thread_block_tile[threadIdx.y][k_i] *
             B_thread_block_tile[k_i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    size_t const idx{row * n + col};
    T const previous{(beta != T{} && C != nullptr) ? C[idx] : T{}};
    D[idx] = alpha * sum + beta * previous;
  }
}

} // namespace

void launch_sgemm_smem(const SgemmProblem &problem, const float *a,
                       const float *b, const float *c, float *d,
                       cudaStream_t stream) {
  constexpr size_t BLOCK_TILE_SIZE_X{16};
  constexpr size_t BLOCK_TILE_SIZE_Y{16};
  constexpr size_t BLOCK_TILE_SIZE_K{16};
  dim3 const block_dim{static_cast<unsigned int>(BLOCK_TILE_SIZE_X),
                       static_cast<unsigned int>(BLOCK_TILE_SIZE_Y), 1};
  dim3 const grid_dim{
      static_cast<unsigned int>((problem.n + BLOCK_TILE_SIZE_X - 1) /
                                BLOCK_TILE_SIZE_X),
      static_cast<unsigned int>((problem.m + BLOCK_TILE_SIZE_Y - 1) /
                                BLOCK_TILE_SIZE_Y),
      1};

  sgemm_smem<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
      <<<grid_dim, block_dim, 0, stream>>>(
          static_cast<size_t>(problem.m), static_cast<size_t>(problem.n),
          static_cast<size_t>(problem.k), a, b, c, d, problem.alpha,
          problem.beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

} // namespace gemm
