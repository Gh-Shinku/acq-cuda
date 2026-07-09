#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

namespace gemm {

namespace {

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K>
__global__ void sgemm_smem(int m, int n, int k, T const* A,
                           T const* B, T const* C, T* D, T alpha, T beta) {
  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  int const row{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                static_cast<int>(threadIdx.y)};
  int const col{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X +
                static_cast<int>(threadIdx.x)};

  T sum{};

  int const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                   BLOCK_TILE_SIZE_K};
  for (int thread_block_tile_idx{}; thread_block_tile_idx <
                                   num_thread_block_tiles;
       ++thread_block_tile_idx) {
    int const A_row_idx{row};
    int const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                        static_cast<int>(threadIdx.x)};
    if (A_row_idx < m && A_col_idx < k) {
      A_thread_block_tile[threadIdx.y][threadIdx.x] =
          A[A_row_idx * k + A_col_idx];
    } else {
      A_thread_block_tile[threadIdx.y][threadIdx.x] = T{};
    }

    int const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                        static_cast<int>(threadIdx.y)};
    int const B_col_idx{col};
    if (B_row_idx < k && B_col_idx < n) {
      B_thread_block_tile[threadIdx.y][threadIdx.x] =
          B[B_row_idx * n + B_col_idx];
    } else {
      B_thread_block_tile[threadIdx.y][threadIdx.x] = T{};
    }
    __syncthreads();

    for (int k_i{}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
      sum += A_thread_block_tile[threadIdx.y][k_i] *
             B_thread_block_tile[k_i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    int const idx{row * n + col};
    T const previous{(beta != T{} && C != nullptr) ? C[idx] : T{}};
    D[idx] = alpha * sum + beta * previous;
  }
}

} // namespace

void launch_sgemm_smem(const SgemmProblem &problem, const float *a,
                       const float *b, const float *c, float *d,
                       cudaStream_t stream) {
  constexpr int BLOCK_TILE_SIZE_X{16};
  constexpr int BLOCK_TILE_SIZE_Y{16};
  constexpr int BLOCK_TILE_SIZE_K{16};
  dim3 const block_dim{static_cast<unsigned int>(BLOCK_TILE_SIZE_X),
                       static_cast<unsigned int>(BLOCK_TILE_SIZE_Y), 1};
  dim3 const grid_dim{
      static_cast<unsigned int>((problem.n + BLOCK_TILE_SIZE_X - 1) /
                                BLOCK_TILE_SIZE_X),
      static_cast<unsigned int>((problem.m + BLOCK_TILE_SIZE_Y - 1) /
                                BLOCK_TILE_SIZE_Y),
      1};

  sgemm_smem<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
      <<<grid_dim, block_dim, 0, stream>>>(problem.m, problem.n, problem.k, a,
                                           b, c, d, problem.alpha,
                                           problem.beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

} // namespace gemm
