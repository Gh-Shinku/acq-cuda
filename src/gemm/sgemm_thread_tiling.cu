#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace gemm {

namespace {

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void sgemm_thread_tiling(size_t m, size_t n, size_t k, T const *A,
                                    T const *B, T const *C, T *D, T alpha,
                                    T beta) {
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

  constexpr size_t NUM_THREAD_TILES_X{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X};
  constexpr size_t NUM_THREAD_TILES_Y{BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  constexpr size_t NUM_THREADS_PER_BLOCK{NUM_THREAD_TILES_X *
                                         NUM_THREAD_TILES_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X]{};

  for (size_t thread_block_tile_idx{};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx) {
    for (size_t load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K;
         load_offset += NUM_THREADS_PER_BLOCK) {
      size_t const A_thread_block_tile_row_idx{load_offset / BLOCK_TILE_SIZE_K};
      size_t const A_thread_block_tile_col_idx{load_offset % BLOCK_TILE_SIZE_K};
      size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                             A_thread_block_tile_row_idx};
      size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                             A_thread_block_tile_col_idx};
      if (A_row_idx < m && A_col_idx < k) {
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] =
                               A[A_row_idx * k + A_col_idx];
      } else {
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = T{};
      }
    }

    for (size_t load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
         load_offset += NUM_THREADS_PER_BLOCK) {
      size_t const B_thread_block_tile_row_idx{load_offset / BLOCK_TILE_SIZE_X};
      size_t const B_thread_block_tile_col_idx{load_offset % BLOCK_TILE_SIZE_X};
      size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                             B_thread_block_tile_row_idx};
      size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                             B_thread_block_tile_col_idx};
      if (B_row_idx < k && B_col_idx < n) {
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] =
                               B[B_row_idx * n + B_col_idx];
      } else {
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = T{};
      }
    }
    __syncthreads();

    size_t const thread_tile_row_base{thread_linear_idx / NUM_THREAD_TILES_X *
                                      THREAD_TILE_SIZE_Y};
    size_t const thread_tile_col_base{thread_linear_idx % NUM_THREAD_TILES_X *
                                      THREAD_TILE_SIZE_X};

    for (size_t k_i{}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
      for (size_t thread_tile_row_idx{};
           thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
        size_t const A_thread_block_tile_row_idx{thread_tile_row_base +
                                                 thread_tile_row_idx};
        T const A_val{A_thread_block_tile[A_thread_block_tile_row_idx][k_i]};
        for (size_t thread_tile_col_idx{};
             thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
          size_t const B_thread_block_tile_col_idx{thread_tile_col_base +
                                                   thread_tile_col_idx};
          T const B_val{B_thread_block_tile[k_i][B_thread_block_tile_col_idx]};
          C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=
              A_val * B_val;
        }
      }
    }
    __syncthreads();
  }

  size_t const thread_tile_row_base{thread_linear_idx /
                                    (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                                    THREAD_TILE_SIZE_Y};
  size_t const thread_tile_col_base{thread_linear_idx %
                                    (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                                    THREAD_TILE_SIZE_X};

  for (size_t thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
       ++thread_tile_row_idx) {
    size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                           thread_tile_row_base + thread_tile_row_idx};
    for (size_t thread_tile_col_idx{}; thread_tile_col_idx < THREAD_TILE_SIZE_X;
         ++thread_tile_col_idx) {
      size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                             thread_tile_col_base + thread_tile_col_idx};

      if (C_row_idx < m && C_col_idx < n) {
        size_t const C_idx{C_row_idx * n + C_col_idx};
        T const previous{(beta != T{} && C != nullptr) ? C[C_idx] : T{}};
        D[C_idx] =
            alpha * C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +
            beta * previous;
      }
    }
  }
}

} // namespace

template <typename T>
void launch_sgemm_thread_tiling(size_t m, size_t n, size_t k, T const *A,
                                T const *B, T const *C, T *D, T alpha, T beta,
                                cudaStream_t stream) {
  constexpr size_t BLOCK_TILE_SIZE_X{64};
  constexpr size_t BLOCK_TILE_SIZE_Y{64};
  constexpr size_t BLOCK_TILE_SIZE_K{8};
  constexpr size_t THREAD_TILE_SIZE_X{8};
  constexpr size_t THREAD_TILE_SIZE_Y{8};
  constexpr size_t NUM_THREADS_PER_BLOCK{
      BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
      THREAD_TILE_SIZE_Y};
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);
  dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1, 1};
  dim3 const grid_dim{static_cast<unsigned int>((n + BLOCK_TILE_SIZE_X - 1) /
                                                BLOCK_TILE_SIZE_X),
                      static_cast<unsigned int>((m + BLOCK_TILE_SIZE_Y - 1) /
                                                BLOCK_TILE_SIZE_Y),
                      1};
  sgemm_thread_tiling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                      BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(m, n, k, A, B, C, D, alpha, beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

void launch_sgemm_thread_tiling(const SgemmProblem &problem, const float *a,
                                const float *b, const float *c, float *d,
                                cudaStream_t stream) {
  launch_sgemm_thread_tiling<float>(static_cast<size_t>(problem.m),
                                    static_cast<size_t>(problem.n),
                                    static_cast<size_t>(problem.k), a, b, c, d,
                                    problem.alpha, problem.beta, stream);
}

} // namespace gemm
