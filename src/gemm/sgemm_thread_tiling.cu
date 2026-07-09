#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace gemm {

namespace {

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_X,
          int THREAD_TILE_SIZE_Y>
__global__ void sgemm_thread_tiling(int m, int n, int k, T const* A,
                                    T const* B, T const* C, T* D, T alpha,
                                    T beta) {
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

  constexpr int NUM_THREAD_TILES_X{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X};
  constexpr int NUM_THREAD_TILES_Y{BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  constexpr int NUM_THREADS_PER_BLOCK{NUM_THREAD_TILES_X *
                                      NUM_THREAD_TILES_Y};

  int const thread_linear_idx{static_cast<int>(threadIdx.y * blockDim.x +
                                               threadIdx.x)};

  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  int const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                   BLOCK_TILE_SIZE_K};

  T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X]{};

  for (int thread_block_tile_idx{}; thread_block_tile_idx <
                                   num_thread_block_tiles;
       ++thread_block_tile_idx) {
    for (int load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K;
         load_offset += NUM_THREADS_PER_BLOCK) {
      int const A_thread_block_tile_row_idx{load_offset / BLOCK_TILE_SIZE_K};
      int const A_thread_block_tile_col_idx{load_offset % BLOCK_TILE_SIZE_K};
      int const A_row_idx{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                          A_thread_block_tile_row_idx};
      int const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
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

    for (int load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
         load_offset += NUM_THREADS_PER_BLOCK) {
      int const B_thread_block_tile_row_idx{load_offset / BLOCK_TILE_SIZE_X};
      int const B_thread_block_tile_col_idx{load_offset % BLOCK_TILE_SIZE_X};
      int const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                          B_thread_block_tile_row_idx};
      int const B_col_idx{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X +
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

    int const thread_tile_row_base{thread_linear_idx / NUM_THREAD_TILES_X *
                                   THREAD_TILE_SIZE_Y};
    int const thread_tile_col_base{thread_linear_idx % NUM_THREAD_TILES_X *
                                   THREAD_TILE_SIZE_X};

    for (int k_i{}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
      for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
           ++thread_tile_row_idx) {
        int const A_thread_block_tile_row_idx{thread_tile_row_base +
                                              thread_tile_row_idx};
        T const A_val{A_thread_block_tile[A_thread_block_tile_row_idx][k_i]};

        for (int thread_tile_col_idx{};
             thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
          int const B_thread_block_tile_col_idx{thread_tile_col_base +
                                                thread_tile_col_idx};
          T const B_val{B_thread_block_tile[k_i][B_thread_block_tile_col_idx]};
          C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=
              A_val * B_val;
        }
      }
    }
    __syncthreads();
  }

  int const thread_tile_row_base{thread_linear_idx / NUM_THREAD_TILES_X *
                                 THREAD_TILE_SIZE_Y};
  int const thread_tile_col_base{thread_linear_idx % NUM_THREAD_TILES_X *
                                 THREAD_TILE_SIZE_X};

  for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
       ++thread_tile_row_idx) {
    int const C_row_idx{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                        thread_tile_row_base + thread_tile_row_idx};
    for (int thread_tile_col_idx{}; thread_tile_col_idx < THREAD_TILE_SIZE_X;
         ++thread_tile_col_idx) {
      int const C_col_idx{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X +
                          thread_tile_col_base + thread_tile_col_idx};

      if (C_row_idx < m && C_col_idx < n) {
        int const C_idx{C_row_idx * n + C_col_idx};
        T const previous{(beta != T{} && C != nullptr) ? C[C_idx] : T{}};
        D[C_idx] =
            alpha * C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +
            beta * previous;
      }
    }
  }
}

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_X,
          int THREAD_TILE_SIZE_Y>
void launch_sgemm_thread_tiling_variant(int m, int n, int k, T const* A,
                                        T const* B, T const* C, T* D, T alpha,
                                        T beta, cudaStream_t stream) {
  constexpr int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X *
                                      BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

  dim3 const block_dim{static_cast<unsigned int>(NUM_THREADS_PER_BLOCK), 1, 1};
  dim3 const grid_dim{
      static_cast<unsigned int>((n + BLOCK_TILE_SIZE_X - 1) /
                                BLOCK_TILE_SIZE_X),
      static_cast<unsigned int>((m + BLOCK_TILE_SIZE_Y - 1) /
                                BLOCK_TILE_SIZE_Y),
      1};

  sgemm_thread_tiling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                      BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X,
                      THREAD_TILE_SIZE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(m, n, k, A, B, C, D, alpha, beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_sgemm_thread_tiling(int m, int n, int k, T const* A, T const* B,
                                T const* C, T* D, T alpha, T beta,
                                cudaStream_t stream) {
  int const max_mn{std::max(m, n)};
  if (max_mn <= 256) {
    launch_sgemm_thread_tiling_variant<T, 16, 16, 16, 1, 1>(
        m, n, k, A, B, C, D, alpha, beta, stream);
  } else if (max_mn <= 512) {
    launch_sgemm_thread_tiling_variant<T, 32, 32, 16, 2, 2>(
        m, n, k, A, B, C, D, alpha, beta, stream);
  } else {
    launch_sgemm_thread_tiling_variant<T, 64, 64, 8, 4, 4>(
        m, n, k, A, B, C, D, alpha, beta, stream);
  }
}

}  // namespace

void launch_sgemm_thread_tiling(const SgemmProblem& problem, const float* a,
                                const float* b, const float* c, float* d,
                                cudaStream_t stream) {
  launch_sgemm_thread_tiling<float>(problem.m, problem.n, problem.k, a, b, c, d,
                                    problem.alpha, problem.beta, stream);
}

}  // namespace gemm
