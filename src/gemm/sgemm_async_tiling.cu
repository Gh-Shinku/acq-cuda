#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace gemm {

namespace {

template <typename T>
__device__ __forceinline__ void async_copy_shared_global(T* shared_ptr,
                                                         T const* global_ptr,
                                                         bool predicate) {
  static_assert(sizeof(T) == 4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned const shared_addr{
      static_cast<unsigned>(__cvta_generic_to_shared(shared_ptr))};
  unsigned const pred{predicate ? 1u : 0u};
  asm volatile(
      "{ .reg .pred p;\n"
      "  setp.ne.u32 p, %2, 0;\n"
      "  @p cp.async.ca.shared.global [%0], [%1], 4;\n"
      "  @!p st.shared.u32 [%0], 0;\n"
      "}\n" ::"r"(shared_addr),
      "l"(global_ptr), "r"(pred));
#else
  *shared_ptr = predicate ? *global_ptr : T{};
#endif
}

template <typename T>
__device__ __forceinline__ void async_copy_shared_global_16(T* shared_ptr,
                                                            T const* global_ptr) {
  static_assert(sizeof(T) == 4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned const shared_addr{
      static_cast<unsigned>(__cvta_generic_to_shared(shared_ptr))};
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                   shared_addr),
               "l"(global_ptr));
#else
#pragma unroll
  for (int i{}; i < 4; ++i) {
    shared_ptr[i] = global_ptr[i];
  }
#endif
}

__device__ __forceinline__ void async_copy_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void async_copy_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_X,
          int THREAD_TILE_SIZE_Y>
__global__ void sgemm_async_tiling(int m, int n, int k, T const* A,
                                   T const* B, T const* C, T* D, T alpha,
                                   T beta) {
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

  constexpr int NUM_STAGES{2};
  constexpr int NUM_THREAD_TILES_X{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X};
  constexpr int NUM_THREAD_TILES_Y{BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  constexpr int NUM_THREADS_PER_BLOCK{NUM_THREAD_TILES_X *
                                      NUM_THREAD_TILES_Y};

  int const thread_linear_idx{
      static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x)};
  int const thread_tile_row_base{thread_linear_idx / NUM_THREAD_TILES_X *
                                 THREAD_TILE_SIZE_Y};
  int const thread_tile_col_base{thread_linear_idx % NUM_THREAD_TILES_X *
                                 THREAD_TILE_SIZE_X};

  __shared__ T A_thread_block_tile[NUM_STAGES][BLOCK_TILE_SIZE_Y]
                                  [BLOCK_TILE_SIZE_K + 1];
  __shared__ T B_thread_block_tile[NUM_STAGES][BLOCK_TILE_SIZE_K]
                                  [BLOCK_TILE_SIZE_X + 1];

  int const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                   BLOCK_TILE_SIZE_K};

  T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X]{};
  T A_thread_regs[THREAD_TILE_SIZE_Y]{};
  T B_thread_regs[THREAD_TILE_SIZE_X]{};

  auto load_tile = [&](int stage, int thread_block_tile_idx) {
#pragma unroll
    for (int load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K;
         load_offset += NUM_THREADS_PER_BLOCK) {
      int const A_thread_block_tile_row_idx{load_offset / BLOCK_TILE_SIZE_K};
      int const A_thread_block_tile_col_idx{load_offset % BLOCK_TILE_SIZE_K};
      int const A_row_idx{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                          A_thread_block_tile_row_idx};
      int const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                          A_thread_block_tile_col_idx};
      bool const valid{A_row_idx < m && A_col_idx < k};
      T const* src{valid ? A + A_row_idx * k + A_col_idx : A};
      async_copy_shared_global(
          &A_thread_block_tile[stage][A_thread_block_tile_row_idx]
                              [A_thread_block_tile_col_idx],
          src, valid);
    }

#pragma unroll
    for (int load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
         load_offset += NUM_THREADS_PER_BLOCK) {
      int const B_thread_block_tile_row_idx{load_offset / BLOCK_TILE_SIZE_X};
      int const B_thread_block_tile_col_idx{load_offset % BLOCK_TILE_SIZE_X};
      int const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                          B_thread_block_tile_row_idx};
      int const B_col_idx{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X +
                          B_thread_block_tile_col_idx};
      bool const valid{B_row_idx < k && B_col_idx < n};
      T const* src{valid ? B + B_row_idx * n + B_col_idx : B};
      async_copy_shared_global(
          &B_thread_block_tile[stage][B_thread_block_tile_row_idx]
                              [B_thread_block_tile_col_idx],
          src, valid);
    }
  };

  if (num_thread_block_tiles > 0) {
    load_tile(0, 0);
    async_copy_commit_group();
  }

  for (int thread_block_tile_idx{}; thread_block_tile_idx <
                                   num_thread_block_tiles;
       ++thread_block_tile_idx) {
    int const stage{thread_block_tile_idx % NUM_STAGES};
    int const next_tile_idx{thread_block_tile_idx + 1};
    int const next_stage{next_tile_idx % NUM_STAGES};

    async_copy_wait_all();
    __syncthreads();

    if (next_tile_idx < num_thread_block_tiles) {
      load_tile(next_stage, next_tile_idx);
      async_copy_commit_group();
    }

#pragma unroll
    for (int k_i{}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
#pragma unroll
      for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
           ++thread_tile_row_idx) {
        int const A_thread_block_tile_row_idx{thread_tile_row_base +
                                              thread_tile_row_idx};
        A_thread_regs[thread_tile_row_idx] =
            A_thread_block_tile[stage][A_thread_block_tile_row_idx][k_i];
      }

#pragma unroll
      for (int thread_tile_col_idx{}; thread_tile_col_idx < THREAD_TILE_SIZE_X;
           ++thread_tile_col_idx) {
        int const B_thread_block_tile_col_idx{thread_tile_col_base +
                                              thread_tile_col_idx};
        B_thread_regs[thread_tile_col_idx] =
            B_thread_block_tile[stage][k_i][B_thread_block_tile_col_idx];
      }

#pragma unroll
      for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
           ++thread_tile_row_idx) {
#pragma unroll
        for (int thread_tile_col_idx{};
             thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
          C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=
              A_thread_regs[thread_tile_row_idx] *
              B_thread_regs[thread_tile_col_idx];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
       ++thread_tile_row_idx) {
    int const C_row_idx{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                        thread_tile_row_base + thread_tile_row_idx};
#pragma unroll
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
__global__ void sgemm_async_tiling_full(int m, int n, int k, T const* A,
                                        T const* B, T const* C, T* D, T alpha,
                                        T beta) {
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);
  static_assert(BLOCK_TILE_SIZE_X % 4 == 0);
  static_assert(BLOCK_TILE_SIZE_K % 4 == 0);

  constexpr int NUM_STAGES{2};
  constexpr int NUM_THREAD_TILES_X{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X};
  constexpr int NUM_THREAD_TILES_Y{BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  constexpr int NUM_THREADS_PER_BLOCK{NUM_THREAD_TILES_X *
                                      NUM_THREAD_TILES_Y};

  int const thread_linear_idx{
      static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x)};
  int const thread_tile_row_base{thread_linear_idx / NUM_THREAD_TILES_X *
                                 THREAD_TILE_SIZE_Y};
  int const thread_tile_col_base{thread_linear_idx % NUM_THREAD_TILES_X *
                                 THREAD_TILE_SIZE_X};

  __align__(16) __shared__ T A_thread_block_tile[NUM_STAGES]
                                                   [BLOCK_TILE_SIZE_Y]
                                                   [BLOCK_TILE_SIZE_K];
  __align__(16) __shared__ T B_thread_block_tile[NUM_STAGES]
                                                   [BLOCK_TILE_SIZE_K]
                                                   [BLOCK_TILE_SIZE_X];

  int const num_thread_block_tiles{k / BLOCK_TILE_SIZE_K};

  T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X]{};
  T A_thread_regs[THREAD_TILE_SIZE_Y]{};
  T B_thread_regs[THREAD_TILE_SIZE_X]{};

  auto load_tile = [&](int stage, int thread_block_tile_idx) {
    constexpr int A_VECTOR_TILE_SIZE{BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K /
                                     4};
    constexpr int B_VECTOR_TILE_SIZE{BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X /
                                     4};

    for (int vector_offset{thread_linear_idx}; vector_offset < A_VECTOR_TILE_SIZE;
         vector_offset += NUM_THREADS_PER_BLOCK) {
      int const A_thread_block_tile_row_idx{vector_offset /
                                            (BLOCK_TILE_SIZE_K / 4)};
      int const A_thread_block_tile_col_idx{vector_offset %
                                            (BLOCK_TILE_SIZE_K / 4) * 4};
      int const A_row_idx{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                          A_thread_block_tile_row_idx};
      int const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                          A_thread_block_tile_col_idx};
      async_copy_shared_global_16(
          &A_thread_block_tile[stage][A_thread_block_tile_row_idx]
                              [A_thread_block_tile_col_idx],
          A + A_row_idx * k + A_col_idx);
    }

    for (int vector_offset{thread_linear_idx}; vector_offset < B_VECTOR_TILE_SIZE;
         vector_offset += NUM_THREADS_PER_BLOCK) {
      int const B_thread_block_tile_row_idx{vector_offset /
                                            (BLOCK_TILE_SIZE_X / 4)};
      int const B_thread_block_tile_col_idx{vector_offset %
                                            (BLOCK_TILE_SIZE_X / 4) * 4};
      int const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                          B_thread_block_tile_row_idx};
      int const B_col_idx{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X +
                          B_thread_block_tile_col_idx};
      async_copy_shared_global_16(
          &B_thread_block_tile[stage][B_thread_block_tile_row_idx]
                              [B_thread_block_tile_col_idx],
          B + B_row_idx * n + B_col_idx);
    }
  };

  load_tile(0, 0);
  async_copy_commit_group();

  for (int thread_block_tile_idx{}; thread_block_tile_idx <
                                   num_thread_block_tiles;
       ++thread_block_tile_idx) {
    int const stage{thread_block_tile_idx % NUM_STAGES};
    int const next_tile_idx{thread_block_tile_idx + 1};
    int const next_stage{next_tile_idx % NUM_STAGES};

    async_copy_wait_all();
    __syncthreads();

    if (next_tile_idx < num_thread_block_tiles) {
      load_tile(next_stage, next_tile_idx);
      async_copy_commit_group();
    }

#pragma unroll
    for (int k_i{}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
#pragma unroll
      for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
           ++thread_tile_row_idx) {
        int const A_thread_block_tile_row_idx{thread_tile_row_base +
                                              thread_tile_row_idx};
        A_thread_regs[thread_tile_row_idx] =
            A_thread_block_tile[stage][A_thread_block_tile_row_idx][k_i];
      }

#pragma unroll
      for (int thread_tile_col_idx{}; thread_tile_col_idx < THREAD_TILE_SIZE_X;
           ++thread_tile_col_idx) {
        int const B_thread_block_tile_col_idx{thread_tile_col_base +
                                              thread_tile_col_idx};
        B_thread_regs[thread_tile_col_idx] =
            B_thread_block_tile[stage][k_i][B_thread_block_tile_col_idx];
      }

#pragma unroll
      for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
           ++thread_tile_row_idx) {
#pragma unroll
        for (int thread_tile_col_idx{};
             thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
          C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=
              A_thread_regs[thread_tile_row_idx] *
              B_thread_regs[thread_tile_col_idx];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int thread_tile_row_idx{}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
       ++thread_tile_row_idx) {
    int const C_row_idx{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y +
                        thread_tile_row_base + thread_tile_row_idx};
#pragma unroll
    for (int thread_tile_col_idx{}; thread_tile_col_idx < THREAD_TILE_SIZE_X;
         ++thread_tile_col_idx) {
      int const C_col_idx{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X +
                          thread_tile_col_base + thread_tile_col_idx};
      int const C_idx{C_row_idx * n + C_col_idx};
      T const previous{(beta != T{} && C != nullptr) ? C[C_idx] : T{}};
      D[C_idx] =
          alpha * C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +
          beta * previous;
    }
  }
}

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_X,
          int THREAD_TILE_SIZE_Y>
void launch_sgemm_async_tiling_variant(int m, int n, int k, T const* A,
                                       T const* B, T const* C, T* D, T alpha,
                                       T beta, cudaStream_t stream) {
  constexpr int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X *
                                      BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  dim3 const block_dim{static_cast<unsigned int>(NUM_THREADS_PER_BLOCK), 1, 1};
  dim3 const grid_dim{
      static_cast<unsigned int>((n + BLOCK_TILE_SIZE_X - 1) /
                                BLOCK_TILE_SIZE_X),
      static_cast<unsigned int>((m + BLOCK_TILE_SIZE_Y - 1) /
                                BLOCK_TILE_SIZE_Y),
      1};

  sgemm_async_tiling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                     BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(m, n, k, A, B, C, D, alpha, beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_X,
          int THREAD_TILE_SIZE_Y>
void launch_sgemm_async_tiling_full_variant(int m, int n, int k, T const* A,
                                            T const* B, T const* C, T* D,
                                            T alpha, T beta,
                                            cudaStream_t stream) {
  constexpr int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X *
                                      BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
  dim3 const block_dim{static_cast<unsigned int>(NUM_THREADS_PER_BLOCK), 1, 1};
  dim3 const grid_dim{
      static_cast<unsigned int>(n / BLOCK_TILE_SIZE_X),
      static_cast<unsigned int>(m / BLOCK_TILE_SIZE_Y),
      1};

  sgemm_async_tiling_full<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                          BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X,
                          THREAD_TILE_SIZE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(m, n, k, A, B, C, D, alpha, beta);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_sgemm_async_tiling(int m, int n, int k, T const* A, T const* B,
                               T const* C, T* D, T alpha, T beta,
                               cudaStream_t stream) {
  int const max_mn{std::max(m, n)};
  if (max_mn <= 512) {
    SgemmProblem problem{m, n, k, alpha, beta};
    launch_sgemm_thread_tiling(problem, A, B, C, D, stream);
  } else if (m % 64 == 0 && n % 64 == 0 && k % 16 == 0) {
    launch_sgemm_async_tiling_full_variant<T, 64, 64, 16, 4, 4>(
        m, n, k, A, B, C, D, alpha, beta, stream);
  } else {
    launch_sgemm_async_tiling_variant<T, 64, 64, 16, 4, 4>(
        m, n, k, A, B, C, D, alpha, beta, stream);
  }
}

}  // namespace

void launch_sgemm_async_tiling(const SgemmProblem& problem, const float* a,
                               const float* b, const float* c, float* d,
                               cudaStream_t stream) {
  launch_sgemm_async_tiling<float>(problem.m, problem.n, problem.k, a, b, c, d,
                                   problem.alpha, problem.beta, stream);
}

}  // namespace gemm
