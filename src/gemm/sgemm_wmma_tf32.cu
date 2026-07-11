#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>

namespace gemm {
namespace {

template <int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K>
__global__ void sgemm_wmma_tf32_full(int m, int n, int k, const float* A,
                                     const float* B, float* D) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int WARP_SIZE{32};
  constexpr int WMMA_TILE_SIZE_M{16};
  constexpr int WMMA_TILE_SIZE_N{16};
  constexpr int WMMA_TILE_SIZE_K{8};
  static_assert(BLOCK_TILE_SIZE_X == 64);
  static_assert(BLOCK_TILE_SIZE_Y == 64);
  static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0);

  namespace wmma = nvcuda::wmma;

  int const thread_linear_idx{static_cast<int>(threadIdx.x)};
  int const warp_idx{thread_linear_idx / WARP_SIZE};
  int const warp_tile_row{warp_idx / 2};
  int const warp_tile_col{warp_idx % 2};
  int const block_row{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y};
  int const block_col{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X};
  int const warp_row{warp_tile_row * WMMA_TILE_SIZE_M * 2};
  int const warp_col{warp_tile_col * WMMA_TILE_SIZE_N * 2};

  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_00;
  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_01;
  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_10;
  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_11;
  wmma::fill_fragment(C_fragment_00, 0.0f);
  wmma::fill_fragment(C_fragment_01, 0.0f);
  wmma::fill_fragment(C_fragment_10, 0.0f);
  wmma::fill_fragment(C_fragment_11, 0.0f);

  for (int k_tile{}; k_tile < k; k_tile += BLOCK_TILE_SIZE_K) {
#pragma unroll
    for (int k_step{}; k_step < BLOCK_TILE_SIZE_K;
         k_step += WMMA_TILE_SIZE_K) {
      wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          A_fragment_0;
      wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          A_fragment_1;
      wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          B_fragment_0;
      wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          B_fragment_1;

      int const global_k{k_tile + k_step};
      wmma::load_matrix_sync(A_fragment_0,
                             A + (block_row + warp_row) * k + global_k, k);
      wmma::load_matrix_sync(
          A_fragment_1,
          A + (block_row + warp_row + WMMA_TILE_SIZE_M) * k + global_k, k);
      wmma::load_matrix_sync(B_fragment_0,
                             B + global_k * n + block_col + warp_col, n);
      wmma::load_matrix_sync(B_fragment_1,
                             B + global_k * n + block_col + warp_col +
                                 WMMA_TILE_SIZE_N,
                             n);

      wmma::mma_sync(C_fragment_00, A_fragment_0, B_fragment_0,
                     C_fragment_00);
      wmma::mma_sync(C_fragment_01, A_fragment_0, B_fragment_1,
                     C_fragment_01);
      wmma::mma_sync(C_fragment_10, A_fragment_1, B_fragment_0,
                     C_fragment_10);
      wmma::mma_sync(C_fragment_11, A_fragment_1, B_fragment_1,
                     C_fragment_11);
    }
  }

  float* output_tile{D + (block_row + warp_row) * n + block_col + warp_col};
  wmma::store_matrix_sync(output_tile, C_fragment_00, n, wmma::mem_row_major);
  wmma::store_matrix_sync(output_tile + WMMA_TILE_SIZE_N, C_fragment_01, n,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(output_tile + WMMA_TILE_SIZE_M * n, C_fragment_10, n,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(output_tile + WMMA_TILE_SIZE_M * n +
                              WMMA_TILE_SIZE_N,
                          C_fragment_11, n,
                          wmma::mem_row_major);
#else
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)B;
  (void)D;
#endif
}

template <int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K>
__global__ void sgemm_wmma_tf32_shared_full(int m, int n, int k,
                                            const float* A, const float* B,
                                            float* D) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int WARP_SIZE{32};
  constexpr int WMMA_TILE_SIZE_M{16};
  constexpr int WMMA_TILE_SIZE_N{16};
  constexpr int WMMA_TILE_SIZE_K{8};
  constexpr int WARPS_PER_BLOCK{4};
  constexpr int THREADS_PER_BLOCK{WARPS_PER_BLOCK * WARP_SIZE};
  static_assert(BLOCK_TILE_SIZE_X == 64);
  static_assert(BLOCK_TILE_SIZE_Y == 64);
  static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0);

  namespace wmma = nvcuda::wmma;

  int const thread_linear_idx{static_cast<int>(threadIdx.x)};
  int const warp_idx{thread_linear_idx / WARP_SIZE};
  int const warp_tile_row{warp_idx / 2};
  int const warp_tile_col{warp_idx % 2};
  int const block_row{static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE_Y};
  int const block_col{static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE_X};
  int const warp_row{warp_tile_row * WMMA_TILE_SIZE_M * 2};
  int const warp_col{warp_tile_col * WMMA_TILE_SIZE_N * 2};

  __shared__ float A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ float B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_00;
  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_01;
  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_10;
  wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                 WMMA_TILE_SIZE_K, float>
      C_fragment_11;
  wmma::fill_fragment(C_fragment_00, 0.0f);
  wmma::fill_fragment(C_fragment_01, 0.0f);
  wmma::fill_fragment(C_fragment_10, 0.0f);
  wmma::fill_fragment(C_fragment_11, 0.0f);

  for (int k_tile{}; k_tile < k; k_tile += BLOCK_TILE_SIZE_K) {
    for (int load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K;
         load_offset += THREADS_PER_BLOCK) {
      int const row{load_offset / BLOCK_TILE_SIZE_K};
      int const col{load_offset % BLOCK_TILE_SIZE_K};
      A_thread_block_tile[row][col] =
          A[(block_row + row) * k + k_tile + col];
    }

    for (int load_offset{thread_linear_idx};
         load_offset < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
         load_offset += THREADS_PER_BLOCK) {
      int const row{load_offset / BLOCK_TILE_SIZE_X};
      int const col{load_offset % BLOCK_TILE_SIZE_X};
      B_thread_block_tile[row][col] =
          B[(k_tile + row) * n + block_col + col];
    }
    __syncthreads();

#pragma unroll
    for (int k_step{}; k_step < BLOCK_TILE_SIZE_K;
         k_step += WMMA_TILE_SIZE_K) {
      wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          A_fragment_0;
      wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          A_fragment_1;
      wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          B_fragment_0;
      wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N,
                     WMMA_TILE_SIZE_K, wmma::precision::tf32, wmma::row_major>
          B_fragment_1;

      wmma::load_matrix_sync(A_fragment_0,
                             &A_thread_block_tile[warp_row][k_step],
                             BLOCK_TILE_SIZE_K);
      wmma::load_matrix_sync(A_fragment_1,
                             &A_thread_block_tile[warp_row + WMMA_TILE_SIZE_M]
                                                 [k_step],
                             BLOCK_TILE_SIZE_K);
      wmma::load_matrix_sync(B_fragment_0,
                             &B_thread_block_tile[k_step][warp_col],
                             BLOCK_TILE_SIZE_X);
      wmma::load_matrix_sync(B_fragment_1,
                             &B_thread_block_tile[k_step]
                                                 [warp_col + WMMA_TILE_SIZE_N],
                             BLOCK_TILE_SIZE_X);

      wmma::mma_sync(C_fragment_00, A_fragment_0, B_fragment_0,
                     C_fragment_00);
      wmma::mma_sync(C_fragment_01, A_fragment_0, B_fragment_1,
                     C_fragment_01);
      wmma::mma_sync(C_fragment_10, A_fragment_1, B_fragment_0,
                     C_fragment_10);
      wmma::mma_sync(C_fragment_11, A_fragment_1, B_fragment_1,
                     C_fragment_11);
    }
    __syncthreads();
  }

  float* output_tile{D + (block_row + warp_row) * n + block_col + warp_col};
  wmma::store_matrix_sync(output_tile, C_fragment_00, n, wmma::mem_row_major);
  wmma::store_matrix_sync(output_tile + WMMA_TILE_SIZE_N, C_fragment_01, n,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(output_tile + WMMA_TILE_SIZE_M * n, C_fragment_10, n,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(output_tile + WMMA_TILE_SIZE_M * n +
                              WMMA_TILE_SIZE_N,
                          C_fragment_11, n,
                          wmma::mem_row_major);
#else
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)B;
  (void)D;
#endif
}

bool supports_tensor_cores() {
  static thread_local bool supported = [] {
    int device = 0;
    GEMM_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    GEMM_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.major >= 8;
  }();
  return supported;
}

bool can_use_wmma_tf32_fast_path(const SgemmProblem& problem) {
  return problem.m % 64 == 0 && problem.n % 64 == 0 && problem.k % 32 == 0 &&
         problem.alpha == 1.0f && problem.beta == 0.0f;
}

}  // namespace

void launch_sgemm_wmma_tf32(const SgemmProblem& problem, const float* a,
                            const float* b, const float* c, float* d,
                            cudaStream_t stream) {
  if (!supports_tensor_cores() || !can_use_wmma_tf32_fast_path(problem)) {
    launch_sgemm_async_tiling(problem, a, b, c, d, stream);
    return;
  }

  constexpr int BLOCK_TILE_SIZE_X{64};
  constexpr int BLOCK_TILE_SIZE_Y{64};
  constexpr int BLOCK_TILE_SIZE_K{32};
  constexpr int THREADS_PER_BLOCK{128};
  dim3 const block_dim{THREADS_PER_BLOCK, 1, 1};
  dim3 const grid_dim{static_cast<unsigned int>(problem.n / BLOCK_TILE_SIZE_X),
                      static_cast<unsigned int>(problem.m / BLOCK_TILE_SIZE_Y),
                      1};

  if (std::max(problem.m, problem.n) >= 4096) {
    sgemm_wmma_tf32_shared_full<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0, stream>>>(problem.m, problem.n, problem.k,
                                             a, b, d);
  } else {
    sgemm_wmma_tf32_full<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                         BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0, stream>>>(problem.m, problem.n, problem.k,
                                             a, b, d);
  }
  GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm
