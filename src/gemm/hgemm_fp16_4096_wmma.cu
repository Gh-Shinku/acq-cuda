#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace gemm::fp16_4096 {
namespace {

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// 256 threads = 8 warps arranged as WARPS_M x WARPS_N.
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;

// Per-warp tile: 64 x 32 = (4 x 2) fragments of 16 x 16 x 16.
constexpr int WARP_M = BM / WARPS_M;
constexpr int WARP_N = BN / WARPS_N;
constexpr int M_TILES = WARP_M / 16;
constexpr int N_TILES = WARP_N / 16;

constexpr int NUM_BUFFERS = 2;

// Leading dimensions are padded to a multiple of 8 halfs (16 bytes) so that
// every row start stays 16-byte aligned for wmma::load_matrix_sync and so that
// consecutive rows land on different shared-memory banks.
constexpr int A_STRIDE = BK + 8;  // 40 halfs, 80 bytes
constexpr int B_STRIDE = BN + 8;  // 136 halfs, 272 bytes

// Each thread copies two 16-byte (8 half) chunks per tile and per matrix.
constexpr int A_CHUNK_ROWS = BM / 2;  // 64 rows per pass

using namespace nvcuda;

__global__ void wmma_hgemm(const __half* __restrict__ a,
                           const __half* __restrict__ b,
                           __half* __restrict__ d) {
  __shared__ __align__(16) __half A_sm[NUM_BUFFERS][BM][A_STRIDE];
  __shared__ __align__(16) __half B_sm[NUM_BUFFERS][BK][B_STRIDE];
  // Per-warp fp32 staging for the float -> half epilogue.
  __shared__ __align__(16) float C_stage[NUM_WARPS][16 * 16];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;

  // Global -> shared copy coordinates (two 16-byte chunks per thread).
  const int a_row = tid / 4;         // 0 .. 63
  const int a_col = (tid % 4) * 8;   // 0, 8, 16, 24
  const int b_row = tid / 16;        // 0 .. 15
  const int b_col = (tid % 16) * 8;  // 0 .. 120

  const int a_grow = blockIdx.y * BM;
  const int b_gcol = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[M_TILES][N_TILES];
  #pragma unroll
  for (int i = 0; i < M_TILES; ++i) {
    #pragma unroll
    for (int j = 0; j < N_TILES; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  auto issue_tile = [&](int buffer, int k0) {
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int ar = a_row + pass * A_CHUNK_ROWS;
      __pipeline_memcpy_async(&A_sm[buffer][ar][a_col],
                              &a[(a_grow + ar) * K + k0 + a_col], 16);
      const int br = b_row + pass * (BK / 2);
      __pipeline_memcpy_async(&B_sm[buffer][br][b_col],
                              &b[(k0 + br) * N + b_gcol + b_col], 16);
    }
  };

  issue_tile(0, 0);
  __pipeline_commit();

  int buffer = 0;
  for (int k = 0; k < K; k += BK, buffer ^= 1) {
    const int next_k = k + BK;
    const bool has_next = next_k < K;
    if (has_next) {
      issue_tile(buffer ^ 1, next_k);
      __pipeline_commit();
    }
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    // Tensor-core inner loop over the two 16-wide K slices of the tile.
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>
          b_frag[N_TILES];
      #pragma unroll
      for (int j = 0; j < N_TILES; ++j) {
        wmma::load_matrix_sync(
            b_frag[j], &B_sm[buffer][kk][warp_n * WARP_N + j * 16], B_STRIDE);
      }
      #pragma unroll
      for (int i = 0; i < M_TILES; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
            a_frag;
        wmma::load_matrix_sync(
            a_frag, &A_sm[buffer][warp_m * WARP_M + i * 16][kk], A_STRIDE);
        #pragma unroll
        for (int j = 0; j < N_TILES; ++j) {
          wmma::mma_sync(acc[i][j], a_frag, b_frag[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

  // Epilogue: the fp32 fragment layout is opaque, so round-trip each 16x16
  // tile through shared memory to convert it to half before the global store.
  #pragma unroll
  for (int i = 0; i < M_TILES; ++i) {
    #pragma unroll
    for (int j = 0; j < N_TILES; ++j) {
      wmma::store_matrix_sync(C_stage[warp], acc[i][j], 16,
                              wmma::mem_row_major);
      __syncwarp();
      const int r0 = a_grow + warp_m * WARP_M + i * 16;
      const int c0 = b_gcol + warp_n * WARP_N + j * 16;
      #pragma unroll
      for (int e = lane; e < 16 * 16; e += 32) {
        const int r = e / 16;
        const int c = e % 16;
        d[(r0 + r) * N + c0 + c] = __float2half_rn(C_stage[warp][e]);
      }
      __syncwarp();
    }
  }
}

}  // namespace

void launch_wmma(const __half* a, const __half* b, __half* d,
                 cudaStream_t stream) {
  dim3 block {NUM_THREADS};
  dim3 grid {N / BN, M / BM};
  wmma_hgemm<<<grid, block, 0, stream>>>(a, b, d);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm::fp16_4096
