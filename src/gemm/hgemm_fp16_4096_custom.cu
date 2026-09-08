#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cuda_fp16.h>
#include <mma.h>

namespace gemm::fp16_4096 {
namespace {

// This is deliberately a simple, correct Tensor Core starting point. It is
// not intended to be competitive with cuBLAS. Replace it (and only the launch
// below) during kernel iteration; the benchmark-facing API stays unchanged.
__global__ void custom_hgemm_starter(const __half* a, const __half* b,
                                     __half* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  namespace wmma = nvcuda::wmma;
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 64;
  constexpr int kWarpM = 16;
  constexpr int kWarpN = 16;
  constexpr int kWarpK = 16;
  constexpr int kWarpsPerBlock = 4;

  const int lane_warp = static_cast<int>(threadIdx.x) / 32;
  const int warp_row = lane_warp / 2;
  const int warp_col = lane_warp % 2;
  const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
  const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
  const int tile_row = warp_row * 32;
  const int tile_col = warp_col * 32;

  using Accumulator = wmma::fragment<wmma::accumulator, kWarpM, kWarpN,
                                     kWarpK, float>;
  Accumulator c00;
  Accumulator c01;
  Accumulator c10;
  Accumulator c11;
  wmma::fill_fragment(c00, 0.0f);
  wmma::fill_fragment(c01, 0.0f);
  wmma::fill_fragment(c10, 0.0f);
  wmma::fill_fragment(c11, 0.0f);

  for (int tile_k = 0; tile_k < kDimension; tile_k += kWarpK) {
    wmma::fragment<wmma::matrix_a, kWarpM, kWarpN, kWarpK, __half,
                   wmma::row_major>
        a0;
    wmma::fragment<wmma::matrix_a, kWarpM, kWarpN, kWarpK, __half,
                   wmma::row_major>
        a1;
    wmma::fragment<wmma::matrix_b, kWarpM, kWarpN, kWarpK, __half,
                   wmma::row_major>
        b0;
    wmma::fragment<wmma::matrix_b, kWarpM, kWarpN, kWarpK, __half,
                   wmma::row_major>
        b1;

    wmma::load_matrix_sync(a0,
                           a + (block_row + tile_row) * kDimension + tile_k,
                           kDimension);
    wmma::load_matrix_sync(
        a1, a + (block_row + tile_row + kWarpM) * kDimension + tile_k,
        kDimension);
    wmma::load_matrix_sync(b0,
                           b + tile_k * kDimension + block_col + tile_col,
                           kDimension);
    wmma::load_matrix_sync(
        b1, b + tile_k * kDimension + block_col + tile_col + kWarpN,
        kDimension);
    wmma::mma_sync(c00, a0, b0, c00);
    wmma::mma_sync(c01, a0, b1, c01);
    wmma::mma_sync(c10, a1, b0, c10);
    wmma::mma_sync(c11, a1, b1, c11);
  }

  __shared__ float fp32_output[kBlockM][kBlockN];
  wmma::store_matrix_sync(&fp32_output[tile_row][tile_col], c00, kBlockN,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(&fp32_output[tile_row][tile_col + kWarpN], c01,
                          kBlockN, wmma::mem_row_major);
  wmma::store_matrix_sync(&fp32_output[tile_row + kWarpM][tile_col], c10,
                          kBlockN, wmma::mem_row_major);
  wmma::store_matrix_sync(
      &fp32_output[tile_row + kWarpM][tile_col + kWarpN], c11, kBlockN,
      wmma::mem_row_major);
  __syncthreads();

  for (int offset = static_cast<int>(threadIdx.x); offset < kBlockM * kBlockN;
       offset += kWarpsPerBlock * 32) {
    const int row = offset / kBlockN;
    const int col = offset % kBlockN;
    d[(block_row + row) * kDimension + block_col + col] =
        __float2half_rn(fp32_output[row][col]);
  }
#else
  (void)a;
  (void)b;
  (void)d;
#endif
}

}  // namespace

void launch_custom(const __half* a, const __half* b, __half* d,
                   cudaStream_t stream) {
  constexpr int kBlockTile = 64;
  static_assert(kDimension % kBlockTile == 0);
  dim3 block(128);
  dim3 grid(kDimension / kBlockTile, kDimension / kBlockTile);
  custom_hgemm_starter<<<grid, block, 0, stream>>>(a, b, d);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm::fp16_4096
