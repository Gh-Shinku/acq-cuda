#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cuda_fp16.h>

namespace gemm::fp16_4096 {
namespace {

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int NUM_THREADS = 256;
constexpr int TM = 8;
constexpr int TN = 8;

// TODO: double buffering
__global__ void custom_hgemm(const __half* a,
                             const __half* b,
                             __half* d
                            ) {
    __shared__ __half A_sm[BM][BK];
    __shared__ __half B_sm[BK][BN];
    float C_reg[TM][TN] {};

    // NUM_THREADS = 256 = 128 * 2
    // (BM * BK) / NUM_THREADS = 16
    const int A_tile_row_idx = threadIdx.x / 2;
    const int A_tile_col_idx = threadIdx.x % 2;
    // NUM_THREADS = 256 = 32 * 8
    const int B_tile_row_idx = threadIdx.x / 8;
    const int B_tile_col_idx = threadIdx.x % 8;
    // NUM_THREADS = 16 * 16
    // (BM * BN) / NUM_THREADS = 64 = 8 * 8
    const int C_tile_row_idx = threadIdx.x / 16;
    const int C_tile_col_idx = threadIdx.x % 16;
    const int D_global_row_idx = blockIdx.y * BM + C_tile_row_idx * TM;
    const int D_global_col_idx = blockIdx.x * BN + C_tile_col_idx * TN;

    for (int k = 0; k < K; k += BK) {
        const int A_global_row_idx = blockIdx.y * BM + A_tile_row_idx;
        const int A_global_col_idx = k + A_tile_col_idx * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            A_sm[A_tile_row_idx][A_tile_col_idx * 16 + i] = a[A_global_row_idx * K + A_global_col_idx + i];
        }
        
        const int B_global_row_idx = k + B_tile_row_idx;
        const int B_global_col_idx = blockIdx.x * BN + B_tile_col_idx * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            B_sm[B_tile_row_idx][B_tile_col_idx * 16 + i] = b[B_global_row_idx * N + B_global_col_idx + i];
        }
        __syncthreads();

        // we've finished copying data from global memory to shared memory

        __half A_reg[TM];
        __half B_reg[TN];
        #pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                A_reg[i] = A_sm[C_tile_row_idx * TM + i][tk];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                B_reg[i] = B_sm[tk][C_tile_col_idx * TN + i];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    C_reg[i][j] += __half2float(A_reg[i]) * __half2float(B_reg[j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            d[(D_global_row_idx + i) * N + D_global_col_idx + j] = __float2half_rn(C_reg[i][j]);
        }
    }
}

}  // namespace

void launch_custom(const __half* a, const __half* b, __half* d,
                   cudaStream_t stream) {
    dim3 block {NUM_THREADS};
    dim3 grid {N / BN, M / BM};
    custom_hgemm<<<grid, block, 0, stream>>>(a, b, d);
    GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm::fp16_4096
