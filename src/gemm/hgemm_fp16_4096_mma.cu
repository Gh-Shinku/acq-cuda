#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdint>

namespace gemm::fp16_4096 {
namespace {

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

#ifndef GEMM_MMA_BM
#define GEMM_MMA_BM 128
#endif
#ifndef GEMM_MMA_BN
#define GEMM_MMA_BN 128
#endif
#ifndef GEMM_MMA_BK
#define GEMM_MMA_BK 32
#endif
#ifndef GEMM_MMA_THREADS
#define GEMM_MMA_THREADS 512
#endif
#ifndef GEMM_MMA_WARPS_M
#define GEMM_MMA_WARPS_M 4
#endif
#ifndef GEMM_MMA_WARPS_N
#define GEMM_MMA_WARPS_N 4
#endif
#ifndef GEMM_MMA_STAGES
#define GEMM_MMA_STAGES 2
#endif
// Diagnostic knob: shrink the M grid to study the wave-quantisation effect
// (a 32x32 = 1024-block grid lands just past 3*340 resident blocks, costing a
// nearly-empty fourth wave). Defaults to the full problem.
#ifndef GEMM_MMA_GRID_M
#define GEMM_MMA_GRID_M M
#endif
// Diagnostic knob: pad the dynamic shared-memory request to change the
// resident-blocks-per-SM (and therefore the wave quantisation).
#ifndef GEMM_MMA_SMEM_PAD
#define GEMM_MMA_SMEM_PAD 0
#endif


constexpr int BM = GEMM_MMA_BM;
constexpr int BN = GEMM_MMA_BN;
constexpr int BK = GEMM_MMA_BK;

// 512 threads = 16 warps arranged as WARPS_M x WARPS_N.
constexpr int NUM_THREADS = GEMM_MMA_THREADS;
constexpr int WARPS_M = GEMM_MMA_WARPS_M;
constexpr int WARPS_N = GEMM_MMA_WARPS_N;

// Per-warp tile 32 x 32 = MT x NT fragments of m16n8k16.
constexpr int WARP_M = BM / WARPS_M;
constexpr int WARP_N = BN / WARPS_N;
constexpr int MT = WARP_M / 16;
constexpr int NT = WARP_N / 8;

constexpr int NUM_STAGES = GEMM_MMA_STAGES;

// 16-byte padded strides: keeps every row 16-byte aligned for ldmatrix and
// spreads consecutive rows across distinct shared-memory banks.
constexpr int A_STRIDE = BK + 8;  // 40 halfs for BK=32, 80 bytes
constexpr int B_STRIDE = BN + 8;  // 136 halfs, 272 bytes

constexpr int CHUNKS_PER_ROW_A = BK / 8;
constexpr int CHUNKS_PER_ROW_B = BN / 8;
constexpr int A_ROWS_PER_PASS = NUM_THREADS / CHUNKS_PER_ROW_A;
constexpr int B_ROWS_PER_PASS = NUM_THREADS / CHUNKS_PER_ROW_B;
constexpr int A_PASSES = (BM * CHUNKS_PER_ROW_A) / NUM_THREADS;
constexpr int B_PASSES = (BK * CHUNKS_PER_ROW_B) / NUM_THREADS;

constexpr int A_ELEMS = NUM_STAGES * BM * A_STRIDE;
constexpr int B_ELEMS = NUM_STAGES * BK * B_STRIDE;
constexpr int SMEM_BYTES = (A_ELEMS + B_ELEMS) * sizeof(__half);

// ---------------------------------------------------------------------------
// Tensor-core primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ldmatrix taking a precomputed shared-memory address, so the address maths can
// be hoisted out of the k-loop.
__device__ __forceinline__ void ldsm_x4_at(uint32_t r[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans_at(uint32_t r[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4],
                                             const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------

__global__ void mma_hgemm(const __half* __restrict__ a,
                          const __half* __restrict__ b,
                          __half* __restrict__ d) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* const A_sm = reinterpret_cast<__half*>(smem_raw);
  __half* const B_sm = A_sm + A_ELEMS;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;

  const int a_row = tid / CHUNKS_PER_ROW_A;
  const int a_col = (tid % CHUNKS_PER_ROW_A) * 8;
  const int b_row = tid / CHUNKS_PER_ROW_B;
  const int b_col = (tid % CHUNKS_PER_ROW_B) * 8;

  const int a_grow = blockIdx.y * BM;
  const int b_gcol = blockIdx.x * BN;

  const int a_ld_row = lane % 16;        // m
  const int a_ld_col = (lane / 16) * 8;  // k
  const int b_ld_row = lane % 16;        // k
  const int b_ld_col = (lane / 16) * 8;  // n

  const int acc_row = lane / 4;  // g
  const int acc_col = lane % 4;  // t

  float acc[MT][NT][4];
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j)
      #pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  auto issue_tile = [&](int stage, int k0) {
#ifndef GEMM_MMA_NO_LOAD
    #pragma unroll
    for (int pass = 0; pass < A_PASSES; ++pass) {
      const int ar = a_row + pass * A_ROWS_PER_PASS;
      __pipeline_memcpy_async(
          &A_sm[(stage * BM + ar) * A_STRIDE + a_col],
          &a[(a_grow + ar) * K + k0 + a_col], 16);
    }
    #pragma unroll
    for (int pass = 0; pass < B_PASSES; ++pass) {
      const int br = b_row + pass * B_ROWS_PER_PASS;
      __pipeline_memcpy_async(
          &B_sm[(stage * BK + br) * B_STRIDE + b_col],
          &b[(k0 + br) * N + b_gcol + b_col], 16);
    }
#endif
  };

  constexpr int NUM_TILES = K / BK;
  constexpr int KK = BK / 16;

  // Hoist every per-thread ldmatrix address out of the k-loop. Each entry is
  // the shared-memory address for stage 0; the per-stage offset is added at
  // use time.
  uint32_t a_ldm[KK][MT];
  #pragma unroll
  for (int s = 0; s < KK; ++s) {
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
      a_ldm[s][i] = smem_addr(
          &A_sm[(warp_m * WARP_M + i * 16 + a_ld_row) * A_STRIDE + s * 16 +
                a_ld_col]);
    }
  }
  uint32_t b_ldm[KK][NT / 2];
  #pragma unroll
  for (int s = 0; s < KK; ++s) {
    #pragma unroll
    for (int h = 0; h < NT / 2; ++h) {
      b_ldm[s][h] = smem_addr(&B_sm[(s * 16 + b_ld_row) * B_STRIDE +
                                    warp_n * WARP_N + h * 16 + b_ld_col]);
    }
  }
  constexpr uint32_t A_STAGE_BYTES = BM * A_STRIDE * sizeof(__half);
  constexpr uint32_t B_STAGE_BYTES = BK * B_STRIDE * sizeof(__half);

  // Prologue: fill the pipeline with the first NUM_STAGES-1 tiles.
  #pragma unroll
  for (int s = 0; s < NUM_STAGES - 1; ++s) {
    if (s < NUM_TILES) issue_tile(s, s * BK);
    __pipeline_commit();
  }

  // Unrolling by NUM_STAGES makes `stage` (and therefore the shared-memory
  // stage offsets) a compile-time constant in each copy of the body.
  #pragma unroll 2
  for (int tile = 0; tile < NUM_TILES; ++tile) {
    // Wait for this tile, then barrier once. Issuing the next prefetch AFTER
    // the barrier guarantees every thread has finished reading the stage that
    // the prefetch is about to overwrite, so a single barrier per tile is
    // enough (instead of one before and one after the compute).
    __pipeline_wait_prior(NUM_STAGES - 2);
    __syncthreads();

    const int prefetch = tile + (NUM_STAGES - 1);
    if (prefetch < NUM_TILES) {
      issue_tile(prefetch % NUM_STAGES, prefetch * BK);
    }
    __pipeline_commit();

    const int stage = tile % NUM_STAGES;
    const uint32_t a_off = static_cast<uint32_t>(stage) * A_STAGE_BYTES;
    const uint32_t b_off = static_cast<uint32_t>(stage) * B_STAGE_BYTES;

    #pragma unroll
    for (int s = 0; s < KK; ++s) {
      uint32_t a_frag[MT][4];
      #pragma unroll
      for (int i = 0; i < MT; ++i) {
        ldsm_x4_at(a_frag[i], a_ldm[s][i] + a_off);
      }
      uint32_t b_frag[NT][2];
      #pragma unroll
      for (int h = 0; h < NT / 2; ++h) {
        uint32_t r[4];
        ldsm_x4_trans_at(r, b_ldm[s][h] + b_off);
        b_frag[h * 2 + 0][0] = r[0];
        b_frag[h * 2 + 0][1] = r[1];
        b_frag[h * 2 + 1][0] = r[2];
        b_frag[h * 2 + 1][1] = r[3];
      }
      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
          mma_m16n8k16(acc[i][j], a_frag[i], b_frag[j]);
    }
  }

  // Epilogue: the m16n8k16 accumulator layout is fixed and known, so the fp32
  // results are converted and stored straight from registers.
  #pragma unroll
  for (int i = 0; i < MT; ++i) {
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int row0 = a_grow + warp_m * WARP_M + i * 16 + acc_row;
      const int col = b_gcol + warp_n * WARP_N + j * 8 + acc_col * 2;
      const __half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
      const __half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
      *reinterpret_cast<__half2*>(&d[row0 * N + col]) = lo;
      *reinterpret_cast<__half2*>(&d[(row0 + 8) * N + col]) = hi;
    }
  }
}

}  // namespace

void launch_mma(const __half* a, const __half* b, __half* d,
                cudaStream_t stream) {
  constexpr int LAUNCH_SMEM =
      SMEM_BYTES > GEMM_MMA_SMEM_PAD ? SMEM_BYTES : GEMM_MMA_SMEM_PAD;
  static bool configured = false;
  if (!configured) {
    GEMM_CUDA_CHECK(cudaFuncSetAttribute(
        mma_hgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, LAUNCH_SMEM));
    configured = true;
  }
  dim3 block {NUM_THREADS};
  dim3 grid {N / BN, GEMM_MMA_GRID_M / BM};
  mma_hgemm<<<grid, block, LAUNCH_SMEM, stream>>>(a, b, d);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm::fp16_4096
