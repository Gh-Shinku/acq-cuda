#include "gemm/check.hpp"
#include "gemm/hgemm_fp16_4096.hpp"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdint>

// Stream-K scheduling for the fixed 4096^3 FP16 workload.
//
// The plain tiled kernel launches (M/BM)*(N/BN) = 1024 blocks. With two
// resident blocks per SM that is 3.01 "waves" of 340, so the last wave holds
// only four blocks and costs a full round of latency (measured: 0.698 ms vs
// 0.604 ms for 992 blocks, a 13% cliff).
//
// Stream-K decouples the work from the output tiling: the (output tile,
// k-slice) space is flattened into 1024*64 = 65536 units and split evenly
// across a fixed number of resident blocks. Every block therefore does the
// same amount of work and the tail disappears.
//
// A block's contiguous unit range touches at most a handful of output tiles.
// A tile that lies entirely inside a range is written straight to `d`; a tile
// split between two adjacent blocks is accumulated in fp32: the block that
// owns the tile's first k-slice stashes its partial and sets a flag, and the
// block that owns the tile's last k-slice waits for the flag, adds it, and
// writes the final FP16 result.

namespace gemm::fp16_4096 {
namespace {

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;

constexpr int NUM_THREADS = 512;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int WARP_M = BM / WARPS_M;
constexpr int WARP_N = BN / WARPS_N;
constexpr int MT = WARP_M / 16;
constexpr int NT = WARP_N / 8;

// A 16-half row skew keeps ldmatrix rows aligned and changes the shared-memory
// bank pattern. It also gave the best measured throughput for this tile shape.
constexpr int A_STRIDE = BK + 16;
constexpr int B_STRIDE = BN + 16;

constexpr int CHUNKS_PER_ROW_A = BK / 8;
constexpr int CHUNKS_PER_ROW_B = BN / 8;
constexpr int A_ROWS_PER_PASS = NUM_THREADS / CHUNKS_PER_ROW_A;
constexpr int B_ROWS_PER_PASS = NUM_THREADS / CHUNKS_PER_ROW_B;
constexpr int A_PASSES = (BM * CHUNKS_PER_ROW_A) / NUM_THREADS;
constexpr int B_PASSES = (BK * CHUNKS_PER_ROW_B) / NUM_THREADS;

constexpr int NUM_STAGES = 2;
constexpr int A_ELEMS = NUM_STAGES * BM * A_STRIDE;
constexpr int B_ELEMS = NUM_STAGES * BK * B_STRIDE;
constexpr int SMEM_BYTES = (A_ELEMS + B_ELEMS) * sizeof(__half);

// Override for the number of resident stream-K blocks (0 = auto). Must not
// exceed the number of co-resident blocks, otherwise the split-tile hand-off
// can deadlock.
#ifndef GEMM_STREAMK_BLOCKS
#define GEMM_STREAMK_BLOCKS 0
#endif

constexpr int TILE_M = M / BM;
constexpr int TILE_N = N / BN;
constexpr int NUM_TILES = TILE_M * TILE_N;
constexpr int K_TILES = K / BK;  // k-slices per output tile
constexpr int TOTAL_UNITS = NUM_TILES * K_TILES;

// ---------------------------------------------------------------------------
// Tensor-core primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

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

__global__ void streamk_hgemm(const __half* __restrict__ a,
                              const __half* __restrict__ b,
                              __half* __restrict__ d,
                              float* __restrict__ partial,
                              int* __restrict__ flag, int num_blocks,
                              int generation) {
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

  const int a_ld_row = lane % 16;
  const int a_ld_col = (lane / 16) * 8;
  const int b_ld_row = lane % 16;
  const int b_ld_col = (lane / 16) * 8;
  const int acc_row = lane / 4;
  const int acc_col = lane % 4;

  constexpr int KK = BK / 16;
  constexpr uint32_t A_STAGE_BYTES = BM * A_STRIDE * sizeof(__half);
  constexpr uint32_t B_STAGE_BYTES = BK * B_STRIDE * sizeof(__half);

  // This block's contiguous slice of the flattened (tile, k-slice) space.
  const long start = static_cast<long>(blockIdx.x) * TOTAL_UNITS / num_blocks;
  const long end = static_cast<long>(blockIdx.x + 1) * TOTAL_UNITS / num_blocks;

  for (long unit = start; unit < end;) {
    const int tile = static_cast<int>(unit / K_TILES);
    const long tile_start = static_cast<long>(tile) * K_TILES;
    const long tile_end = tile_start + K_TILES;
    const long seg_start = unit > tile_start ? unit : tile_start;
    const long seg_end = end < tile_end ? end : tile_end;
    const int ks0 = static_cast<int>(seg_start - tile_start);
    const int nks = static_cast<int>(seg_end - seg_start);

    const int tile_m = tile / TILE_N;
    const int tile_n = tile % TILE_N;
    const int a_grow = tile_m * BM;
    const int b_gcol = tile_n * BN;

    // Hoisted ldmatrix addresses for this tile.
    uint32_t a_ldm[KK][MT];
    uint32_t b_ldm[KK][NT / 2];
    #pragma unroll
    for (int s = 0; s < KK; ++s) {
      #pragma unroll
      for (int i = 0; i < MT; ++i) {
        a_ldm[s][i] = smem_addr(
            &A_sm[(warp_m * WARP_M + i * 16 + a_ld_row) * A_STRIDE + s * 16 +
                  a_ld_col]);
      }
      #pragma unroll
      for (int h = 0; h < NT / 2; ++h) {
        b_ldm[s][h] = smem_addr(&B_sm[(s * 16 + b_ld_row) * B_STRIDE +
                                      warp_n * WARP_N + h * 16 + b_ld_col]);
      }
    }

    float acc[MT][NT][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
      #pragma unroll
      for (int j = 0; j < NT; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    auto issue_slice = [&](int stage, int ks) {
      const int k0 = ks * BK;
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
    };

    issue_slice(0, ks0);
    __pipeline_commit();

    for (int i = 0; i < nks; ++i) {
      // Wait for this slice, then barrier: the barrier both makes every
      // thread's cp.async writes visible and certifies that no warp is still
      // reading the stage the prefetch below overwrites.
      __pipeline_wait_prior(0);
      __syncthreads();
      if (i + 1 < nks) {
        issue_slice((i + 1) % NUM_STAGES, ks0 + i + 1);
      }
      __pipeline_commit();

      const uint32_t a_off =
          static_cast<uint32_t>(i % NUM_STAGES) * A_STAGE_BYTES;
      const uint32_t b_off =
          static_cast<uint32_t>(i % NUM_STAGES) * B_STAGE_BYTES;
      #pragma unroll
      for (int s = 0; s < KK; ++s) {
        uint32_t a_frag[MT][4];
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          ldsm_x4_at(a_frag[mi], a_ldm[s][mi] + a_off);
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
        for (int mi = 0; mi < MT; ++mi)
          #pragma unroll
          for (int nj = 0; nj < NT; ++nj)
            mma_m16n8k16(acc[mi][nj], a_frag[mi], b_frag[nj]);
      }
    }

    // Make sure every warp is done reading the staging buffers before either
    // the next tile's issue or the reduction below touches shared memory.
    __syncthreads();

    // A block's range owns a tile's head when it starts at the tile boundary
    // and its tail when it ends at the tile boundary. The head is processed at
    // the very end of the neighbouring (lower-index) block's range, while the
    // tail is processed at the very start of this block's range. So the
    // tail-owner finishes first and must stash its partial + flag; the
    // head-owner arrives later and does the reduction. Doing it the other way
    // round makes the tail-owner spin for a whole tile's worth of work.
    const bool owns_head = (ks0 == 0);
    const bool owns_tail = (ks0 + nks == K_TILES);

    if (owns_head && owns_tail) {
      // Sole owner: write the FP16 result straight out.
      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int r0 = a_grow + warp_m * WARP_M + i * 16 + acc_row;
          const int c = b_gcol + warp_n * WARP_N + j * 8 + acc_col * 2;
          *reinterpret_cast<__half2*>(&d[r0 * N + c]) =
              __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
          *reinterpret_cast<__half2*>(&d[(r0 + 8) * N + c]) =
              __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
        }
    } else if (owns_tail) {
      // Early finisher: stash the fp32 partial and hand off.
      float* const base =
          partial + static_cast<size_t>(tile) * BM * BN;
      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int r0 = warp_m * WARP_M + i * 16 + acc_row;
          const int c = warp_n * WARP_N + j * 8 + acc_col * 2;
          *reinterpret_cast<float2*>(&base[r0 * BN + c]) =
              make_float2(acc[i][j][0], acc[i][j][1]);
          *reinterpret_cast<float2*>(&base[(r0 + 8) * BN + c]) =
              make_float2(acc[i][j][2], acc[i][j][3]);
        }
      __syncthreads();
      __threadfence();
      if (tid == 0) {
        atomicExch(&flag[tile], generation);
      }
    } else {
      // Late finisher: wait for the stashed partial, add it, then store. The
      // flag carries a per-launch generation so no reset between launches is
      // needed.
      if (tid == 0) {
        while (atomicAdd(&flag[tile], 0) != generation) {
        }
      }
      __syncthreads();
      __threadfence();
      const float* const base =
          partial + static_cast<size_t>(tile) * BM * BN;
      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int r0 = warp_m * WARP_M + i * 16 + acc_row;
          const int c = warp_n * WARP_N + j * 8 + acc_col * 2;
          const float2 p0 = *reinterpret_cast<const float2*>(&base[r0 * BN + c]);
          const float2 p1 =
              *reinterpret_cast<const float2*>(&base[(r0 + 8) * BN + c]);
          acc[i][j][0] += p0.x;
          acc[i][j][1] += p0.y;
          acc[i][j][2] += p1.x;
          acc[i][j][3] += p1.y;
        }
      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int r0 = a_grow + warp_m * WARP_M + i * 16 + acc_row;
          const int c = b_gcol + warp_n * WARP_N + j * 8 + acc_col * 2;
          *reinterpret_cast<__half2*>(&d[r0 * N + c]) =
              __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
          *reinterpret_cast<__half2*>(&d[(r0 + 8) * N + c]) =
              __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
        }
    }

    unit = seg_end;
  }
}

}  // namespace

void launch_streamk(const __half* a, const __half* b, __half* d,
                    cudaStream_t stream) {
  static float* partial = nullptr;
  static int* flag = nullptr;
  static int num_blocks = 0;
  static int generation = 0;

  if (partial == nullptr) {
    GEMM_CUDA_CHECK(
        cudaMalloc(&partial, static_cast<size_t>(NUM_TILES) * BM * BN *
                                sizeof(float)));
    GEMM_CUDA_CHECK(cudaMalloc(&flag, NUM_TILES * sizeof(int)));
    GEMM_CUDA_CHECK(cudaMemset(flag, 0, NUM_TILES * sizeof(int)));
    int device = 0;
    GEMM_CUDA_CHECK(cudaGetDevice(&device));
    int sm_count = 0;
    GEMM_CUDA_CHECK(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                               device));
    int blocks_per_sm = 0;
    GEMM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, streamk_hgemm, NUM_THREADS, SMEM_BYTES));
    if (blocks_per_sm < 1) {
      blocks_per_sm = 1;
    }
    num_blocks = sm_count * blocks_per_sm;
#if GEMM_STREAMK_BLOCKS > 0
    num_blocks = GEMM_STREAMK_BLOCKS;
#endif
    GEMM_CUDA_CHECK(cudaFuncSetAttribute(
        streamk_hgemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES));
  }

  // The hand-off flags are never reset between launches; instead each launch
  // uses a fresh generation value.
  ++generation;
  if (generation == 0) {
    GEMM_CUDA_CHECK(cudaMemsetAsync(flag, 0, NUM_TILES * sizeof(int), stream));
    generation = 1;
  }

  streamk_hgemm<<<num_blocks, NUM_THREADS, SMEM_BYTES, stream>>>(
      a, b, d, partial, flag, num_blocks, generation);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm::fp16_4096
