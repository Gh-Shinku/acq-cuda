#define GEMM_ENABLE_CUTLASS_CHECK
#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

namespace gemm {
namespace {

using RowMajor = cutlass::layout::RowMajor;
using EpilogueOutputOp =
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK>
using SimtSgemm = cutlass::gemm::device::Gemm<
    float, RowMajor, float, RowMajor, float, RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<1, 1, 1>, EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using WideSgemm = SimtSgemm<64, 128, 8, 32, 64, 8>;
using CompactSgemm = SimtSgemm<64, 64, 8, 32, 32, 8>;
using TallSgemm = SimtSgemm<128, 64, 8, 32, 64, 8>;
using LargeSgemm = SimtSgemm<128, 128, 8, 32, 64, 8>;

template <typename CutlassGemm>
void launch_cutlass_gemm(const SgemmProblem& problem, const float* a,
                         const float* b, const float* c, float* d,
                         cudaStream_t stream) {
  CutlassGemm gemm;
  typename CutlassGemm::Arguments args({problem.m, problem.n, problem.k},
                                       {a, problem.k}, {b, problem.n},
                                       {c, problem.n}, {d, problem.n},
                                       {problem.alpha, problem.beta});

  GEMM_CUTLASS_CHECK(gemm(args, nullptr, stream));
}

}  // namespace

void launch_sgemm_cutlass(const SgemmProblem& problem, const float* a,
                          const float* b, const float* c, float* d,
                          cudaStream_t stream) {
  int max_mn = problem.m > problem.n ? problem.m : problem.n;
  if (max_mn <= 64) {
    launch_cutlass_gemm<WideSgemm>(problem, a, b, c, d, stream);
  } else if (max_mn <= 768) {
    launch_cutlass_gemm<CompactSgemm>(problem, a, b, c, d, stream);
  } else if (max_mn <= 1280) {
    launch_cutlass_gemm<TallSgemm>(problem, a, b, c, d, stream);
  } else {
    launch_cutlass_gemm<LargeSgemm>(problem, a, b, c, d, stream);
  }
}

}  // namespace gemm
