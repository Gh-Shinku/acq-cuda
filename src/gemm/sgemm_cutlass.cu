#define GEMM_ENABLE_CUTLASS_CHECK
#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

namespace gemm {

void launch_sgemm_cutlass(const SgemmProblem& problem, const float* a,
                          const float* b, const float* c, float* d,
                          cudaStream_t stream) {
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm = cutlass::gemm::device::Gemm<
      float, RowMajor, float, RowMajor, float, RowMajor, float,
      cutlass::arch::OpClassSimt>;

  CutlassGemm gemm;
  typename CutlassGemm::Arguments args({problem.m, problem.n, problem.k},
                                       {a, problem.k}, {b, problem.n},
                                       {c, problem.n}, {d, problem.n},
                                       {problem.alpha, problem.beta});

  GEMM_CUTLASS_CHECK(gemm(args, nullptr, stream));
}

}  // namespace gemm
