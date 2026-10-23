// CUTLASS 2.x device::Gemm runners for the OPT-T01 3xTF32 screen (Sm80 tensor ops, float in/out).
#pragma once
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

struct Prob {
  int m, n, k;
  const float* A; int lda;
  const float* B; int ldb;
  float* C; int ldc;  // C is row-major m x n
};

struct Runner {
  virtual ~Runner() {}
  // false: can_implement or initialize refused the problem (alignment, ...)
  virtual bool init(const Prob& p) = 0;
  virtual bool run(cudaStream_t s) = 0;
};

template <class G>
struct CutlassRunner : Runner {
  G op;
  typename G::Arguments args;
  bool init(const Prob& p) override {
    args = typename G::Arguments({p.m, p.n, p.k}, {p.A, p.lda}, {p.B, p.ldb}, {p.C, p.ldc}, {p.C, p.ldc},
                                 {1.0f, 0.0f});
    if (op.can_implement(args) != cutlass::Status::kSuccess) return false;
    return op.initialize(args) == cutlass::Status::kSuccess;
  }
  bool run(cudaStream_t s) override { return op.run(s) == cutlass::Status::kSuccess; }
};

using cutlass::gemm::GemmShape;

// Tile configs. T0 = CUTLASS example 27 (128x128x16, warp 64x64x16, 3 stages).
template <int T> struct Tile;
template <> struct Tile<0> { using TB = GemmShape<128, 128, 16>; using W = GemmShape<64, 64, 16>; static constexpr int S = 3; };
template <> struct Tile<1> { using TB = GemmShape<64, 64, 16>;   using W = GemmShape<32, 32, 16>; static constexpr int S = 4; };
template <> struct Tile<2> { using TB = GemmShape<64, 128, 16>;  using W = GemmShape<32, 64, 16>; static constexpr int S = 4; };
static const char* kTileName[3] = {"128x128x16_w64x64x16_s3", "64x64x16_w32x32x16_s4", "64x128x16_w32x64x16_s4"};

template <class LA, class LB, int AlA, int AlB, int AlC, int T, class Op>
using GemmT = cutlass::gemm::device::Gemm<
    float, LA, float, LB, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    typename Tile<T>::TB, typename Tile<T>::W, GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, AlC, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Tile<T>::S, AlA, AlB, false, Op>;

// fast: OpMultiplyAddFastF32 (3xTF32); else OpMultiplyAdd (1xTF32). A TU compiled with -DCUTLASS_BENCH_STUB (its
// real build failed) returns nullptr, which the bench records as "no_instantiation".
template <class LA, class LB, int AlA, int AlB, int AlC, int T>
Runner* make_runner(bool fast) {
#ifdef CUTLASS_BENCH_STUB
  (void)fast; return nullptr;
#else
  if (fast) return new CutlassRunner<GemmT<LA, LB, AlA, AlB, AlC, T, cutlass::arch::OpMultiplyAddFastF32>>();
  return new CutlassRunner<GemmT<LA, LB, AlA, AlB, AlC, T, cutlass::arch::OpMultiplyAdd>>();
#endif
}
