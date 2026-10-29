// CUTLASS Sm80 tensor-core FP32 GEMMs for the tensor-core GEMM screen (29 October 2026), as a C ABI for ctypes.
//
// C[b] = A[b] @ B[b] for b < batch, float32 in and out, A and C row-major with leading dimensions lda, ldc and
// batch strides; B row- or column-major. The batch form covers segment_gemm's joined layout directly: product i's
// row (atom, m) of A sits at atom * a_n + m * a_m, so with batch = m, lda = a_n and batch stride a_m it is a strided
// batched GEMM over the components m, and the output rows likewise. GemmUniversal in kBatched mode needs no
// workspace, so a call is one kernel launch on the given stream (capturable in a CUDA graph).
//
// Compiled once per variant (-DFM_VARIANT): 0 = CUTLASS's 3xTF32 (OpMultiplyAddFastF32: big rounded toward zero,
// small half-ulp truncated, small*big + big*small + big*big), 1 = round-to-nearest split, three products, 2 = the
// same plus small*small, 3 = plain 1xTF32 (rate reference only, not FP32-accurate).
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "fast_f32_variants.h"

using cutlass::gemm::GemmShape;

#if FM_VARIANT == 0
using Operator = cutlass::arch::OpMultiplyAddFastF32;
#elif FM_VARIANT == 1
using Operator = cutlass::arch::OpMultiplyAddFastF32Rn3;
#elif FM_VARIANT == 2
using Operator = cutlass::arch::OpMultiplyAddFastF32Rn4;
#else
using Operator = cutlass::arch::OpMultiplyAdd;
#endif

// Tile 0: 64x64x16, warps 32x32x16, 4 stages; tile 1: 64x128x16, warps 32x64x16, 4 stages (the two tiles that won
// all 92 shapes of opt-T01 §5 between them).
template <int T> struct Tile;
template <> struct Tile<0> { using TB = GemmShape<64, 64, 16>; using W = GemmShape<32, 32, 16>; };
template <> struct Tile<1> { using TB = GemmShape<64, 128, 16>; using W = GemmShape<32, 64, 16>; };

template <int T, class LayoutB, int Align>
using Gemm = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor, float, LayoutB, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Tile<T>::TB, typename Tile<T>::W, GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, Align, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, Align, Align, Operator>;

template <int T, class LayoutB, int Align>
static int run(int m, int n, int k, int batch, const float* A, long long lda, long long sA, const float* B,
               long long ldb, long long sB, float* C, long long ldc, long long sC, cudaStream_t stream) {
  using G = Gemm<T, LayoutB, Align>;
  typename G::Arguments args(cutlass::gemm::GemmUniversalMode::kBatched, {m, n, k}, batch, {1.0f, 0.0f}, A, B, C, C,
                             sA, sB, sC, sC, lda, ldb, ldc, ldc);
  G op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return 2;
  if (op.run(stream) != cutlass::Status::kSuccess) return 3;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

template <int T, class LayoutB>
static int dispatch_align(int align, int m, int n, int k, int batch, const float* A, long long lda, long long sA,
                          const float* B, long long ldb, long long sB, float* C, long long ldc, long long sC,
                          cudaStream_t s) {
  if (align == 4) return run<T, LayoutB, 4>(m, n, k, batch, A, lda, sA, B, ldb, sB, C, ldc, sC, s);
  return run<T, LayoutB, 1>(m, n, k, batch, A, lda, sA, B, ldb, sB, C, ldc, sC, s);
}

#define FM_CAT2(a, b) a##b
#define FM_CAT(a, b) FM_CAT2(a, b)

// Returns 0 on success; 1 if CUTLASS refuses the problem at this alignment (the caller retries with align 1);
// >1 on a launch failure; 9 for an unknown tile.
extern "C" int FM_CAT(fm_tc_gemm_v, FM_VARIANT)(int tile, int b_col_major, int align, int m, int n, int k, int batch,
                                                const float* A, long long lda, long long sA, const float* B,
                                                long long ldb, long long sB, float* C, long long ldc, long long sC,
                                                void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  if (tile == 0)
    return b_col_major ? dispatch_align<0, Col>(align, m, n, k, batch, A, lda, sA, B, ldb, sB, C, ldc, sC, s)
                       : dispatch_align<0, Row>(align, m, n, k, batch, A, lda, sA, B, ldb, sB, C, ldc, sC, s);
  if (tile == 1)
    return b_col_major ? dispatch_align<1, Col>(align, m, n, k, batch, A, lda, sA, B, ldb, sB, C, ldc, sC, s)
                       : dispatch_align<1, Row>(align, m, n, k, batch, A, lda, sA, B, ldb, sB, C, ldc, sC, s);
  return 9;
}
