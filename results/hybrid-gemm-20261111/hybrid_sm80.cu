// Hybrid GEMM lab (review retry, 11 November 2026): C = A @ B in float32 with the big * big product on TF32 tensor
// cores and the two cross products in ONE FP16 tensor-core instruction, inside CUTLASS's sm_80 kernel that
// production's 3xTF32 uses (fastmace/kernels/cutlass/tf32x3_sm80.cu: same tiles, stages, staged accumulation).
//
// Per element x of A (and B): Ah = x with its low 13 significand bits cleared (TF32 big, round toward zero, as
// production's 3xTF32), Al' = (x - Ah) * 2^11 (exact in float32). Per warp k8 step and m16n8 tile:
//   D += Ah Bh                                   one mma.m16n8k8 TF32 (exact products: Ah, Bh have 11 bits)
//   T  = [Ah | Al'] @ [Bl'; Bh]  (= 2^11 cross) one mma.m16n8k16 FP16, K = 2 x 8 (concatenated cross products)
//   D += 2^-11 T                                 four FFMA per thread
// against 3xTF32's three TF32 mma (A100: one FP16 k16 mma costs one TF32 k8 mma). The k16 slot pairing: the TF32 k8
// fragments hold k = t, t + 4 per thread (t = lane % 4); slots 2t, 2t+1 take (Ah, Al') at k = t and slots 2t+8, 2t+9
// at k = t + 4, with B's slots (Bl', Bh) at the same k, so every slot multiplies matching k (a k permutation shared
// by A and B). Range: Ah and Al' go to float16, so |x| must stay below 65504; below 2^-14 the cross term loses bits
// (an error relative to |A| |B| of at most ~2^-36 per term). Not part of the library.
//
// Entries as fm_tf32x3_{rr,rc}_64x64 (same arguments, grid and shared memory: fm_hyb_info).

#include <cutlass/cutlass.h>
#include "../../fastmace/kernels/cutlass/hybrid_f16.h"

#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace {

template <typename LayoutB, typename Op>
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    float, cutlass::layout::RowMajor, 4, float, LayoutB, 4, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 4, false, Op>::GemmKernel;

using HybRR = GemmKernel<cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddHybridF16>;
using HybRC = GemmKernel<cutlass::layout::ColumnMajor, cutlass::arch::OpMultiplyAddHybridF16>;
using RefRR = GemmKernel<cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddFastF32>;  // same build, 3xTF32

template <typename Kernel>
__device__ __forceinline__ void run(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c,
                                    int ldc) {
  extern __shared__ __align__(16) char storage[];
  cutlass::gemm::GemmCoord problem(m, n, k);
  cutlass::gemm::GemmCoord grid(gridDim.x, gridDim.y, 1);
  typename Kernel::Params params(problem, grid,
                                 {const_cast<float *>(a), typename Kernel::Mma::IteratorA::Layout(lda)},
                                 {const_cast<float *>(b), typename Kernel::Mma::IteratorB::Layout(ldb)},
                                 {c, typename Kernel::Epilogue::OutputTileIterator::Layout(ldc)},
                                 {c, typename Kernel::Epilogue::OutputTileIterator::Layout(ldc)},
                                 typename Kernel::OutputOp::Params(1.0f, 0.0f));
  Kernel op;
  op(params, *reinterpret_cast<typename Kernel::SharedStorage *>(storage));
}

}  // namespace

extern "C" {

// [threads, shared bytes rr, shared bytes rc, shared bytes of the 3xTF32 reference]
__constant__ int fm_hyb_info[4] = {HybRR::kThreadCount, int(sizeof(typename HybRR::SharedStorage)),
                                   int(sizeof(typename HybRC::SharedStorage)),
                                   int(sizeof(typename RefRR::SharedStorage))};

__global__ void __launch_bounds__(HybRR::kThreadCount)
fm_hyb_rr_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<HybRR>(m, n, k, a, lda, b, ldb, c, ldc);
}

__global__ void __launch_bounds__(HybRC::kThreadCount)
fm_hyb_rc_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<HybRC>(m, n, k, a, lda, b, ldb, c, ldc);
}

__global__ void __launch_bounds__(RefRR::kThreadCount)
fm_ref_rr_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<RefRR>(m, n, k, a, lda, b, ldb, c, ldc);
}

}  // extern "C"
