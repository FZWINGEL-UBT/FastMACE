// 3xTF32 GEMMs for compute capability 8.0 (A100): CUTLASS's OpMultiplyAddFastF32 (example 27's split, big * big +
// big * small + small * big on TF32 tensor cores, FP32 accumulation), C = A @ B, all float32.
//
// Built once, offline, into tf32x3_sm80.cubin by tools/build_cutlass_tf32x3.sh (nvcc, CUTLASS v4.8.1 headers);
// FastMACE loads the cubin through cuda.bindings (fastmace/kernels/cutlass/tf32x3.py), so neither nvcc nor CUTLASS
// is needed at run time. Each entry builds CUTLASS's kernel parameters on the device (Gemm::Params is
// CUTLASS_HOST_DEVICE) from plain pointers and leading dimensions, so the host passes only scalars.
//
// Entries (grid: ceil(M / 64) x ceil(N / 64) x 1 blocks of fm_tf32x3_info[0] threads, fm_tf32x3_info[1] (rr) or
// [2] (rc) bytes of dynamic shared memory):
//   fm_tf32x3_rr_64x64(M, N, K, A, lda, B, ldb, C, ldc): A row-major [M, K], B row-major [K, N], C row-major.
//   fm_tf32x3_rc_64x64(M, N, K, A, lda, B, ldb, C, ldc): B column-major (the transpose of a row-major [N, K]).
// Alignment 4 (128-bit loads): lda, ldb, ldc multiples of 4 and every pointer 16-byte aligned; the caller checks.
// Tile 64x64x16, warp 32x32x16, 4 stages: the winner on 83 of the 92 MACE shapes on an A100
// (docs/validation/opt-T01-tensor-cores.md §5). No split-K: deterministic.

#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace {

template <typename LayoutB>
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    float, cutlass::layout::RowMajor, 4,
    float, LayoutB, 4,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 4, false,
    cutlass::arch::OpMultiplyAddFastF32>::GemmKernel;

using RowRow = GemmKernel<cutlass::layout::RowMajor>;
using RowCol = GemmKernel<cutlass::layout::ColumnMajor>;

template <typename Kernel>
__device__ __forceinline__ void run(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c,
                                    int ldc) {
  extern __shared__ __align__(16) char storage[];
  cutlass::gemm::GemmCoord problem(m, n, k);
  cutlass::gemm::GemmCoord grid(gridDim.x, gridDim.y, 1);
  typename Kernel::Params params(
      problem, grid,
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

// [threads per block, dynamic shared bytes rr, dynamic shared bytes rc, tile M, tile N]
__constant__ int fm_tf32x3_info[5] = {RowRow::kThreadCount, int(sizeof(typename RowRow::SharedStorage)),
                                      int(sizeof(typename RowCol::SharedStorage)), 64, 64};

__global__ void __launch_bounds__(RowRow::kThreadCount)
fm_tf32x3_rr_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<RowRow>(m, n, k, a, lda, b, ldb, c, ldc);
}

__global__ void __launch_bounds__(RowCol::kThreadCount)
fm_tf32x3_rc_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<RowCol>(m, n, k, a, lda, b, ldb, c, ldc);
}

}  // extern "C"
