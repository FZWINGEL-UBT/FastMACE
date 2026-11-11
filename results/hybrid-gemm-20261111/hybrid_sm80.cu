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
#include <cutlass/gemm/warp/mma_tensor_op_fast_f32.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>

namespace cutlass {
namespace arch {
struct OpMultiplyAddHybridF16 {};
}  // namespace arch

namespace gemm {
namespace warp {

template <typename Shape_, typename LayoutA_, typename LayoutB_, typename LayoutC_, typename Policy_, int PartitionsK_,
          bool AccumulatorsInRowMajor>
class MmaTensorOpHybridF16
    : public MmaTensorOpFastF32<Shape_, float, LayoutA_, float, LayoutB_, float, LayoutC_, Policy_, PartitionsK_,
                                AccumulatorsInRowMajor> {
 public:
  using Base = MmaTensorOpFastF32<Shape_, float, LayoutA_, float, LayoutB_, float, LayoutC_, Policy_, PartitionsK_,
                                  AccumulatorsInRowMajor>;
  using typename Base::FragmentA;
  using typename Base::FragmentB;
  using typename Base::FragmentC;
  using typename Base::TransformedFragmentA;
  using typename Base::TransformedFragmentB;
  using typename Base::AccessTypeFragmentA;
  using typename Base::AccessTypeFragmentB;
  using typename Base::MmaIterations;
  using typename Base::ArchMmaOperator;
  static_assert(ArchMmaOperator::Shape::kM == 16 && ArchMmaOperator::Shape::kN == 8 && ArchMmaOperator::Shape::kK == 8,
                "hybrid: m16n8k8 TF32 instruction shape");
  static_assert(FragmentA::kElements == MmaIterations::kRow * 4 && FragmentB::kElements == MmaIterations::kColumn * 2,
                "hybrid: one k8 step per call");

  CUTLASS_DEVICE
  MmaTensorOpHybridF16() {}

  static CUTLASS_DEVICE uint32_t pack(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
  }

  /// big: Ah / Bh as TF32 (first half of the transformed fragment); second half: the packed FP16 pairs, one 32-bit
  /// word per element of the big half (A: (Ah, Al'), B: (Bl', Bh)).
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B, FragmentA const &A,
                 FragmentB const &B) const {
    uint32_t *a = reinterpret_cast<uint32_t *>(&dst_A);
    uint32_t *b = reinterpret_cast<uint32_t *>(&dst_B);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentA::kElements; ++i) {
      float const x = A[i];
      float const h = __uint_as_float(__float_as_uint(x) & 0xffffe000u);
      a[i] = __float_as_uint(h);
      a[FragmentA::kElements + i] = pack(h, (x - h) * 2048.0f);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentB::kElements; ++i) {
      float const x = B[i];
      float const h = __uint_as_float(__float_as_uint(x) & 0xffffe000u);
      b[i] = __float_as_uint(h);
      b[FragmentB::kElements + i] = pack((x - h) * 2048.0f, h);
    }
  }

  CUTLASS_DEVICE
  void operator()(FragmentC &D, TransformedFragmentA const &A, TransformedFragmentB const &B,
                  FragmentC const &C) const {
    AccessTypeFragmentA const *ptr_A = reinterpret_cast<AccessTypeFragmentA const *>(&A);
    AccessTypeFragmentB const *ptr_B = reinterpret_cast<AccessTypeFragmentB const *>(&B);
    D = C;
    this->mma_operator(D, ptr_A[0], ptr_B[0], D);  // big * big, TF32
    uint32_t const *ha = reinterpret_cast<uint32_t const *>(&A) + FragmentA::kElements;
    uint32_t const *hb = reinterpret_cast<uint32_t const *>(&B) + FragmentB::kElements;
    float *d = reinterpret_cast<float *>(&D);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {
        int const idx = AccumulatorsInRowMajor ? n + m * MmaIterations::kColumn : m + n * MmaIterations::kRow;
        float t0, t1, t2, t3;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};"
            : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
            : "r"(ha[m * 4 + 0]), "r"(ha[m * 4 + 1]), "r"(ha[m * 4 + 2]), "r"(ha[m * 4 + 3]), "r"(hb[n * 2 + 0]),
              "r"(hb[n * 2 + 1]), "f"(0.0f));
        d[idx * 4 + 0] = fmaf(t0, 4.8828125e-4f, d[idx * 4 + 0]);
        d[idx * 4 + 1] = fmaf(t1, 4.8828125e-4f, d[idx * 4 + 1]);
        d[idx * 4 + 2] = fmaf(t2, 4.8828125e-4f, d[idx * 4 + 2]);
        d[idx * 4 + 3] = fmaf(t3, 4.8828125e-4f, d[idx * 4 + 3]);
      }
    }
  }
};

template <typename WarpShape_, typename LayoutA, typename LayoutB, typename LayoutC, int PartitionsK,
          bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_, GemmShape<16, 8, 8>, float, LayoutA, float, LayoutB, float, LayoutC,
                          arch::OpMultiplyAddHybridF16, PartitionsK, AccumulatorsInRowMajor> {
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<GemmShape<16, 8, 8>, 32, cutlass::tfloat32_t, cutlass::layout::RowMajor, cutlass::tfloat32_t,
                         cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>>;
  using Type = MmaTensorOpHybridF16<WarpShape_, LayoutA, LayoutB, LayoutC, Policy, PartitionsK,
                                    AccumulatorsInRowMajor>;
};

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

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
