// HYBRID (11 November 2026, review retry of the A100 screen's idea 4; lab results/hybrid-gemm-20261111): a warp-level
// op on CUTLASS's 3xTF32 (OpMultiplyAddFastF32) path with the big * big product on TF32 tensor cores and the two cross
// products in ONE FP16 tensor-core instruction. Included by tf32x3_sm80.cu when built with -DFM_HYBRID_F16
// (tools/build_cutlass_tf32x3.sh: tf32x3_hyb_sm80.cubin / _sm90.cubin); the same tiles, stages and staged
// accumulation as the 3xTF32 build (the subclass keeps MathOperator = OpMultiplyAddFastF32).
//
// Per element x of A (and B): Ah = x with its low 13 significand bits cleared (TF32 big, round toward zero, as the
// 3xTF32 build), Al' = (x - Ah) * 2^11 (exact in float32). Per warp k8 step and m16n8 tile:
//   D += Ah Bh                                   one mma.m16n8k8 TF32 (exact products: Ah, Bh have 11 bits)
//   T  = [Ah | Al'] @ [Bl'; Bh]  (= 2^11 cross) one mma.m16n8k16 FP16, K = 2 x 8 (the concatenated cross products)
//   D += 2^-11 T                                 four FFMA per thread
// against 3xTF32's three TF32 mma (on sm_80 one FP16 k16 mma costs one TF32 k8 mma). The k16 slot pairing: the TF32
// k8 fragments hold k = t, t + 4 per thread (t = lane % 4); slots 2t, 2t+1 take (Ah, Al') at k = t and slots 2t+8,
// 2t+9 at k = t + 4, with B's slots (Bl', Bh) at the same k: a k permutation shared by A and B.
// Range: Ah and Al' are float16, so every |element| of A and B must stay below 32752 (|Al'| < 2 |x|); above it the
// cross term overflows to inf. Below 2^-14 the cross term loses bits (at most ~2^-36 |x| |y| per term).

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

