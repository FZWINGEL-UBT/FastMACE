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
// (docs/validation/speed/opt-T01-tensor-cores.md §5). No split-K: deterministic.
// fm_tf32x3_{rr,rc}_32x64 and _64x32 (4 November 2026): the same with a 32x64 or 64x32 tile (two warps of 32x32x16),
// grids of ceil(M / 32) x ceil(N / 64) or ceil(M / 64) x ceil(N / 32) blocks of fm_tf32x3_info[5] threads with
// fm_tf32x3_info[6..9] bytes (rr, rc for 32x64; rr, rc for 64x32): twice the blocks for the A100's 108 SMs on
// products of 5,000 atom rows and 128 columns (158 blocks of 64x64). tf32x3.TILE selects them.
// fm_tf32x3_species_rr_64x64 (4 November 2026, species_gemm.CUTLASS): the 64x64 kernel with A gathered and D scattered
// through index arrays (CUTLASS's GatherA / ScatterD), one launch over all species: the grid is species_gemm's fixed
// tile grid (ceil(rows / 64) + species) x ceil(N / 64); each block finds its species' tile from the device counts, as
// species_gemm.tile_rows, and shifts the index arrays so that CUTLASS's own tile offset lands on that tile's rows.
// Shared memory and threads: fm_tf32x3_species_info.

#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace {

template <typename LayoutB, int TileM = 64, int TileN = 64>
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    float, cutlass::layout::RowMajor, 4,
    float, LayoutB, 4,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TileM, TileN, 16>, cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 4, false,
    cutlass::arch::OpMultiplyAddFastF32>::GemmKernel;

using RowRow = GemmKernel<cutlass::layout::RowMajor>;
using RowCol = GemmKernel<cutlass::layout::ColumnMajor>;
using RowRow32x64 = GemmKernel<cutlass::layout::RowMajor, 32, 64>;
using RowCol32x64 = GemmKernel<cutlass::layout::ColumnMajor, 32, 64>;
using RowRow64x32 = GemmKernel<cutlass::layout::RowMajor, 64, 32>;
using RowCol64x32 = GemmKernel<cutlass::layout::ColumnMajor, 64, 32>;
using SpeciesRowRow = typename cutlass::gemm::kernel::DefaultGemm<
    float, cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 4, false,
    cutlass::arch::OpMultiplyAddFastF32, cutlass::gemm::SharedMemoryClearOption::kNone, true, false, true>::GemmKernel;

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

// [threads per block, dynamic shared bytes] of fm_tf32x3_species_rr_64x64
__constant__ int fm_tf32x3_species_info[2] = {SpeciesRowRow::kThreadCount,
                                              int(sizeof(typename SpeciesRowRow::SharedStorage))};

// y[scatter[r]] = a[gather[r]] @ w[s] for the rows r of each species s (rows sorted by species, counts[s] * size of
// them): a row-major with leading dimension lda (row i at a + i * lda), w [S, K, N] row-major, c row-major (ldc).
__global__ void __launch_bounds__(SpeciesRowRow::kThreadCount)
fm_tf32x3_species_rr_64x64(int species_n, int const *counts, int size, int n, int k, float const *a, int lda,
                           int const *gather, float const *w, float *c, int ldc, int const *scatter) {
  int const t = blockIdx.x;
  int species = 0, start = 0, end = 0, tiles_before = 0, rows_before = 0;
  for (int s = 0; s < species_n; ++s) {  // species_gemm.tile_rows
    int const rows = counts[s] * size, tiles = (rows + 63) / 64;
    if (t >= tiles_before && t < tiles_before + tiles) {
      species = s;
      start = rows_before + (t - tiles_before) * 64;
      end = min(start + 64, rows_before + rows);
    }
    tiles_before += tiles;
    rows_before += rows;
  }
  if (start >= end) return;
  // CUTLASS's tile offset is blockIdx.x * 64 rows: shift the index arrays by that and bound M at the tile's end
  int const shift = start - t * 64;
  extern __shared__ __align__(16) char storage[];
  cutlass::gemm::GemmCoord problem(t * 64 + (end - start), n, k);
  cutlass::gemm::GemmCoord grid(gridDim.x, gridDim.y, 1);
  typename SpeciesRowRow::Params params(
      problem, grid,
      {const_cast<float *>(a), typename SpeciesRowRow::Mma::IteratorA::Layout(lda)},
      {const_cast<float *>(w) + (long long)species * k * n, typename SpeciesRowRow::Mma::IteratorB::Layout(n)},
      {c, typename SpeciesRowRow::Epilogue::OutputTileIterator::Layout(ldc)},
      {c, typename SpeciesRowRow::Epilogue::OutputTileIterator::Layout(ldc)},
      typename SpeciesRowRow::OutputOp::Params(1.0f, 0.0f), nullptr, gather + shift, nullptr, scatter + shift);
  SpeciesRowRow op;
  op(params, *reinterpret_cast<typename SpeciesRowRow::SharedStorage *>(storage));
}

// [threads per block, dynamic shared bytes rr, dynamic shared bytes rc, tile M, tile N; the two-warp tiles' threads,
//  shared bytes rr and rc of 32x64, rr and rc of 64x32]
__constant__ int fm_tf32x3_info[10] = {RowRow::kThreadCount, int(sizeof(typename RowRow::SharedStorage)),
                                       int(sizeof(typename RowCol::SharedStorage)), 64, 64, RowRow32x64::kThreadCount,
                                       int(sizeof(typename RowRow32x64::SharedStorage)),
                                       int(sizeof(typename RowCol32x64::SharedStorage)),
                                       int(sizeof(typename RowRow64x32::SharedStorage)),
                                       int(sizeof(typename RowCol64x32::SharedStorage))};

__global__ void __launch_bounds__(RowRow::kThreadCount)
fm_tf32x3_rr_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<RowRow>(m, n, k, a, lda, b, ldb, c, ldc);
}

__global__ void __launch_bounds__(RowCol::kThreadCount)
fm_tf32x3_rc_64x64(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
  run<RowCol>(m, n, k, a, lda, b, ldb, c, ldc);
}

#define FM_TF32X3_ENTRY(name, Kernel)                                                                            \
  __global__ void __launch_bounds__(Kernel::kThreadCount)                                                         \
  name(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc) {                \
    run<Kernel>(m, n, k, a, lda, b, ldb, c, ldc);                                                                 \
  }

FM_TF32X3_ENTRY(fm_tf32x3_rr_32x64, RowRow32x64)
FM_TF32X3_ENTRY(fm_tf32x3_rc_32x64, RowCol32x64)
FM_TF32X3_ENTRY(fm_tf32x3_rr_64x32, RowRow64x32)
FM_TF32X3_ENTRY(fm_tf32x3_rc_64x32, RowCol64x32)

}  // extern "C"
