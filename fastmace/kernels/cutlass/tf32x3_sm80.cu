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

// ---------------------------------------------------------------------------------------------------------------
// Idea 31 (6 November 2026; fastmace.kernels.cutlass.epilogue): the 64x64 kernels with an epilogue that stores row r
// of the product at  c + (r / rows_per_atom) * atom_stride + (r % rows_per_atom) * row_stride  (an atom's (2l+1)
// rows into its slice of a wider [atoms, width] tensor: the ir_mul cat that join_rows builds) and, if density is
// not null, divides each element by (density[r / rows_per_atom] + 1) on the way: the "x / (density + 1)" of the
// interaction, with Inductor's instructions (add.f32 for density + 1, then div.full.f32; idea31-glue.md, step 1).
// The mainloop is RowRow's / RowCol's own (same Mma type, same K iterations), and alpha 1 / beta 0 as mm_into, so
// each stored value is mm_into's accumulator, divided as the glue kernel divides it: bitwise the same.
//   fm_tf32x3_remap_div_{rr,rc}_64x64(M, N, K, A, lda, B, ldb, C, rows_per_atom, atom_stride, row_stride, density):
//   grid ceil(M / 64) x ceil(N / 64), fm_tf32x3_remap_info = [threads, shared bytes rr, shared bytes rc].
// N, atom_stride and row_stride multiples of 4 and C 16-byte aligned (128-bit stores); the caller checks.
// fm_tf32x3_remap_select_{rr,rc}_64x64(..., density, kind, select_columns) (idea 31, skip_select): the same, B holding
// one block of select_columns columns per species side by side ([K, species * select_columns]); column j (species
// j / select_columns) of row r is stored only if kind[r / rows_per_atom] is that species, at column
// j % select_columns of the remapped row: the species skip's own-species selection, each element stored once from
// its own block's accumulator. select_columns a multiple of 4 (a 128-bit access never straddles two species).

namespace {

__device__ __forceinline__ float div_full(float a, float b) {
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

// The output tile iterator of the epilogue: PredicatedTileIterator's thread map and row walk, with the row remap and
// the division in store. load is never used (beta = 0: no source).
template <typename ThreadMap_, bool Select = false>
class RemapDivIterator {
 public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;
  using Element = float;
  using Layout = cutlass::layout::RowMajor;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;
  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
                                               ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                                               ThreadMap::kElementsPerAccess>;
  using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  struct Params {
    float *base;
    float const *density;
    int rows_per_atom;
    long long atom_stride, row_stride;
    int const *kind;  // Select: each atom's species
    int select_columns;  // Select: the columns of one species' block
  };

  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;
    bool predicates[kCount];
    CUTLASS_HOST_DEVICE Mask() { enable(); }
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) predicates[i] = false;
    }
    CUTLASS_HOST_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) predicates[i] = true;
    }
  };

 private:
  Params params_;
  Mask mask_;
  Index extent_row_, extent_column_, thread_start_row_, thread_start_column_;
  int state_[3];

 public:
  CUTLASS_DEVICE
  RemapDivIterator(Params const &params, TensorCoord extent, int thread_idx, TensorCoord threadblock_offset)
      : params_(params) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;
    extent_row_ = extent.row();
    extent_column_ = extent.column();
    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] = (thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column();
    }
    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_DEVICE void load(Fragment &frag) const { frag.clear(); }

  CUTLASS_DEVICE void store(Fragment const &frag) const {
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int const frag_row_idx =
              row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster);
          int const row_offset =
              row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
          int const r = row_offset + thread_start_row_;
          if (r >= extent_row_) continue;
          int const atom = r / params_.rows_per_atom;
          int const m = r - atom * params_.rows_per_atom;
          float *row_pointer = params_.base + atom * params_.atom_stride + m * params_.row_stride + thread_start_column_;
          float scale = 0.0f;
          if (params_.density) scale = params_.density[atom] + 1.0f;
          int species = 0;
          if (Select) species = params_.kind[atom];
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            AccessType value = frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
            if (params_.density) {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < kElementsPerAccess; ++e) value[e] = div_full(value[e], scale);
            }
            if (Select) {
              int const j = thread_start_column_ + column * ThreadMap::Delta::kColumn;
              int const block = j / params_.select_columns;
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                  value, (void *)(row_pointer - thread_start_column_ + (j - block * params_.select_columns)),
                  mask_.predicates[column] && block == species);
            } else {
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                  value, (void *)(row_pointer + column * ThreadMap::Delta::kColumn), mask_.predicates[column]);
            }
          }
        }
      }
    }
  }

  CUTLASS_HOST_DEVICE RemapDivIterator &operator++() {
    ++state_[0];
    thread_start_row_ += ThreadMap::Shape::kRow;
    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        thread_start_row_ +=
            ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          thread_start_row_ +=
              ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
        }
      }
    }
    return *this;
  }

  CUTLASS_DEVICE void clear_mask() { mask_.clear(); }
  CUTLASS_DEVICE void enable_mask() { mask_.enable(); }
  CUTLASS_DEVICE void get_mask(Mask &mask) const { mask = mask_; }
  CUTLASS_DEVICE void set_mask(Mask const &mask) { mask_ = mask; }
  CUTLASS_DEVICE TensorCoord thread_start() const { return TensorCoord(thread_start_row_, thread_start_column_); }
  CUTLASS_DEVICE Index thread_start_row() const { return thread_start_row_; }
  CUTLASS_DEVICE Index thread_start_column() const { return thread_start_column_; }
};

// The default kernel's epilogue with RemapDivIterator as its output iterator (every other part the same).
template <typename Kernel, bool Select = false>
struct RemapDiv {
  using Mma = typename Kernel::Mma;
  using Default = typename Kernel::Epilogue;
  using Iterator = RemapDivIterator<typename Default::OutputTileIterator::ThreadMap, Select>;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename Default::Shape, typename Default::WarpMmaOperator, Default::kPartitionsK, Iterator,
      typename Default::AccumulatorFragmentIterator, typename Default::WarpTileIterator,
      typename Default::SharedLoadIterator, typename Default::OutputOp, typename Default::Padding,
      Default::kFragmentsPerIteration>;
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
};

// Kernel::operator() without split-K (gemm_k_size = K, identity swizzle of one tile), with RemapDiv's epilogue.
template <typename Kernel, bool Select = false>
__device__ __forceinline__ void run_remap_div(int m, int n, int k, float const *a, int lda, float const *b, int ldb,
                                              float *c, int rows_per_atom, long long atom_stride,
                                              long long row_stride, float const *density, int const *kind = nullptr,
                                              int select_columns = 0) {
  using R = RemapDiv<Kernel, Select>;
  using Mma = typename R::Mma;
  extern __shared__ __align__(16) char storage[];
  auto &shared = *reinterpret_cast<typename R::SharedStorage *>(storage);
  int const tile_m = blockIdx.x, tile_n = blockIdx.y;
  cutlass::MatrixCoord tb_offset_a{tile_m * Mma::Shape::kM, 0};
  cutlass::MatrixCoord tb_offset_b{0, tile_n * Mma::Shape::kN};
  int const gemm_k_iterations = (k + Mma::Shape::kK - 1) / Mma::Shape::kK;
  int const thread_idx = threadIdx.x;
  typename Mma::IteratorA iterator_a(typename Mma::IteratorA::Params(typename Mma::IteratorA::Layout(lda)),
                                     const_cast<float *>(a), {m, k}, thread_idx, tb_offset_a);
  typename Mma::IteratorB iterator_b(typename Mma::IteratorB::Params(typename Mma::IteratorB::Layout(ldb)),
                                     const_cast<float *>(b), {k, n}, thread_idx, tb_offset_b);
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;
  Mma mma(shared.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();
  mma(gemm_k_iterations, accumulators, iterator_a, iterator_b, accumulators);
  typename R::Epilogue::OutputOp output_op(typename R::Epilogue::OutputOp::Params(1.0f, 0.0f));
  typename R::Iterator::Params params{c, density, rows_per_atom, atom_stride, row_stride, kind, select_columns};
  cutlass::MatrixCoord tb_offset{tile_m * Mma::Shape::kM, tile_n * Mma::Shape::kN};
  typename R::Iterator iterator_d(params, {m, n}, thread_idx, tb_offset);
  typename R::Epilogue epilogue(shared.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_d, accumulators, iterator_d);
}

}  // namespace

extern "C" {

__constant__ int fm_tf32x3_remap_info[3] = {RowRow::kThreadCount, int(sizeof(typename RemapDiv<RowRow>::SharedStorage)),
                                            int(sizeof(typename RemapDiv<RowCol>::SharedStorage))};

#define FM_TF32X3_REMAP_ENTRY(name, Kernel)                                                                       \
  __global__ void __launch_bounds__(Kernel::kThreadCount)                                                         \
  name(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int rows_per_atom,        \
       long long atom_stride, long long row_stride, float const *density) {                                       \
    run_remap_div<Kernel>(m, n, k, a, lda, b, ldb, c, rows_per_atom, atom_stride, row_stride, density);           \
  }

FM_TF32X3_REMAP_ENTRY(fm_tf32x3_remap_div_rr_64x64, RowRow)
FM_TF32X3_REMAP_ENTRY(fm_tf32x3_remap_div_rc_64x64, RowCol)

#define FM_TF32X3_SELECT_ENTRY(name, Kernel)                                                                      \
  __global__ void __launch_bounds__(Kernel::kThreadCount)                                                         \
  name(int m, int n, int k, float const *a, int lda, float const *b, int ldb, float *c, int rows_per_atom,        \
       long long atom_stride, long long row_stride, float const *density, int const *kind, int select_columns) {   \
    run_remap_div<Kernel, true>(m, n, k, a, lda, b, ldb, c, rows_per_atom, atom_stride, row_stride, density, kind, \
                                select_columns);                                                                  \
  }

FM_TF32X3_SELECT_ENTRY(fm_tf32x3_remap_select_rr_64x64, RowRow)
FM_TF32X3_SELECT_ENTRY(fm_tf32x3_remap_select_rc_64x64, RowCol)

}  // extern "C"
