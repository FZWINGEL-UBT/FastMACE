#include "kernels.cuh"
Runner* make_RC_1_1_2_t0(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, 1, 1, 2, 0>(fast);
}
