#include "kernels.cuh"
Runner* make_RC_4_4_4_t0(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, 4, 4, 4, 0>(fast);
}
