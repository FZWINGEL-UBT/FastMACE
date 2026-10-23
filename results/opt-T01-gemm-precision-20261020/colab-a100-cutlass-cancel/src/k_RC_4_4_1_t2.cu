#include "kernels.cuh"
Runner* make_RC_4_4_1_t2(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, 4, 4, 1, 2>(fast);
}
