#include "kernels.cuh"
Runner* make_RC_4_4_1_t1(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, 4, 4, 1, 1>(fast);
}
