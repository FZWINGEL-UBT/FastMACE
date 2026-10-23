#include "kernels.cuh"
Runner* make_RC_2_2_4_t1(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, 2, 2, 4, 1>(fast);
}
