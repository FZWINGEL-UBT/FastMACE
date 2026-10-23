#include "kernels.cuh"
Runner* make_CR_1_4_4_t2(bool fast) {
  return make_runner<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, 1, 4, 4, 2>(fast);
}
