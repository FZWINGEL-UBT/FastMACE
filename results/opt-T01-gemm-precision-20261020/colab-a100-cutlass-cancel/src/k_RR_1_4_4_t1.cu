#include "kernels.cuh"
Runner* make_RR_1_4_4_t1(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::RowMajor, 1, 4, 4, 1>(fast);
}
