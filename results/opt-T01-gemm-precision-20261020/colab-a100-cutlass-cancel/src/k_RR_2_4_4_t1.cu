#include "kernels.cuh"
Runner* make_RR_2_4_4_t1(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::RowMajor, 2, 4, 4, 1>(fast);
}
