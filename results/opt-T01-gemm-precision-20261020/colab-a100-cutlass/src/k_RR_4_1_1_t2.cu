#include "kernels.cuh"
Runner* make_RR_4_1_1_t2(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::RowMajor, 4, 1, 1, 2>(fast);
}
