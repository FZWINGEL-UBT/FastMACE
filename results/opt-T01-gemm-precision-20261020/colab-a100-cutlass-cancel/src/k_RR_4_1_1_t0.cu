#include "kernels.cuh"
Runner* make_RR_4_1_1_t0(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::RowMajor, 4, 1, 1, 0>(fast);
}
