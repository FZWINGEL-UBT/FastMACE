#include "kernels.cuh"
Runner* make_RR_4_2_2_t0(bool fast) {
  return make_runner<cutlass::layout::RowMajor, cutlass::layout::RowMajor, 4, 2, 2, 0>(fast);
}
