# Third-party code

`tf32x3_sm80.cubin` is compiled from `tf32x3_sm80.cu` and the headers of NVIDIA
CUTLASS v4.8.1 (https://github.com/NVIDIA/cutlass, commit
2c85f497547eb441f0fe80bfb0fb2e6f9add1719): the sm_80 tensor-core GEMM
(`cutlass::gemm::kernel::DefaultGemm` with `OpMultiplyAddFastF32`, the 3xTF32
split of CUTLASS example 27). CUTLASS is distributed under the BSD 3-Clause
licence (`LICENSE` here, copied from CUTLASS's `LICENSE.txt`); its copyright
notice and licence terms apply to the cubin.

`tf32x3_sm80.cu` itself is FastMACE's: two `extern "C"` entries that build
CUTLASS's kernel parameters on the device. The build command, the compiler and
the cubin's SHA-256 are in `tf32x3_sm80.build.txt`
(`tools/build_cutlass_tf32x3.sh`).
