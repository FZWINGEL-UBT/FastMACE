#!/usr/bin/env bash
cd ~/tmp/f64probe
P=~/environments/mace_cueq_oeq/bin/python
rm -f gemm64b.jsonl
$P gemm64b.py native >> gemm64b.jsonl 2>gemm64b.err
CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager $P gemm64b.py eager >> gemm64b.jsonl 2>>gemm64b.err
CUBLAS_EMULATE_DOUBLE_PRECISION=1 $P gemm64b.py default >> gemm64b.jsonl 2>>gemm64b.err
echo ALLDONE >> gemm64b.jsonl
