#!/usr/bin/env bash
cd ~/tmp/f64probe; P=~/environments/mace_cueq_oeq/bin/python; rm -f emu.jsonl
env -u CUBLAS_EMULATE_DOUBLE_PRECISION $P gemm_emu.py native >> emu.jsonl 2>emu.err
CUBLAS_EMULATE_DOUBLE_PRECISION=1 $P gemm_emu.py emu-default >> emu.jsonl 2>>emu.err
CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager $P gemm_emu.py emu-eager >> emu.jsonl 2>>emu.err
for b in 40 48 56 64 79; do CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=$b $P gemm_emu.py emu-eager-m$b >> emu.jsonl 2>>emu.err; done
echo DONE >> emu.err
