#!/usr/bin/env bash
# OPT-T01 CUTLASS 3xTF32 rerun with cancellation / exponent operand sets on a fresh Colab A100.
set -u
S=fm-cutlass-cancel2-a100
OUT=$(cd "$(dirname "$0")" && pwd)
flt() { grep -v 'new version\|colab update\|silence'; }
tar -cf $OUT/src.tar -C $OUT/src .
colab usage 2>&1 | flt | head -3 > $OUT/cu-before.txt
colab new --gpu A100 -s $S 2>&1 | flt | tail -2
trap 'colab stop -s $S 2>&1 | flt | tail -1' EXIT
colab exec -s $S -f $OUT/mkdirs.py 2>&1 | flt
colab upload -s $S $OUT/src.tar /content/cutlass_bench/src.tar 2>&1 | flt
colab exec -s $S -f $OUT/remote_build.py --timeout 3500 2>&1 | flt > $OUT/build.txt; grep -E '^exit|FAILED|built|^v[0-9]' $OUT/build.txt
colab exec -s $S -f $OUT/remote_run.py --timeout 3500 2>&1 | flt | tee $OUT/remote.txt | tail -12
colab download -s $S /content/cutlass_bench/result.tgz $OUT/result.tgz 2>&1 | flt
tar -xzf $OUT/result.tgz -C $OUT && echo results in $OUT
