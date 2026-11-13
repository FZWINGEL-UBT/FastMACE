#!/bin/bash
# leto, CPU only: Triton 3.8 (prod env) split lab for sm_89 (and sm_120 for reference); Triton 3.9 envs if present
set -u
B=/mnt/homes/zwingel/t215; D=$B/splitlab; mkdir -p $D $B/tmp
export CUDA_VISIBLE_DEVICES= TMPDIR=$B/tmp TRITON_CACHE_DIR=$D/cache
for e in /mnt/homes/zwingel/environments/mace_cueq_oeq $(ls -d /mnt/homes/zwingel/environments/*triton39* /mnt/homes/zwingel/environments/mace_cueq_oeq_torch215 2>/dev/null); do
  n=$(basename $e); PY=$e/bin/python
  export PATH=$(ls -d $e/lib/python3.*/site-packages/triton/backends/nvidia/bin):/usr/bin:/bin
  echo "== $n $($PY -c 'import triton;print(triton.__version__)' 2>&1)"
  $PY -W ignore $D/split_lab.py $D/split-$n.json 89 120 2>&1 | tail -8
done
