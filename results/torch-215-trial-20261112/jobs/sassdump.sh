#!/bin/bash
# SASS of the Ada's SrTiO3 w' render (481ed92397) and spline forward per NVRTC version, sm_89 (and sm_120). CPU only.
B=/mnt/homes/zwingel/t215; C=$B/census; S=$B/src-b7c19b19; O=$C/sass89; mkdir -p $O
PY=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
export PATH=$(dirname $PY)/../lib/python3.10/site-packages/triton/backends/nvidia/bin:$PATH CUDA_VISIBLE_DEVICES= TMPDIR=$B/tmp
cd $S
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do lib=$C/nvrtc-$v/nvidia/cu13/lib/libnvrtc.so.13
  for a in 89 120; do for s in $C/src89/oeq_projected_backward_wprime-sm89-481ed92397.cu; do
    LD_LIBRARY_PATH=$(dirname $lib) PYTHONPATH=$S $PY -W ignore $B/jobs.p11/sass_dump.py $lib $a $O/$v $s &
  done; done; done; wait
ls -la $O/*/ | head -20
