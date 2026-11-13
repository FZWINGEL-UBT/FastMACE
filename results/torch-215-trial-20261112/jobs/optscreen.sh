#!/bin/bash
# Offline screen of NVRTC/ptxas options on the Ada's SrTiO3 w' render (481ed92397), NVRTC 13.0 and 13.5, sm_89 and
# sm_120: SASS per option set for sass_sched.py (stall cycles, load-to-use distances). CPU only (leto).
B=/mnt/homes/zwingel/t215; C=$B/census; S=$B/src-b7c19b19; O=$C/optscreen; mkdir -p $O
PY=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
export PATH=$(dirname $PY)/../lib/python3.10/site-packages/triton/backends/nvidia/bin:$PATH CUDA_VISIBLE_DEVICES= TMPDIR=$B/tmp
cd $S; src=$C/src89/oeq_projected_backward_wprime-sm89-481ed92397.cu
declare -A OPTS=(
  [base]=""
  [vec]="--extra-device-vectorization"
  [r96]="--maxrregcount=96"
  [r128]="--maxrregcount=128"
  [r72]="--maxrregcount=72"
  [xexp]="--ptxas-options=--allow-expensive-optimizations=true"
  [o2]="--ptxas-options=-O2"
  [nobra]="--ptxas-options=--disable-optimizer-constants"
  [mncta3]="--minnctapersm=3"
  [mncta4]="--minnctapersm=4"
  [ofc0]="-Ofast-compile=0"
  [dopt]="--dopt=on"
)
for v in 13.0.88 13.5.37; do lib=$C/nvrtc-$v/nvidia/cu13/lib/libnvrtc.so.13
  for k in "${!OPTS[@]}"; do for a in 89 120; do
    args=(); for o in ${OPTS[$k]}; do args+=(--opt "$o"); done
    ( LD_LIBRARY_PATH=$(dirname $lib) PYTHONPATH=$S $PY -W ignore $B/jobs.p11/sass_dump.py $lib $a $O/$v-$k $src "${args[@]}" > $O/$v-$k-$a.log 2>&1 || echo "FAIL $v $k $a: $(tail -2 $O/$v-$k-$a.log | tr '\n' ' ' | cut -c1-200)" ) &
  done; done; wait
done
ls $O | wc -l
