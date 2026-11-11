#!/usr/bin/env bash
# RTX 5060 Ti stage 2 (11 November 2026), after local_all.sh, through local_gpu.py, default allocator, on src-screen
# (the screen branch with main and the registry fix): the stage-1 winners together (set D, with fold_product_linear)
# against the 5060 Ti defaults (X5), then leave one out from D (X6-X9; a = D, b = D minus one, so above 1 means the
# switch helps); X9 also SC_FACTORED on top of D. tf32x3 and compose stay out (t3c NbCl5 1.041 / 1.040).
J=$HOME/fm-ada5060/jobs S=$HOME/fm-ada5060/src-screen R=$HOME/fm-ada5060/runs/local-stage2
mkdir -p $R
unset PYTORCH_CUDA_ALLOC_CONF
while pgrep -f "[l]ocal_all.sh" > /dev/null; do sleep 60; done
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
f=fastmace.kernels.oeq_streaming.fused_forward k=fastmace.neighbors.skin g=fastmace.rewrites.sc_generated
q=fastmace.rewrites.cueq_projection t=fastmace.rewrites.species e=fastmace.model.energy
P=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD R=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
Y=fastmace.rewrites.radial:SYMMETRIC_PAIRS
SPL="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True"
WSE="$w:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True"
D="$SPL $s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $q:CUEQ_PROJECTED=True $t:GENERIC_FOLD_READOUT_TAIL=True $s:BLOCK_ORDER=True $e:GRAPH_SUM=True $f:DEGREE_PARTS=True $P=True $R=True $Y=True $w:SPLINE_VECTOR=4 $WSE"
FOLD=fold_product_linear=on
export PACK_X5="all|$D $FOLD;nofold|$D"
export PACK_X6="no-p8|$s:PACKED8=False;no-sbl|$k:SORT_BY_LENGTH=False;no-scg|$g:SC_GENERATED=False;no-cueq|$q:CUEQ_PROJECTED=False"
export PACK_X7="no-tail|$t:GENERIC_FOLD_READOUT_TAIL=False;no-bo|$s:BLOCK_ORDER=False;no-gsum|$e:GRAPH_SUM=False;no-dparts|$f:DEGREE_PARTS=False"
export PACK_X8="no-power|$P=False;no-rsum|$R=False;no-sym|$Y=False;no-vec4|$w:SPLINE_VECTOR=2"
export PACK_X9="no-wse|$w:STAGE_EDGES=False $w:STAGE_DOUT=False $w:STAGE_SH=False;no-fold|fold_product_linear=off;scf|$g:SC_FACTORED=True"
for x in X6 X7 X8 X9; do export AC_$x="--a $FOLD --b $FOLD --a-constant $D --b-constant $D"; done
for group in "X5 X6" "X7 X8" X9; do
  python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen stage 2 $group (OMAT-0 medium, d66)" -- \
    env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$R bash $J/ab3.sh $S $group
  echo "$(date -u +%FT%TZ) group '$group' exit=$?" >> $R/chain.log
done
