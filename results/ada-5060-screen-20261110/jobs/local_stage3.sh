#!/usr/bin/env bash
# RTX 5060 Ti stage 3 (11 November 2026), through local_gpu.py, default allocator, on src-screen. Set F = stage 2's D
# without the switches its leave-one-out found useless or slower (CUEQ_PROJECTED, BLOCK_ORDER, GRAPH_SUM, DEGREE_PARTS,
# SPLINE_VECTOR 4; SYMMETRIC_PAIRS mixed by structure) and with SC_FACTORED (X9: faster in both orders on both).
#   X11: F (with fold_product_linear) against today's 5060 Ti defaults.
#   X12: a = F; b = F without SC_GENERATED + SC_FACTORED, without SORT_BY_LENGTH, with SYMMETRIC_PAIRS.
#   Gate (3x float32) of F, OMAT-0 medium: NaCl, NbCl5, SrTiO3 3x3x3.
#   Other models (as on the Ada): POLAR-1-M NaCl + NbCl5, OFF23-medium water_216; gates of POLAR-1-M and OFF23-medium.
J=$HOME/fm-ada5060/jobs S=$HOME/fm-ada5060/src-screen OUT=$HOME/fm-ada5060/runs/local-stage3
mkdir -p $OUT
unset PYTORCH_CUDA_ALLOC_CONF
# stage 2 wrote its JSONs into src-screen/<RECEIVER_SUM constant> (R named two things): move them to its run dir
stray="$S/fastmace.kernels.triton.receiver_sum:RECEIVER_SUM"
[ -d "$stray" ] && mkdir -p $HOME/fm-ada5060/runs/local-stage2 && cp -r "$stray"/. $HOME/fm-ada5060/runs/local-stage2/ && rm -r "$stray"
cp -n /mnt/g/Margraf/ResearchModule/Structures/Organic/water_216.extxyz $HOME/fm-ada5060/water_216.extxyz
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin g=fastmace.rewrites.sc_generated t=fastmace.rewrites.species
PW=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD RS=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
SYM=fastmace.rewrites.radial:SYMMETRIC_PAIRS
F="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $g:SC_FACTORED=True $t:GENERIC_FOLD_READOUT_TAIL=True $PW=True $RS=True $w:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True"
FOLD=fold_product_linear=on
export PACK_X11="final|$F $FOLD"
export PACK_X12="no-scgf|$g:SC_GENERATED=False $g:SC_FACTORED=False;no-sbl|$k:SORT_BY_LENGTH=False;sym|$SYM=True"
export AC_X12="--a $FOLD --b $FOLD --a-constant $F --b-constant $F"
gpu() { python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen stage 3 $1 (d66)" -- \
  env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$OUT "${@:2}"; echo "$(date -u +%FT%TZ) $1 exit=$?" >> $OUT/chain.log; }
gpu "X11 X12" bash $J/ab3.sh $S X11 X12
# shellcheck disable=SC2086
gpu gate-omat0 bash $J/gate.sh $S set-5060 omat0-medium mace-omat-0-medium.model \
  "NaCl NbCl5 results/loup-a100-20261103/SrTiO3.cif" "3 3 3" $F -- $FOLD
export F_MODELS="$F" W=$HOME/fm-ada5060/water_216.extxyz
gpu models bash $J/local_models.sh $S
echo "$(date -u +%FT%TZ) DONE" >> $OUT/chain.log
