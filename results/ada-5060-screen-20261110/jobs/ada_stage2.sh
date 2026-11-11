#!/usr/bin/env bash
# Ada stage 2 (ab2.sh X packs): the stage-1 winners together against the Ada defaults (X1), then leave one out from
# that set (X2-X4; side a = the set, side b = the set with one switch off, so above 1 means the switch helps).
#   bash ada_stage2.sh NAME PACK...   (behind queue.sh; logs under runs/NAME)
B=/mnt/homes/zwingel/fm-ada5060
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin g=fastmace.rewrites.sc_generated e=fastmace.model.energy
P=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD R=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
C="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $w:BUTTERFLY=True $s:BLOCK_ORDER=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $P=True $R=True $e:GRAPH_SUM=True $w:SCALAR_PATHS=True"
CORE="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $w:BUTTERFLY=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $P=True"
export PACK_X1="all|$C;core|$CORE"
export PACK_X2="no-p8|$s:PACKED8=False;no-sbl|$k:SORT_BY_LENGTH=False;no-scg|$g:SC_GENERATED=False;no-bfly|$w:BUTTERFLY=False"
export PACK_X3="no-bo|$s:BLOCK_ORDER=False;no-wse|$w:STAGE_EDGES=False;no-sshwse|$w:STAGE_SH=False $w:STAGE_EDGES=False;no-sdall|$w:STAGE_DOUT=False $w:STAGE_SH=False $w:STAGE_EDGES=False"
export PACK_X4="no-power|$P=False;no-rsum|$R=False;no-gsum|$e:GRAPH_SUM=False;no-scalar|$w:SCALAR_PATHS=False"
for x in X2 X3 X4; do export AC_$x="--a-constant $C --b-constant $C"; done
name=$1; shift
mkdir -p $B/runs/$name
bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/$name bash $B/jobs/ab2.sh $B/src-7d18028d "$@"
