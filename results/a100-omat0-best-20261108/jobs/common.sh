# Sourced by the a100-omat0-best job files (8 November 2026): the switch sets, as MODULE:NAME=VALUE.
s=fastmace.kernels.oeq_streaming.spline_forward
w=fastmace.kernels.oeq_streaming.wprime_backward
c=fastmace.kernels.triton.compose_skip
q=fastmace.rewrites.cueq_projection
t=fastmace.rewrites.species
p=fastmace.kernels.triton.tabled_outer_sum
ALL="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $w:SPLINE_VECTOR=4 $w:STAGE_SH=True $c:COMPOSE_SKIP=True $c:COMPOSE_FAST=True $c:COUNT_PLAN=True $q:CUEQ_PROJECTED=True $t:GENERIC_FOLD_READOUT_TAIL=True"
B=/home/atuin/v130cb/v130cb18/fm-a100omat
