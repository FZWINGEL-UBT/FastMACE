#!/usr/bin/env bash
# The final adopted Ada set (conductor, 11 November 2026) in one A/B against today's Ada defaults (src-screen: main
# 4e9388c8 with the registry fix, without the screen's GPU-list changes), both orders, on murphy after ada_virial2.sh.
B=/mnt/homes/zwingel/fm-ada5060
until grep -q DONE $B/runs/ada-virial2/run.log 2>/dev/null; do sleep 60; done
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin g=fastmace.rewrites.sc_generated
SET="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $g:SC_FACTORED=True $w:BUTTERFLY=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD=True"
export PACK_X10="final|$SET"
mkdir -p $B/runs/ada-final
QUIET=3 bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/ada-final bash $B/jobs/ab4.sh $B/src-screen X10
