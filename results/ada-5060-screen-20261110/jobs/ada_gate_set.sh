#!/usr/bin/env bash
# The force/stress gate (3x float32) of the Ada's adopted OMAT-0 medium set (rounding switches among them: PACKED8,
# SC_GENERATED, SC_FACTORED, BUTTERFLY, POWER_BACKWARD, STAGE_*), on murphy after murphy_chain7.sh, on src-screen:
# NaCl, NbCl5 and SrTiO3 3x3x3. The GEOM / SC_FACTORED / VIRIAL decisions follow ada_new.sh; their own gates there.
B=/mnt/homes/zwingel/fm-ada5060
while pgrep -f "[m]urphy_chain7.sh" > /dev/null; do sleep 60; done
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin g=fastmace.rewrites.sc_generated
SET="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $w:BUTTERFLY=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD=True $g:SC_FACTORED=True"
mkdir -p $B/runs/ada-gate
QUIET=3 bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/ada-gate bash $B/jobs/gate.sh $B/src-screen ada-set omat0-medium \
  mace-omat-0-medium.model "NaCl NbCl5 results/loup-a100-20261103/SrTiO3.cif" "3 3 3" $SET
echo "$(date -u +%FT%TZ) DONE" >> $B/runs/ada-gate/run.log
