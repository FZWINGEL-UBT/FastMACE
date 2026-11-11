#!/usr/bin/env bash
# SEGMENT_ROWS on the Ada (ab4.sh A7), on murphy after ada_gate_set.sh, on src-screen. The RTX 5060 Ti is out of scope:
# the CUTLASS backend has sm_80 / sm_90 cubins only (kernels/cutlass/tf32x3.py), none for sm_120.
B=/mnt/homes/zwingel/fm-ada5060
until grep -q DONE $B/runs/ada-gate/run.log 2>/dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-srows
mkdir -p $ROOT
bash $B/jobs/ab4.sh $B/src-screen A7
