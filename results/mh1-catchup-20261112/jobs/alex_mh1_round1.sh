#!/bin/bash -l
# MH-1 catch-up round 1 as ONE Alex A100 job (owner via conductor, 12 November 2026: one long A100 job per round).
# In sequence: jobs/alex_mh1_base.sh (baseline step times, NVRTC sources, torch 2.15 attribution), then
# jobs/alex_mh1_ab.sh pack 1 (+ gate tablew) and pack 2 (+ gate default). Each part keeps its own records.
#   sbatch jobs/alex_mh1_round1.sh
#SBATCH --job-name=fm-mh1r1
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=09:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-mh1/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
B=/home/atuin/v130cb/v130cb18/fm-mh1
W=fastmace.kernels.oeq_streaming.wprime_backward
C=fastmace.kernels.cutlass.tf32x3
P1="tablew|$W:TABLE_W=True;tablew_noskip|$W:TABLE_W=True $W:SKIP_OUTSIDE=False;noweighted|$W:WEIGHTED_VECTOR=False;nohybrid|$C:HYBRID=False"
P2="nowalk|fastmace.kernels.triton.edge_outer_sum:BACKWARD_WALK=False;nosegrows|fastmace.kernels.cutlass.segment_rows:SEGMENT_ROWS=False;noint32|fastmace.neighbors.skin:EDGE_INDEX_INT32=False;nosort|fastmace.neighbors.skin:SORT_BY_LENGTH=False"
echo "$(date +%FT%T%z) round1 job $SLURM_JOB_ID on $(hostname)"
bash $B/jobs/alex_mh1_base.sh 236fa60c; echo "$(date +%FT%T%z) base exit=$?"
rm -rf $TMPDIR/mh1
O1=$B/runs/r1-ab1-wprime; mkdir -p $O1
bash $B/jobs/alex_mh1_ab.sh $B/src-236fa60c $O1 "$P1" "tablew|$W:TABLE_W=True"; echo "$(date +%FT%T%z) ab1 exit=$?"
O2=$B/runs/r1-ab2-gaps; mkdir -p $O2
bash $B/jobs/alex_mh1_ab.sh $B/src-236fa60c $O2 "$P2" "default|"; echo "$(date +%FT%T%z) ab2 exit=$?"
