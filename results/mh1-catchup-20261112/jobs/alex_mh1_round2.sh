#!/bin/bash -l
# MH-1 catch-up round 2 as ONE Alex A100 job (owner via conductor: one long A100 job per round), source 62536879:
# main 53666e77 + TABLE_W for MH-1 on every A100 (round 1, Alex 4990788).
# In sequence:
#   1. jobs/alex_mh1_base2.sh: MD step times on the new defaults (MH-1 NaCl / NaNbO3 / NbCl5 3x3x3, NbCl5 6x6x6;
#      OMAT-0 medium NbCl5 3x3x3), then the torch 2.15 CUDA-graph attribution with CUPTI >= 13.3 (round 1 failed on
#      CUPTI 13.2) for MH-1 NbCl5 / NaCl / NaNbO3 and OMAT-0 medium NbCl5;
#   2. jobs/alex_mh1_ab.sh, one pack against the new defaults: the round-1 gains to confirm on the TABLE_W base
#      (nosegrows: NaCl 0.952 / 0.953, NbCl5 0.991 / 0.990; noweighted: 0.991 / 0.991, 0.993 / 0.992), both together,
#      and the two small ones (noint32, nosort), then the 3x float32 gate of each rounding candidate.
# Source tree and caches node-local ($TMPDIR); records under $B/runs.
#   sbatch jobs/alex_mh1_round2.sh
#SBATCH --job-name=fm-mh1r2
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-mh1/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
B=/home/atuin/v130cb/v130cb18/fm-mh1
SHA=62536879
W=fastmace.kernels.oeq_streaming.wprime_backward
S=fastmace.kernels.cutlass.segment_rows
K=fastmace.neighbors.skin
PACK="nosegrows|$S:SEGMENT_ROWS=False;noweighted|$W:WEIGHTED_VECTOR=False;nosegrows_noweighted|$S:SEGMENT_ROWS=False $W:WEIGHTED_VECTOR=False;noint32|$K:EDGE_INDEX_INT32=False;nosort|$K:SORT_BY_LENGTH=False"
GATES="nosegrows|$S:SEGMENT_ROWS=False;noweighted|$W:WEIGHTED_VECTOR=False;nosegrows_noweighted|$S:SEGMENT_ROWS=False $W:WEIGHTED_VECTOR=False;nosort|$K:SORT_BY_LENGTH=False"
echo "$(date +%FT%T%z) round2 job $SLURM_JOB_ID on $(hostname)"
bash $B/jobs/alex_mh1_base2.sh $SHA; echo "$(date +%FT%T%z) base exit=$?"
rm -rf $TMPDIR/mh1
SRC=$TMPDIR/src-$SHA; mkdir -p $SRC && tar -xf $B/src-$SHA.tar -C $SRC && echo $SHA > $SRC/.source_commit
O=$B/runs/r2-ab; mkdir -p $O
bash $B/jobs/alex_mh1_ab.sh $SRC $O "$PACK" "$GATES"; echo "$(date +%FT%T%z) ab exit=$?"
rm -rf $SRC
