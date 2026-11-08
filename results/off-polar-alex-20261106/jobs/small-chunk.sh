#!/bin/bash -l
#SBATCH --job-name=fm-small-chunk
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-982e89cf; cd $PYTHONPATH
R=$B/runs/smallchunk-m80-off23-small-$SLURM_JOB_ID
GS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1"
OUT=$R MODEL=off23-small CKPT=MACE-OFF23_small.model STRUCTS="$ORG/water_216.extxyz@1x1x1" SETS_FILE=$B/sets/small_chunk.tsv WARMUP=500 BLOCK=1000 BLOCKS=16 bash tools/offpolar_ab.sh
OUT=$R MODEL=off23-small CKPT=MACE-OFF23_small.model STRUCTS=x GATE_STRUCTS="$GS" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
OUT=$R MODEL=off23-small CKPT=MACE-OFF23_small.model STRUCTS=x GATE_STRUCTS="$GS" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --rewrites chunked_conv_backward=on fused_folded_forward=on" bash tools/offpolar_ab.sh
cat $R/run.log
