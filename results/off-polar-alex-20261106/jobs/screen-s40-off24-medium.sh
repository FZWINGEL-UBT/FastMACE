#!/bin/bash -l
#SBATCH --job-name=fm-screen-s40-off24-medium
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_40
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
OUT=$B/runs/screen-s40-off24-medium-$SLURM_JOB_ID MODEL=off24-medium CKPT=MACE-OFF24_medium.model STRUCTS="$ORG/water_216.extxyz@1x1x1" SETS_FILE=/home/atuin/v130cb/v130cb18/fm-offpolar/sets/off.tsv WARMUP=500 BLOCK=1000 BLOCKS=16 bash tools/offpolar_ab.sh
cat $B/runs/screen-s40-off24-medium-$SLURM_JOB_ID/run.log
