#!/bin/bash -l
#SBATCH --job-name=fm-merged-polar-sm
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-982e89cf; cd $PYTHONPATH
OUT=$B/runs/merged-m80-polar1-m-$SLURM_JOB_ID MODEL=polar1-m CKPT=MACE-POLAR-1-M.model STRUCTS="NaCl NbCl5" GATE_NAME=auto GATE_ARGS="--seed 20261015" SETS_FILE=$B/sets/merged_polar.tsv BLOCK=400 BLOCKS=12 bash tools/offpolar_ab.sh
OUT=$B/runs/merged-m80-polar1-s-$SLURM_JOB_ID MODEL=polar1-s CKPT=MACE-POLAR-1-S.model STRUCTS="NaCl NbCl5" SETS_FILE=$B/sets/merged_polar.tsv BLOCK=400 BLOCKS=12 bash tools/offpolar_ab.sh
grep -h "end " $B/runs/merged-m80-*-$SLURM_JOB_ID/run.log
