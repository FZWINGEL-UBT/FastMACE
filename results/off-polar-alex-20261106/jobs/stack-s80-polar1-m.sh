#!/bin/bash -l
#SBATCH --job-name=fm-stack-s80-polar1-m
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
OUT=$B/runs/stack-s80-polar1-m-$SLURM_JOB_ID polar1-m=polar1-m MACE-POLAR-1-M.model=CKPT STRUCTS="NaCl NbCl5" SETS_FILE=$B/sets/polar_stack.tsv BLOCK=400 BLOCKS=12 bash tools/offpolar_ab.sh
cat $B/runs/stack-s80-polar1-m-$SLURM_JOB_ID/run.log
