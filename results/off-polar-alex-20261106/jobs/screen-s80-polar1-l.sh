#!/bin/bash -l
#SBATCH --job-name=fm-screen-s80-polar1-l
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
OUT=$B/runs/screen-s80-polar1-l-$SLURM_JOB_ID MODEL=polar1-l CKPT=MACE-POLAR-1-L.model STRUCTS="NaCl NbCl5" SETS_FILE=/home/atuin/v130cb/v130cb18/fm-offpolar/sets/polar.tsv BLOCK=400 BLOCKS=12 bash tools/offpolar_ab.sh
cat $B/runs/screen-s80-polar1-l-$SLURM_JOB_ID/run.log
