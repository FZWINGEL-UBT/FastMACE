#!/bin/bash -l
#SBATCH --job-name=fm-gate-g80tf-polar1-l
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
R=$B/runs/gate-g80tf-polar1-l-$SLURM_JOB_ID
OUT=$R MODEL=polar1-l CKPT=MACE-POLAR-1-L.model STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
OUT=$R MODEL=polar1-l CKPT=MACE-POLAR-1-L.model STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --rewrites tf32x3_gemms=on" bash tools/offpolar_ab.sh
cat $R/run.log
