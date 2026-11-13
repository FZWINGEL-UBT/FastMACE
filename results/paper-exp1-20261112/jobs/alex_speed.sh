#!/bin/bash -l
# Measurement 5 on Alex A100: speed.sh (a against b = PACKED8 off, both build orders, SrTiO3 10x10x10, NbCl5 3x3x3).
#SBATCH --job-name=fm-pexp1-speed
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:00:00
source $SLURM_SUBMIT_DIR/alex_common.sh
mkdir -p $LOUT/speed
RUNNER=alex_step; alex_step() { name=$1; shift 2; run $name "$@"; }
SPEED_LIMIT=0 PY=python source $SLURM_SUBMIT_DIR/speed.sh
note DONE
