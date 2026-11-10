#!/bin/bash -l
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext
export PYTHONPATH=/home/atuin/v130cb/v130cb18/fm-offpolar/src-b4ec7053; cd $PYTHONPATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
ORG=/home/atuin/v130cb/v130cb18/fm-offpolar/baseline/Structures/Organic
m=polar1-l; c=MACE-POLAR-1-L.model
R=/home/atuin/v130cb/v130cb18/fm-offpolar/runs/gate-cueq-$m-$SLURM_JOB_ID
OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
python /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/gatesum.py $R/gate-*.log
