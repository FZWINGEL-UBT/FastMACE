#!/bin/bash -l
#SBATCH --job-name=fm-gate-g80-off23b-medium
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
R=$B/runs/gate-g80-off23b-medium-$SLURM_JOB_ID
OUT=$R MODEL=off23b-medium CKPT=MACE-OFF23b_medium.model STRUCTS=x GATE_STRUCTS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
OUT=$R MODEL=off23b-medium CKPT=MACE-OFF23b_medium.model STRUCTS=x GATE_STRUCTS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --rewrites tf32x3_gemms=on" bash tools/offpolar_ab.sh
cat $R/run.log
