#!/bin/bash -l
#SBATCH --job-name=fm-profile2
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-91e428a1; cd $PYTHONPATH
R=$B/runs/profile-off-$SLURM_JOB_ID; mkdir -p $R
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off24-medium:MACE-OFF24_medium.model off23-large:MACE-OFF23_large.model; do
  m=${p%%:*}; c=${p#*:}
  python tools/module_profile.py --output $R/$m --checkpoint $FASTMACE_MODELS/$c --model $m --structure $ORG/water_216.extxyz@1x1x1 > $R/$m.log 2>&1
done
ls $R/*
