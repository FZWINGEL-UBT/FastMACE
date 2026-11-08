#!/bin/bash -l
#SBATCH --job-name=fm-radtf32-ab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-67a0d977; cd $PYTHONPATH
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off24-medium:MACE-OFF24_medium.model off23b-medium:MACE-OFF23b_medium.model off23-large:MACE-OFF23_large.model; do
  m=${p%%:*}; c=${p#*:}
  OUT=$B/runs/radtf32-m80-$m-$SLURM_JOB_ID MODEL=$m CKPT=$c STRUCTS="$ORG/water_216.extxyz@1x1x1" SETS_FILE=$B/sets/radtf32.tsv WARMUP=500 BLOCK=1000 BLOCKS=16 bash tools/offpolar_ab.sh
done
grep -h "end ab" $B/runs/radtf32-m80-*-$SLURM_JOB_ID/run.log
