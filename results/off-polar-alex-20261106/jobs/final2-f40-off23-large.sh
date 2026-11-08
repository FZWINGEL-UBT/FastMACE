#!/bin/bash -l
#SBATCH --job-name=fm-final2-f40-off23-large
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_40
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-91e428a1; cd $PYTHONPATH
R=$B/runs/final-f40-off23-large-$SLURM_JOB_ID
OUT=$R MODEL=off23-large CKPT=MACE-OFF23_large.model STRUCTS="$ORG/water_216.extxyz@1x1x1" SETS_FILE=$B/sets/off_final.tsv WARMUP=500 BLOCK=1000 BLOCKS=16 bash tools/offpolar_ab.sh
python tools/plan_describe.py --output $R/plan --structure $ORG/water_216.extxyz@1x1x1 --models off23-large:MACE-OFF23_large.model
cat $R/run.log
