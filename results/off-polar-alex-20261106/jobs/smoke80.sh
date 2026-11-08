#!/bin/bash -l
#SBATCH --job-name=fm-smoke80
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
python tools/plan_describe.py --output $B/runs/plans-$SLURM_JOB_ID --structure $ORG/water_216.extxyz@1x1x1 --models off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23-large:MACE-OFF23_large.model off23b-medium:MACE-OFF23b_medium.model off24-medium:MACE-OFF24_medium.model
python tools/plan_describe.py --output $B/runs/plans-$SLURM_JOB_ID-polar --structure NaCl --models polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model
printf 'smoke\t--b-constant fastmace.model.energy:GRAPH_SUM=False\n' > $B/runs/smoke.tsv
OUT=$B/runs/smoke-$SLURM_JOB_ID MODEL=off23-small CKPT=MACE-OFF23_small.model STRUCTS="$ORG/water_216.extxyz@1x1x1" SETS_FILE=$B/runs/smoke.tsv WARMUP=100 BLOCK=100 BLOCKS=4 bash tools/offpolar_ab.sh
cat $B/runs/smoke-$SLURM_JOB_ID/run.log
