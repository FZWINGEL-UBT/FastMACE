#!/bin/bash
# On Alex: mkjob.sh <name> <a100_80|a100_40> <HH:MM:SS> <body-file>  -> writes jobs/<name>.sh and submits it.
set -euo pipefail
B=/home/atuin/v130cb/v130cb18/fm-offpolar
name=$1; feat=$2; time=$3; body=$4
mkdir -p $B/jobs $B/logs
f=$B/jobs/$name.sh
{
  echo '#!/bin/bash -l'
  echo "#SBATCH --job-name=fm-$name"
  echo "#SBATCH -p a100 --gres=gpu:a100:1 -C $feat"
  echo "#SBATCH --time=$time"
  echo '#SBATCH --export=NONE'
  echo "#SBATCH --output=$B/logs/%x-%j.out"
  echo 'unset SLURM_EXPORT_ENV'
  echo "source $B/jobs/env.sh"
  echo 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)'
  cat "$body"
} > $f
sbatch --parsable $f
