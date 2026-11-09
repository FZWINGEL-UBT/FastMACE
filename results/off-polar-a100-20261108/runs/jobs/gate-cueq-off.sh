#!/bin/bash -l
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext
export PYTHONPATH=/home/atuin/v130cb/v130cb18/fm-offpolar/src-1bd4f3be; cd $PYTHONPATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
ORG=/home/atuin/v130cb/v130cb18/fm-offpolar/baseline/Structures/Organic
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23b-medium:MACE-OFF23b_medium.model off24-medium:MACE-OFF24_medium.model; do
  m=${p%%:*}; c=${p#*:}
  GS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1"
  R=/home/atuin/v130cb/v130cb18/fm-offpolar/runs/gate-cueq-$m-$SLURM_JOB_ID
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="$GS" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="$GS" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --production-constant fastmace.rewrites.cueq_projection:CUEQ_PROJECTED=True" bash tools/offpolar_ab.sh
done
python /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/gatesum.py /home/atuin/v130cb/v130cb18/fm-offpolar/runs/gate-cueq-*-$SLURM_JOB_ID/gate-*.log
