#!/bin/bash -l
#SBATCH --job-name=fm-radtf32-gate-a
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-67a0d977; cd $PYTHONPATH
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23-large:MACE-OFF23_large.model; do
  m=${p%%:*}; c=${p#*:}
  if [ $m = off23-large ]; then GS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif@2x2x2 $ORG/water_216.extxyz@1x1x1"
  else GS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1"; fi
  R=$B/runs/gate-radtf-$m-$SLURM_JOB_ID
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="$GS" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="$GS" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --production-constant fastmace.kernels.triton.skinny_linear:RADIAL_TF32X3=True" bash tools/offpolar_ab.sh
done
python $B/jobs/gatesum.py $B/runs/gate-radtf-*-$SLURM_JOB_ID/gate-*.log
