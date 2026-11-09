#!/bin/bash -l
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext
export PYTHONPATH=/home/atuin/v130cb/v130cb18/fm-offpolar/src-1bd4f3be; cd $PYTHONPATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
ORG=/home/atuin/v130cb/v130cb18/fm-offpolar/baseline/Structures/Organic
for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model; do
  m=${p%%:*}; c=${p#*:}
  R=/home/atuin/v130cb/v130cb18/fm-offpolar/runs/gate-mega-$m-$SLURM_JOB_ID
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --production-constant fastmace.longrange.polar:MEGAKERNEL=True" bash tools/offpolar_ab.sh
done
python /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/gatesum.py /home/atuin/v130cb/v130cb18/fm-offpolar/runs/gate-mega-*-$SLURM_JOB_ID/gate-*.log
