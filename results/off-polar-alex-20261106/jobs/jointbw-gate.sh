#!/bin/bash -l
#SBATCH --job-name=fm-jointbw-gate
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-0f82e749; cd $PYTHONPATH
F=fastmace.longrange.polar:FUSED_FOURIER=True
J=fastmace.kernels.triton.fourier:JOINT_BACKWARD=True
for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model; do
  m=${p%%:*}; c=${p#*:}
  R=$B/runs/gate-jb-$m-$SLURM_JOB_ID
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
  OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="NaCl NbCl5" GATE_NAME=candidate GATE_ARGS="--seed 20261015 --production-constant $F $J" bash tools/offpolar_ab.sh
done
python $B/jobs/gatesum.py $B/runs/gate-jb-*-$SLURM_JOB_ID/gate-*.log
