#!/bin/bash -l
#SBATCH --job-name=fm-profile1
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-91e428a1; cd $PYTHONPATH
R=$B/runs/profile-polar-$SLURM_JOB_ID; mkdir -p $R
for s in NaCl NbCl5; do
  python tools/module_profile.py --output $R/s-$s --checkpoint $FASTMACE_MODELS/MACE-POLAR-1-S.model --model polar1-s --structure $s --rewrites tf32x3_gemms=on > $R/s-$s.log 2>&1
  python tools/module_profile.py --output $R/m-$s --checkpoint $FASTMACE_MODELS/MACE-POLAR-1-M.model --model polar1-m --structure $s --rewrites tf32x3_gemms=on wprime_backward=on fused_folded_forward=on > $R/m-$s.log 2>&1
  python tools/module_profile.py --output $R/l-$s --checkpoint $FASTMACE_MODELS/MACE-POLAR-1-L.model --model polar1-l --structure $s --rewrites tf32x3_gemms=on wprime_backward=on fused_folded_forward=on > $R/l-$s.log 2>&1
done
ls $R/*
