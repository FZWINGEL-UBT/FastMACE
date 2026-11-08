#!/bin/bash -l
#SBATCH --job-name=fm-fusedff-ab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-offpolar/logs/%x-%j.out
unset SLURM_EXPORT_ENV
source /home/atuin/v130cb/v130cb18/fm-offpolar/jobs/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo node $SLURM_JOB_NODELIST commit $(cat .source_commit)
export PYTHONPATH=$B/src-96493263; cd $PYTHONPATH
python -m pytest -q -p no:cacheprovider tests/gpu/test_fourier_cuda.py 2>&1 | tail -3 | sed 's/^/FOURIERTEST /' | tee $B/runs/fourier-test-$SLURM_JOB_ID.txt
grep -q ' failed\| error' $B/runs/fourier-test-$SLURM_JOB_ID.txt && { echo TRACEBACK fourier kernel test failed; exit 1; }
for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model; do
  m=${p%%:*}; c=${p#*:}
  OUT=$B/runs/fusedff-m80-$m-$SLURM_JOB_ID MODEL=$m CKPT=$c STRUCTS="NaCl NbCl5" SETS_FILE=$B/sets/fusedff.tsv BLOCK=400 BLOCKS=12 bash tools/offpolar_ab.sh
done
grep -h "end ab" $B/runs/fusedff-m80-*-$SLURM_JOB_ID/run.log
