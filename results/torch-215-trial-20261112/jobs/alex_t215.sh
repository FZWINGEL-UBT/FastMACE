#!/bin/bash -l
# PyTorch 2.15 trial on the A100 (12 November 2026, results/torch-215-trial-20261112): job.sh with a = production
# (fm-omat0small/env: Python 3.10.12, torch 2.14.1+cu130) and b = a Python 3.11.16 venv built here on $TMPDIR with the
# same pins, torch 2.15.0+cu132, Triton 3.9.0 and nvidia-cuda-crt 13.2.86 (OEq's host extension needs crt/ headers,
# which the cu132 wheels no longer pull in). Everything runs on node-local $TMPDIR (atuin file quota); one tarball of
# the records goes back to $B/runs.  Usage: sbatch alex_t215.sh   (src tarball $B/src-b7c19b19.tar, jobs in $B/t215-jobs)
#SBATCH --job-name=fm-t215
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:45:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
O=/home/atuin/v130cb/v130cb18/fm-offpolar
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
L=$TMPDIR/t215; mkdir -p $L/envs $L/src $L/jobs
echo "node $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
export UV_CACHE_DIR=$TMPDIR/uvcache UV_PYTHON_INSTALL_DIR=$L/uv-python UV=/home/hpc/v130cb/v130cb18/.local/bin/uv
t0=$(date +%s)
$E/env/bin/python -m pip freeze 2>/dev/null | grep -v -i -E '^(nvidia|cuda|torch==|triton==|exceptiongroup|tomli)' \
  | grep -v '@ file' > $L/req.txt
grep -i 'nfs.soft.locks' <($E/env/bin/python -m pip freeze 2>/dev/null) | head -1 > $L/req-local.txt
{ $UV python install 3.11.16 && $UV venv --python 3.11.16 --seed $L/envs/mace_cueq_oeq_torch215 &&
  $UV pip install --python $L/envs/mace_cueq_oeq_torch215/bin/python -r $L/req.txt "torch==2.15.0+cu132" \
    nvidia-cuda-crt==13.2.86 --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match; } \
  > $L/build.log 2>&1 || { echo "BUILD failed"; tail -30 $L/build.log; exit 3; }
[ -s $L/req-local.txt ] && $UV pip install --python $L/envs/mace_cueq_oeq_torch215/bin/python -r $L/req-local.txt >> $L/build.log 2>&1
echo "BUILD $(( $(date +%s) - t0 )) s"
ln -s $E/env $L/envs/mace_cueq_oeq
tar -xf $B/src-b7c19b19.tar -C $L/src && echo b7c19b19 > $L/src/.source_commit
cp $B/t215-jobs/* $L/jobs/
export RUN=$L ENVROOT=$L/envs MODELS=$B/models STRUCTURES=$E/baseline PROFILE_C=0 STAGE=all \
  CARD_LIMIT_MIB=78000 RSS_LIMIT_KIB=$(( 200 * 1024 * 1024 )) CUDA_CACHE_DISABLE=1
bash $L/jobs/job.sh
$L/envs/mace_cueq_oeq_torch215/bin/python -m pip freeze > $L/out/freeze-b.txt 2>/dev/null
$E/env/bin/python $L/jobs/analyze.py $L/out a100-alex $L/out > $L/out/analyze.txt 2>&1
cd $L/out && tar -czf $B/runs/t215-$SLURM_JOB_ID.tgz run.log analyze.txt summary-a100-alex.json freeze-b.txt compat-*.json \
  profile/kernels-*.json profile/trace-prof-b.json.gz points/*/point-*.json gate-*/metadata.json *.log 2>/dev/null
echo "DONE $(date +%T) records $B/runs/t215-$SLURM_JOB_ID.tgz"; cat $L/out/analyze.txt
