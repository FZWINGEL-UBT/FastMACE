#!/bin/bash -l
# Step attribution on the A100 (conductor, 12 November 2026, results/torch-215-trial-20261112): production OMAT-0 medium
# float32 plan under torch 2.15 + cupti-python (CUDA-graph kernel annotations; native Linux), SrTiO3 10^3 and NbCl5
# 3^3, fresh Inductor caches with comment_origin, 5 annotated steps each; attribution.py runs here, small files back.
#   sbatch alex_attr.sh   (src tarball $B/src-b7c19b19.tar, jobs in $B/t215-jobs)
#SBATCH --job-name=fm-t215attr
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
L=$TMPDIR/t215; mkdir -p $L/envs $L/src $L/jobs $L/tmp
echo "node $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
export UV_CACHE_DIR=$TMPDIR/uvcache UV_PYTHON_INSTALL_DIR=$L/uv-python UV=/home/hpc/v130cb/v130cb18/.local/bin/uv
$E/env/bin/python -m pip freeze 2>/dev/null | grep -v '@ file' | grep -v -i -E '^(nvidia|cuda|torch==|triton==|exceptiongroup|tomli)' > $L/req.txt
$UV python install 3.11.16 > $L/build.log 2>&1
P=$L/envs/mace_cueq_oeq_torch215_prof
$UV venv -q --python 3.11.16 --seed $P && $UV pip install -q --python $P/bin/python -r $L/req.txt "torch==2.15.0+cu132" \
  nvidia-cuda-crt==13.2.86 cupti-python --extra-index-url https://download.pytorch.org/whl/cu132 \
  --index-strategy unsafe-best-match >> $L/build.log 2>&1 || { echo "BUILD failed"; tail -20 $L/build.log; exit 3; }
$UV pip freeze --python $P/bin/python > $P.requirements.txt; grep -i -E "^(torch|triton|cupti)" $P.requirements.txt
tar -xf $B/src-b7c19b19.tar -C $L/src && echo b7c19b19 > $L/src/.source_commit
cp $B/t215-jobs/* $L/jobs/
R=$L/attr; mkdir -p $R
cd $L/src
export ENVROOT=$L/envs PYTHONPATH=$L/src FM_SRC=$L/src FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$E/baseline \
  TMPDIR=$L/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=b7c19b19 REFUSE_FALLBACK=1 \
  MEMORY_FRACTION=0.9 CUDA_CACHE_DISABLE=1 PYSTACKS=1 PROFILE_OUT=$R/profile PROFILE_STEPS=5
for st in SrTiO3:10x10x10 NbCl5:3x3x3; do s=${st%%:*}; c=$R/cache-$s; mkdir -p $c
  FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile \
  XDG_CACHE_HOME=$c/xdg PROFILE_TAG=attr-$s timeout 1500 bash $L/jobs/py-c.sh $L/jobs/profile_point.py point \
    --structure $s --repeat ${st#*:} --model omat0-medium --precision float32 --output $R/points/$s --warmup 200 \
    --steps 200 --timestep-fs 1 --constant "torch._inductor.config:comment_origin=True" > $R/attr-$s.log 2>&1
  echo "$s exit=$? $(grep -a '^point' $R/attr-$s.log | cut -c1-160)"
  $P/bin/python $L/jobs/attribution.py $R/profile/trace-attr-$s.json.gz $R/profile/pystacks-attr-$s.json $c/inductor 5 \
    $R/attribution-$s.json > $R/attribution-$s.log 2>&1; echo "attribution $s exit=$?"
done
cd $L && tar -czf $B/runs/t215attr-$SLURM_JOB_ID.tgz attr/attribution-*.json attr/attribution-*.log attr/attr-*.log \
  attr/points attr/profile envs/*.requirements.txt build.log 2>/dev/null
echo "DONE $(date +%T) records $B/runs/t215attr-$SLURM_JOB_ID.tgz"
