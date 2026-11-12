#!/bin/bash -l
# CUDA library matrix on the A100 (12 November 2026, results/torch-215-trial-20261112): matrix.sh's variants (production
# 2.14.1; cuBLAS 13.4.1 / 13.8.2; torch 2.14.1 with Triton 3.9 on Python 3.11; our kernels by NVRTC 13.2 / 13.4 / 13.5;
# torch 2.15; torch 2.15 with NVRTC 13.0), then job.sh's compat probe for a and b, the 3x gate for b, t9, c4 and c8
# against the production reference, and the offline SASS
# census of the sources the A100 plan compiled. Everything on node-local $TMPDIR (atuin file quota); one tarball back.
#   sbatch alex_matrix.sh   (src tarball $B/src-b7c19b19.tar, jobs in $B/t215-jobs)
#SBATCH --job-name=fm-t215mx
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
L=$TMPDIR/t215; mkdir -p $L/envs $L/src $L/jobs $L/nv
echo "node $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
export UV_CACHE_DIR=$TMPDIR/uvcache UV_PYTHON_INSTALL_DIR=$L/uv-python UV=/home/hpc/v130cb/v130cb18/.local/bin/uv
t0=$(date +%s)
$E/env/bin/python -m pip freeze 2>/dev/null | grep -v '@ file' > $L/prod-freeze.txt
grep -v -i -E '^(nvidia|cuda|torch==|triton==|exceptiongroup|tomli)' $L/prod-freeze.txt > $L/req.txt
build() { local name=$1 py=$2; shift 2
  $UV venv -q --python $py --seed $L/envs/$name && $UV pip install -q --python $L/envs/$name/bin/python "$@" \
    >> $L/build.log 2>&1 || { echo "BUILD $name failed"; tail -20 $L/build.log; return 1; }
  echo "BUILD $name $(( $(date +%s) - t0 )) s"; }
$UV python install 3.11.16 3.10.12 >> $L/build.log 2>&1
build mace_cueq_oeq_torch215 3.11.16 -r $L/req.txt "torch==2.15.0+cu132" nvidia-cuda-crt==13.2.86 \
  --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match || exit 3
for v in 134:13.4.1.3 138:13.8.2.12; do
  sed "s/^nvidia-cublas==.*/nvidia-cublas==${v#*:}/" $L/prod-freeze.txt > $L/req-cublas${v%%:*}.txt
  build mace_cueq_oeq_cublas${v%%:*} 3.10.12 -r $L/req-cublas${v%%:*}.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match || exit 3
done
grep -v -i -E '^(exceptiongroup|tomli)==' $L/prod-freeze.txt > $L/req-t9.txt
build mace_cueq_oeq_triton39 3.11.16 -r $L/req-t9.txt --extra-index-url https://download.pytorch.org/whl/cu130   --index-strategy unsafe-best-match || exit 3
$UV pip install -q --python $L/envs/mace_cueq_oeq_triton39/bin/python --no-deps triton==3.9.0 >> $L/build.log 2>&1
for e in mace_cueq_oeq_torch215 mace_cueq_oeq_triton39 mace_cueq_oeq_cublas134 mace_cueq_oeq_cublas138; do
  $UV pip freeze --python $L/envs/$e/bin/python > $L/envs/$e.requirements.txt; done
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do
  $UV pip install -q --python $E/env/bin/python --target $L/nv/nvrtc-$v --no-deps nvidia-cuda-nvrtc==$v >> $L/build.log 2>&1
done
ln -s $E/env $L/envs/mace_cueq_oeq
tar -xf $B/src-b7c19b19.tar -C $L/src && echo b7c19b19 > $L/src/.source_commit
cp $B/t215-jobs/* $L/jobs/
export RUN=$L NV=$L/nv ENVROOT=$L/envs MODELS=$B/models STRUCTURES=$E/baseline PROFILE_C=0 \
  CARD_LIMIT_MIB=78000 RSS_LIMIT_KIB=$(( 200 * 1024 * 1024 )) CUDA_CACHE_DISABLE=1
bash $L/jobs/matrix.sh
STAGE=compat bash $L/jobs/job.sh
cd $L/src
for e in b t9 c4 c8; do
  C=$L/cache/gate-$e; export FASTMACE_CACHE=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
    MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg
  bash $L/jobs/py-a.sh tools/force_stress_gate.py --output $L/matrix/gate-$e --engine fastmace --model omat0-medium \
    --checkpoint $B/models/mace-omat-0-medium.model --structures-root $STRUCTURES \
    --structures "results/loup-a100-20261103/SrTiO3.cif@3x3x3" NbCl5 \
    --reference-command "bash $L/jobs/py-a.sh" --production-command "bash $L/jobs/py-$e.sh" > $L/matrix/gate-$e.log 2>&1
  echo "gate $e exit=$?"; grep -h "summary" $L/matrix/gate-$e.log | cut -c1-300
done
export PATH=$(echo $E/env/lib/python3.*/site-packages/triton/backends/nvidia/bin):$PATH
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do
  lib=$L/nv/nvrtc-$v/nvidia/cu13/lib/libnvrtc.so.13
  LD_LIBRARY_PATH=$(dirname $lib) CUDA_VISIBLE_DEVICES= PYTHONPATH=$L/src $E/env/bin/python -W ignore \
    $L/jobs/census_nvrtc.py $lib $L/matrix/census-a100-$v.json $(ls $L/matrix/sources/a/*.cu) > $L/matrix/census-$v.log 2>&1
done
cd $L && tar -czf $B/runs/t215mx-$SLURM_JOB_ID.tgz matrix/run.log matrix/points matrix/profile/kernels-*.json \
  matrix/profile/trace-prof-a.json.gz matrix/profile/trace-prof-b.json.gz matrix/profile/trace-prof-c8.json.gz \
  matrix/sources matrix/census-*.json envs/*.requirements.txt matrix/gate-*/metadata.json matrix/gate-*.log out/compat-*.json out/run.log build.log 2>/dev/null
echo "DONE $(date +%T) records $B/runs/t215mx-$SLURM_JOB_ID.tgz"
