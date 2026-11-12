#!/bin/bash -l
# MH-1 (mh1-r2scan, float32 production plan) baseline on the A100, round 2 (fastmace/mh1-catchup, 12 November 2026).
# Changes from alex_mh1_base.sh (round 1, Alex 4990788): CUPTI >= 13.3 pinned in the torch 2.15 env (4990788 got
# cupti-python / CUPTI 13.2, so record_py_stacks could not register), and OMAT-0 medium NbCl5 3x3x3 profiled and
# attributed beside MH-1 for the comparison.
# 1. Production env (torch 2.14.1, env-py310.tar, node-local): MD step time, NaCl / NaNbO3 / NbCl5 3x3x3 and
#    NbCl5 6x6x6 (capacity_scan point via profile_point.py: 300 warm-up, 1000 timed steps at 0.5 fs, 300 K; then a
#    20-step Kineto window by kernel name), and every NVRTC source the NbCl5 3x3x3 build compiles (offline census).
# 2. torch 2.15.0+cu132 + cupti-python env built in the job (results/torch-215-trial-20261112 README §6a):
#    CUDA-graph kernel attribution (PYSTACKS=1, Inductor comment_origin) of 5 annotated steps, NaCl / NaNbO3 /
#    NbCl5 3x3x3, then attribution.py.
# Everything (env, source, caches) node-local on $TMPDIR; small records go back as one tgz.
#   sbatch alex_mh1_base.sh SHA   (slim source $B/src-SHA.tar; scripts in $B/jobs, t215 tools in fm-arena/t215-jobs)
#SBATCH --job-name=fm-mh1base
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-mh1/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
R=/home/atuin/v130cb/v130cb18
B=$R/fm-mh1
A=$R/fm-arena
SHA=${1:?SHA}
L=$TMPDIR/mh1; mkdir -p $L/src $L/jobs $L/tmp $L/envs
out=$L/base; mkdir -p $out
note() { echo "$(date +%FT%T%z) $*" | tee -a $out/run.log; }
note "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); $(nproc) cores"
tar -xf $B/src-$SHA.tar -C $L/src && echo $SHA > $L/src/.source_commit
cp $A/t215-jobs/* $L/jobs/
# --- production env, staged from the env tar (docs/nhr-cluster.md §6.3)
t0=$(date +%s); tar -xf $R/fm-alexsetup/env-py310.tar -C $L || { note "env tar failed"; exit 4; }
py=$(ls -d $L/uv-python/cpython-3.10.12-linux-x86_64-gnu); ln -sfn $py/bin/python3.10 $L/env/bin/python
sed -i "s#^home = .*#home = $py/bin#" $L/env/pyvenv.cfg
note "prod env staged in $(( $(date +%s) - t0 )) s"
site=$(echo $L/env/lib/python3.*/site-packages)
n=$(nproc)
common() {
  export PYTHONPATH=$L/src FM_SRC=$L/src FASTMACE_MODELS=$A/models FASTMACE_STRUCTURES=$R/fm-offpolar/baseline \
    TMPDIR=$L/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=$SHA REFUSE_FALLBACK=1 \
    MEMORY_FRACTION=0.9 OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
}
cache() { local c=$L/cache-$1; mkdir -p $c
  export FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile \
    XDG_CACHE_HOME=$c/xdg; }
cd $L/src
for mst in mh1-r2scan:NbCl5:3x3x3 mh1-r2scan:NaCl:3x3x3 mh1-r2scan:NaNbO3:3x3x3 mh1-r2scan:NbCl5:6x6x6 \
    omat0-medium:NbCl5:3x3x3; do m=${mst%%:*}; st=${mst#*:}; s=${st%%:*}; r=${st#*:}; tag=prod-$s-$r
  [ $m = mh1-r2scan ] || tag=prod-$m-$s-$r
  ( common; cache prod
    export PATH=$L/env/bin:/usr/bin:/bin VIRTUAL_ENV=$L/env LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
    export PROFILE_OUT=$out/profile PROFILE_STEPS=20 PROFILE_TAG=$tag
    [ "$tag" = prod-NbCl5-3x3x3 ] && export DUMP_SOURCES=$out/sources
    timeout 1800 python $L/jobs/profile_point.py point --structure $s --repeat $r --model mh1-r2scan --precision float32 \
      --output $out/points/$tag --warmup 300 --steps 1000 --timestep-fs 0.5 > $out/$tag.log 2>&1 < /dev/null )
  note "$tag exit=$? $(grep -a '^point' $out/$tag.log | cut -c1-200)"
done
# --- torch 2.15 profiling env (README §6a recipe)
export UV_CACHE_DIR=$L/uvcache UV_PYTHON_INSTALL_DIR=$L/uv-python311 UV=/home/hpc/v130cb/v130cb18/.local/bin/uv
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
$L/env/bin/python -m pip freeze 2>/dev/null | grep -v '@ file' | grep -v -i -E '^(nvidia|cuda|torch==|triton==|exceptiongroup|tomli)' > $L/req.txt
P=$L/envs/mace_cueq_oeq_torch215_prof
t0=$(date +%s)
$UV python install 3.11.16 > $L/build.log 2>&1
$UV venv -q --python 3.11.16 --seed $P && $UV pip install -q --python $P/bin/python -r $L/req.txt "torch==2.15.0+cu132" \
  nvidia-cuda-crt==13.2.86 cupti-python --extra-index-url https://download.pytorch.org/whl/cu132 \
  --index-strategy unsafe-best-match >> $L/build.log 2>&1 && built=1 || built=0
# torch 2.15 cu132 resolves CUPTI 13.2; record_py_stacks needs >= 13.3. The driver is CUDA 13.4 (615.71.10), so 13.4
# first, else 13.5 as on the Ada prof env (torch-215-trial 7081d9d2).
if [ $built = 1 ]; then
  $UV pip install -q --python $P/bin/python --no-deps "cupti-python>=13.3,<13.5" "nvidia-cuda-cupti>=13.3,<13.5" >> $L/build.log 2>&1 \
    || $UV pip install -q --python $P/bin/python --no-deps cupti-python==13.5.0 nvidia-cuda-cupti==13.5.25 >> $L/build.log 2>&1 \
    || built=0
fi
note "t215 env built=$built in $(( $(date +%s) - t0 )) s $(grep -i -c . $L/build.log) log lines"
if [ $built = 1 ]; then
  $UV pip freeze --python $P/bin/python > $out/t215.requirements.txt
  grep -i -E "^(torch|cupti|nvidia-cuda-cupti)" $out/t215.requirements.txt | tr '\n' ' ' | sed 's/^/pins: /' >> $out/run.log
  for mst in mh1-r2scan:NbCl5:3x3x3 mh1-r2scan:NaCl:3x3x3 mh1-r2scan:NaNbO3:3x3x3 omat0-medium:NbCl5:3x3x3; do
    m=${mst%%:*}; st=${mst#*:}; s=${st%%:*}; r=${st#*:}; tag=attr-$s; [ $m = mh1-r2scan ] || tag=attr-$m-$s
    ( common; cache t215
      export ENVROOT=$L/envs PYSTACKS=1 PROFILE_OUT=$out/profile PROFILE_STEPS=5 PROFILE_TAG=$tag
      timeout 1800 bash $L/jobs/py-c.sh $L/jobs/profile_point.py point --structure $s --repeat $r --model $m \
        --precision float32 --output $out/points/$tag --warmup 300 --steps 300 --timestep-fs 0.5 \
        --constant "torch._inductor.config:comment_origin=True" > $out/$tag.log 2>&1 < /dev/null )
    note "$tag exit=$? $(grep -a '^point' $out/$tag.log | cut -c1-200)"
    $P/bin/python $L/jobs/attribution.py $out/profile/trace-$tag.json.gz $out/profile/pystacks-$tag.json \
      $L/cache-t215/inductor 5 $out/attribution-${tag#attr-}.json > $out/attribution-${tag#attr-}.log 2>&1
    note "attribution ${tag#attr-} exit=$?"
  done
else
  tail -30 $L/build.log >> $out/run.log
fi
cp $L/build.log $out/ 2>/dev/null
mkdir -p $B/runs
cd $L && tar -czf $B/runs/mh1base2-$SLURM_JOB_ID.tgz base && note "DONE records $B/runs/mh1base2-$SLURM_JOB_ID.tgz"
cp $out/run.log $B/runs/mh1base2-$SLURM_JOB_ID.run.log
