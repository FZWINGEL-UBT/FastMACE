#!/bin/bash -l
# alex_step_limiters v1: tools/modal/step_limiters.sh (what limits each kernel of the MD step: step breakdown plus an
# Nsight Compute pass with the limiter metrics; OMAT-0 medium, SrTiO3 10x10x10, float32, the GPU's default plan) on
# one NHR@FAU Alex A100-SXM4-80GB. Staging as tools/jobs/alex_ab_pack.sh; ncu from the cluster's cuda module.
#SBATCH --job-name=fm-limiters
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#
# Usage, on a frontend:  sbatch --output=<OUT>/slurm-%j.out tools/jobs/alex_step_limiters.sh <job.env>
# <job.env> sets FM_ROOT, SRC, OUT, ENV_TAR, CACHE, STAGE as for alex_ab_pack.sh; NCU_MODULE (default cuda/13.2.0).
# Results: $OUT/run.log, $OUT/main/ (step_breakdown), $OUT/ncu.ncu-rep*, $OUT/ncu_raw.csv; classify offline with
# tools/step_limiters.py.
unset SLURM_EXPORT_ENV
set -uo pipefail
source "${1:?usage: sbatch tools/jobs/alex_step_limiters.sh <job.env>}"
out=${OUT:?set OUT}; mkdir -p "$out"
log=$out/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
module purge >/dev/null 2>&1
root=${FM_ROOT:?set FM_ROOT}
src=${SRC:?set SRC}
py_dir=$(readlink -f "$root/env/bin/python"); py_dir=${py_dir%/bin/*}  # .../uv-python/cpython-X-linux-x86_64-gnu
env_dir=$root/env
if [ "${STAGE:-1}" = 1 ]; then
  local_root=$TMPDIR/fm; mkdir -p "$local_root/uv-python"; t0=$(date +%s)
  if [ -n "${ENV_TAR:-}" ]; then  # one sequential read instead of ~54k small files
    tar -xf "$ENV_TAR" -C "$local_root" && cp -a "$src" "$local_root/src" || { note "STAGE tar failed"; exit 4; }
  else
    cp -a "$py_dir" "$local_root/uv-python/" && cp -a "$env_dir" "$local_root/" && cp -a "$src" "$local_root/src" ||
      { note "STAGE copy failed"; exit 4; }
  fi
  ln -sfn "$local_root/uv-python/$(basename "$py_dir")/bin/$(basename "$(readlink -f "$root/env/bin/python")")" \
    "$local_root/env/bin/python"
  sed -i "s#^home = .*#home = $local_root/uv-python/$(basename "$py_dir")/bin#" "$local_root/env/pyvenv.cfg"
  env_dir=$local_root/env src=$local_root/src
  note "STAGE copied to $local_root in $(( $(date +%s) - t0 )) s"
fi
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=${CACHE:-$root/cache} FASTMACE_MODELS=$root/models FASTMACE_STRUCTURES=$root/baseline
# Thread pools sized to the job's cores (16 per A100; Alex refuses more per GPU)
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
mkdir -p "$FASTMACE_CACHE"
if [ "${STAGE:-1}" = 1 ]; then
  # The staged env is node-local, and every process points <cache>/oeq/cuda-runtime (OEq's stand-in CUDA home: links
  # to the env's nvidia/cu13 headers and libraries) at its own env. Concurrent staged jobs sharing one cache would leave
  # links into another node's $TMPDIR, and OEq's host-extension build fails ("cuda_runtime_api.h: No such file";
  # 4963282, 8 November 2026). So the job gets a node-local cache: links to the shared cache's entries (Inductor,
  # Triton, radial tables stay warm and shared) except oeq/, which is the job's own (its extension build, about 1 min).
  shared=$FASTMACE_CACHE local_cache=$TMPDIR/fm/cache
  mkdir -p "$local_cache/oeq" "$shared/inductor" "$shared/triton" "$shared/radial-tables"
  for entry in "$shared"/* "$shared"/.[!.]*; do
    [ -e "$entry" ] && [ "$(basename "$entry")" != oeq ] && ln -sfn "$entry" "$local_cache/$(basename "$entry")"
  done
  export FASTMACE_CACHE=$local_cache
  note "cache $shared shared through $local_cache, oeq/ per job"
fi
# ncu's directory only: the module's LD_LIBRARY_PATH would put its CUDA runtime before the venv's cu13
ncu_bin=$(module load "${NCU_MODULE:-cuda/13.2.0}" >/dev/null 2>&1; command -v ncu) ||
  { note "no ncu in ${NCU_MODULE:-cuda/13.2.0}"; exit 5; }
export PATH=$PATH:$(dirname "$ncu_bin")
cd "$src" || exit 4
note "alex_step_limiters v1 job ${SLURM_JOB_ID:-?} on $(hostname); ncu $ncu_bin"
OUT=$out bash tools/modal/step_limiters.sh
code=$?
note "step_limiters exit=$code"
exit $code
