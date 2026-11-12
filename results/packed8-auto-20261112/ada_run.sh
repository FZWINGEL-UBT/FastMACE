#!/usr/bin/env bash
# Workstation wrapper (RTX 4000 Ada, shared home): the ada-5060-screen environment, the GPU-idle check, then
# cold_ab.sh.  bash ada_run.sh SRC_DIR OUT
set -u
src=$1 out=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=/mnt/homes/zwingel/fm-ada5060/cache-$(hostname)
SITE=$E/lib/python3.10/site-packages
mkdir -p "$out/tmp" "$C"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C \
  FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule \
  TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg \
  TMPDIR=$out/tmp TMP=$out/tmp TEMP=$out/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
apps=$(LC_ALL=C nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
[ -n "$apps" ] && { echo "GPU busy: $apps" >> "$out/run.log"; exit 3; }
cd "$src" && export PYTHONPATH=$src && OUT=$out bash "$src/../jobs/cold_ab.sh"
