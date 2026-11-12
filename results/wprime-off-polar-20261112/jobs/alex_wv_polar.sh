#!/bin/bash -l
# WEIGHTED_VECTOR on POLAR-1-M, NHR@FAU Alex A100-SXM4-80GB (fastmace/wprime-off-polar, 12 November 2026).
# POLAR-1-M is the only OFF / POLAR plan with the w' kernel on the A100 family, and of the 12 November A100 switches
# only WEIGHTED_VECTOR reaches its kernel (MH-1 class: no TABLE_W, so no SKIP_OUTSIDE / HOIST_DX; README).
# 1. tools/jobs/alex_ab_pack.sh (node-local env, source and cache): a = today's defaults (WEIGHTED_VECTOR on),
#    b = WEIGHTED_VECTOR=False; NaCl and NbCl5 3x3x3, both build orders.
# 2. tools/force_stress_gate.py, the 3x float32 gate, on the defaults with WEIGHTED_VECTOR=True stated explicitly.
# Usage: sbatch --output=<OUT>/slurm-%j.out alex_wv_polar.sh <SRC dir> <OUT dir>
#SBATCH --job-name=fm-wvpolar
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:30:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -uo pipefail
SRC=${1:?SRC}; OUT=${2:?OUT}; mkdir -p "$OUT"
R=/home/atuin/v130cb/v130cb18
M=fastmace.kernels.oeq_streaming.wprime_backward
cat > "$OUT/job.env" <<E
FM_ROOT=$R/fm-offpolar
SRC=$SRC
OUT=$OUT
ENV_TAR=$R/fm-alexsetup/env-py310.tar
CACHE=$R/fm-offpolar/cache
PACK="noweighted|$M:WEIGHTED_VECTOR=False"
MODELS="polar1-m=MACE-POLAR-1-M.model"
STRUCTURES="NaCl NbCl5"
WARMUP=200
BLOCK=256
BLOCKS=6
E
bash "$SRC/tools/jobs/alex_ab_pack.sh" "$OUT/job.env"
echo "ab_pack exit $?" >> "$OUT/run.log"
# the staged env and source of alex_ab_pack (node-local)
L=$TMPDIR/fm
site=$(echo "$L"/env/lib/python3.*/site-packages)
export PATH=$L/env/bin:/usr/bin:/bin VIRTUAL_ENV=$L/env PYTHONPATH=$L/src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$L/cache FASTMACE_MODELS=$R/fm-offpolar/models FASTMACE_STRUCTURES=$R/fm-offpolar/baseline
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd "$L/src" || exit 4
python tools/force_stress_gate.py --output "$OUT/gate-weighted" --engine fastmace --model polar1-m \
  --checkpoint "$FASTMACE_MODELS/MACE-POLAR-1-M.model" --structures-root "$FASTMACE_STRUCTURES" \
  --structures NaCl NbCl5 --production-constant "$M:WEIGHTED_VECTOR=True" > "$OUT/gate-weighted.log" 2>&1 < /dev/null
echo "$(date +%FT%T%z) gate exit=$? $(grep -h -iE 'verdict' "$OUT/gate-weighted.log" | tail -n 1 | cut -c1-200)" \
  | tee -a "$OUT/run.log"
rm -rf "$L"
echo "$(date +%FT%T%z) job done" >> "$OUT/run.log"
