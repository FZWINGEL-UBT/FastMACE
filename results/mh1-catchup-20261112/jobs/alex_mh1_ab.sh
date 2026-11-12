#!/bin/bash -l
# MH-1 (mh1-r2scan) A/B pack on an NHR@FAU Alex A100 (fastmace/mh1-catchup, 12 November 2026).
# 1. tools/jobs/alex_ab_pack.sh (node-local env, source and cache): a = today's A100 defaults for MH-1, one b per
#    PACK entry; NaCl and NbCl5 3x3x3, both build orders, 6 x 256 steps per side after 200 warm-up.
# 2. tools/force_stress_gate.py, the 3x float32 gate (NaCl, NaNbO3, NbCl5 3x3x3), once per GATES entry
#    ("name|MODULE:NAME=VALUE ..."; "default|" = today's defaults).
# Usage: sbatch --output=<OUT>/slurm-%j.out alex_mh1_ab.sh <SRC dir> <OUT dir> <PACK> <GATES>
#SBATCH --job-name=fm-mh1ab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:30:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -uo pipefail
SRC=${1:?SRC}; OUT=${2:?OUT}; PACK_ARG=${3:?PACK}; GATES=${4:-}; mkdir -p "$OUT"
R=/home/atuin/v130cb/v130cb18
cat > "$OUT/job.env" <<E
FM_ROOT=$R/fm-offpolar
SRC=$SRC
OUT=$OUT
ENV_TAR=$R/fm-alexsetup/env-py310.tar
CACHE=\$TMPDIR/mh1cache
PACK="$PACK_ARG"
MODELS="mh1-r2scan=../../fm-arena/models/mace-mh-1.model"
STRUCTURES="NaCl NbCl5"
WARMUP=200
BLOCK=256
BLOCKS=6
E
bash "$SRC/tools/jobs/alex_ab_pack.sh" "$OUT/job.env"
echo "ab_pack exit $?" >> "$OUT/run.log"
L=$TMPDIR/fm
site=$(echo "$L"/env/lib/python3.*/site-packages)
export PATH=$L/env/bin:/usr/bin:/bin VIRTUAL_ENV=$L/env PYTHONPATH=$L/src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$L/cache FASTMACE_MODELS=$R/fm-arena/models FASTMACE_STRUCTURES=$R/fm-offpolar/baseline
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd "$L/src" || exit 4
IFS=';' read -r -a gates <<< "$GATES"
for g in "${gates[@]}"; do
  name=${g%%|*}; consts=${g#*|}
  # shellcheck disable=SC2086
  python tools/force_stress_gate.py --output "$OUT/gate-$name" --engine fastmace --model mh1-r2scan \
    --checkpoint "$FASTMACE_MODELS/mace-mh-1.model" --structures-root "$FASTMACE_STRUCTURES" \
    --structures NaCl NaNbO3 NbCl5 ${consts:+--production-constant $consts} > "$OUT/gate-$name.log" 2>&1 < /dev/null
  echo "$(date +%FT%T%z) gate $name exit=$? $(grep -h -iE 'verdict' "$OUT/gate-$name.log" | tail -n 1 | cut -c1-200)" \
    | tee -a "$OUT/run.log"
done
rm -rf "$L"
echo "$(date +%FT%T%z) job done" >> "$OUT/run.log"
