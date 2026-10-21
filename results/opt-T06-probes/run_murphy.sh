#!/usr/bin/env bash
# OPT-T06 on murphy, untimed: run_murphy.sh RUN_NAME
# The copy-vs-reload probe on MH-1 for hybrid and plain cuEq in float32 and float64 (one process each), then the
# GPU test file on the tiny model. Everything under /mnt/homes/zwingel (ResearchModule AGENTS.md). Refuses to start,
# and stops between processes, while any other compute process is on the GPU.
set -uo pipefail
name=$1
work=/mnt/homes/zwingel/fastmace-opt-t06
src=$work/src
run=$work/runs/$name
cache=$work/cache
oeq=/mnt/homes/zwingel/environments/mace_cueq_oeq
latest=/mnt/homes/zwingel/environments/mace_cueq_latest
mkdir -p "$run/tmp" "$cache"
busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -q .; }
check() {
  { echo "utc $(date -u +%FT%TZ) host $(hostname) $1"; uptime
    nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; } >> "$run/idle_check.txt"
  if busy; then echo "GPU busy before $1; stopping" | tee -a "$run/idle_check.txt" >&2; exit 3; fi
}
echo "source commit $(cat "$src/.source_commit")" > "$run/idle_check.txt"
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton MACE_COMPILE_CACHE=$cache/compile
export XDG_CACHE_HOME=$cache/xdg TMPDIR=$run/tmp TMP=$run/tmp TEMP=$run/tmp TORCHINDUCTOR_COMPILE_THREADS=4
site=$oeq/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
cd "$run"
for backend in hybrid cueq; do
  for dtype in float32 float64; do
    check "probe $backend $dtype"
    "$oeq/bin/python" "$src/results/opt-T06-probes/gpu_copy_vs_load.py" --checkpoint "$checkpoint" \
      --label mh1-r2scan --backend "$backend" --dtype "$dtype" --engine-steps 10 \
      --json "$run/mh1-$backend-$dtype.json" > "$run/mh1-$backend-$dtype.log" 2>&1
    echo "probe $backend $dtype exit $?" >> "$run/idle_check.txt"
  done
done
check "pytest"
FASTMACE_HYBRID_PYTHON=$oeq/bin/python "$latest/bin/python" -m pytest -p no:cacheprovider -rf -q \
  --basetemp "$run/pytest" "$src/tests/gpu/test_template_cache_gpu.py" > "$run/pytest.log" 2>&1
echo "pytest exit $?" >> "$run/idle_check.txt"
echo "done $(date -u +%FT%TZ)" >> "$run/idle_check.txt"
