#!/usr/bin/env bash
# leto: the tabled-outer-sum fix (stride-0 shape carrier instead of [E, K] floats). Each step alone on the GPU
# (idle wait as leto.sh: no other user above 20 % CPU, nothing on the GPU, two quiet minutes):
#   1. the new GPU test against the old source (expected to fail) and the whole test file on the fixed source;
#   2. memory profiles with the fix: md and single at 36^3;
#   3. capacity points with the fix: SrTiO3 37^3 and 38^3 (capacity_scan.py point, 40-minute limit).
#   ROOT=... OLD=<source> NEW=<source> bash fix.sh
set -u
B=/mnt/homes/zwingel/fm-ada-capacity
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
SITE=$E/lib/python3.10/site-packages
C=$B/cache-$(hostname)
mkdir -p "$ROOT/tmp"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
busy() {
  local gpu users
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  users=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$users" ] && { echo "gpu=[$gpu] users=[$users]"; return 0; }
  return 1
}
wait_idle() {
  local quiet=0 last=
  while [ $quiet -lt 4 ]; do
    if why=$(busy); then
      quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting: $why"; last=${why//[0-9]/}
    else quiet=$((quiet + 1)); fi
    sleep 30
  done
  note "idle"
}
monitor() { ( while true; do echo "$(date -u +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | paste -sd' ' -)"; sleep 30; done ) > "$ROOT/$1.gpu" 2>&1 & echo $!; }
note "fix chain old=$(cat $OLD/.source_commit) new=$(cat $NEW/.source_commit)"
T=tests/gpu/test_round2_kernels_cuda.py
wait_idle
m=$(monitor test-old)
(cd "$OLD" && PYTHONPATH=$OLD python -m pytest -p no:cacheprovider -q "$NEW/$T" -k recomputing --basetemp "$ROOT/tmp/pt-old") > "$ROOT/test-old.log" 2>&1 < /dev/null
note "test on old source exit=$? $(tail -n 1 $ROOT/test-old.log)"; kill $m
m=$(monitor test-new)
(cd "$NEW" && PYTHONPATH=$NEW python -m pytest -p no:cacheprovider -q "$T" --basetemp "$ROOT/tmp/pt-new") > "$ROOT/test-new.log" 2>&1 < /dev/null
note "test file on new source exit=$? $(tail -n 1 $ROOT/test-new.log)"; kill $m
for job in md:36x36x36 single:36x36x36; do
  case=${job%%:*} repeat=${job#*:}
  wait_idle
  m=$(monitor $case-$repeat)
  PYTHONPATH=$NEW python "$NEW/tools/bench/memory_profile.py" --case "$case" --structure SrTiO3 --repeat "$repeat" \
    --output "$ROOT/profiles" > "$ROOT/$case-$repeat.log" 2>&1 < /dev/null
  note "end $case $repeat exit=$? $(grep -m1 'atoms:' "$ROOT/$case-$repeat.log" | cut -c1-300)"; kill $m
done
for repeat in 37x37x37 38x38x38; do
  wait_idle
  m=$(monitor point-$repeat)
  (cd "$NEW" && PYTHONPATH=$NEW timeout 2400 python tools/bench/capacity_scan.py point --structure SrTiO3 \
    --model omat0-medium --precision float32 --output "$ROOT/points" --repeat "$repeat") > "$ROOT/point-$repeat.log" 2>&1 < /dev/null
  note "end point $repeat exit=$? $(tail -n 2 "$ROOT/point-$repeat.log" | tr '\n' ' ' | cut -c1-300)"; kill $m
done
note DONE
