#!/usr/bin/env bash
# recompute_first_table (rebuild-a) on leeloo (RTX 4000 Ada 20 GB, sm_89): the Ada leg of local.sh (same jobs; see
# there). Launched detached from the PC:
#   ssh -n -f leeloo "setsid nohup env FM=<NAS source copy> bash <copy>/results/rebuild-a-20261031/leeloo.sh ROOT JOB...
#     > ROOT.out 2>&1 < /dev/null &"
# Before each job: no compute process on the GPU and no other user's busy process (root, system accounts and
# zwingel ignored), 6 samples 5 s apart. Everything under /mnt/homes/zwingel (outputs, caches, TMPDIR).
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
H=/mnt/homes/zwingel
B=$H/fastmace-baseline/1a-T18
PY=$H/environments/mace_cueq_oeq/bin/python
PYT=$H/environments/mace_cueq_latest/bin/python
CACHE=${CACHE:-$H/fastmace-cache/rebuild-a-leeloo}
CK=$H/ResearchModule/Models/mace-mh-1.model
mkdir -p "$ROOT/tmp" "$ROOT/ab" "$CACHE"
site=$H/environments/mace_cueq_oeq/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$H/ResearchModule/Models   FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton   MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp   TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
idle() {  # 6 samples, 5 s apart, all idle
  local name=$1 ok=0 tries=0 pids busy util
  while [ $ok -lt 6 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    busy=$(ps -eo user:20,pcpu,comm --sort=-pcpu | awk 'NR > 1 && $1 != "root" && $1 != "zwingel" && $1 !~ /^systemd/       && $1 !~ /^_/ && $2 > 5 {print $1 ":" $3}' | paste -sd, -)
    if [ -z "$pids" ] && [ -z "$busy" ] && [ "$util" -le 10 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1))
    if [ $tries -ge 60 ]; then note "$name NOT IDLE pids=[$pids] busy=[$busy] util=$util"; return 1; fi
    sleep 5
  done
  note "$name idle: util=$util mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | awk '{print $1}' | sort -u | paste -sd, -)"
}
run() {  # name command...
  local name=$1; shift
  idle "$name" || exit 3
  note "start $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "fastmace=$(cat "$FM/.source_commit") cache=$CACHE host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gputest) (cd "$FM" && PYTHONPATH=$FM run $job "$PYT" -m pytest -q -p no:cacheprovider -rs \
        --basetemp="$ROOT/tmp/pytest" tests/gpu/test_kernels_triton_cuda.py) ;;
    gate) PYTHONPATH=$FM run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
        --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
        --reference-command "$PYT" --production-command "$PY" --rewrites recompute_first_table=on ;;
    ledger-*) IFS=- read -r _ structure repeat mode <<< "$job"
      rw=(--rewrites recompute_conv_weights=on recompute_node_segments=on)
      [ "$mode" = table ] && rw+=(recompute_first_table=on) || rw+=(recompute_first_table=off)  # else rule 12 turns it on at the limits
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/memory_ledger.py" --structure "$structure" --repeat "$repeat" \
        --output "$ROOT/ledger" --tag="-$mode" "${rw[@]}" ;;
    ab-*) IFS=- read -r _ structure size order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      spec=$structure; extra=()
      if [ "$size" = big ]; then
        [ "$structure" = NbCl5 ] && spec=NbCl5@4x4x8 || spec=$structure@6x6x6
        extra=(--warmup 100 --block 100 --blocks 8)
      elif [ "$structure" = NaNbO3 ]; then
        spec=NaNbO3@4x4x3
      fi
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/ab_rewrites.py" --b recompute_first_table=on --output "$ROOT/ab/$job" \
        --checkpoint "$CK" --model mh1-r2scan --structures "$spec" --order "$order" --pattern "$pattern" --prepare \
        "${extra[@]}" ;;
    point-*) IFS=- read -r _ structure repeat <<< "$job"
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" \
        --repeat "$repeat" --output "$ROOT/points" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
