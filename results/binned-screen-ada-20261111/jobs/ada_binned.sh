#!/usr/bin/env bash
# BINNED_SCREEN on the RTX 4000 Ada (11 November 2026, conductor's B2 in plans/next-steps-20261111.md): the 5060 Ti setup
# (results/binned-screen-5060ti-20261111). a = fastmace.neighbors.pair_search:BINNED_SCREEN=False (today's Ada
# default), b = True; OMAT-0 medium, SrTiO3 16^3 and 20^3, both orders, NVT 300 K, 1 fs, skin 0.5 A, warm-up 200,
# 6 alternating blocks of 128 steps; then tools/bench/rebuild_parts.py at both sizes.
# Each job waits for 2 quiet minutes (no GPU process, pytest or other user's busy CPU), samples the same check every
# 15 s while it runs, and is repeated (up to 4 tries) when anything else ran beside it; a tainted try's log is kept.
B=/mnt/homes/zwingel/fm-binned-ada
FM=$B/src
export HOST=ada ROOT=$B/runs/ada-binned
source $B/common.sh
cd $FM && export PYTHONPATH=$FM
K=fastmace.neighbors.pair_search
clean() {  # name cmd...: run until a try has nothing beside it
  local name=$1 try; shift
  for try in 1 2 3 4; do
    QUIET=2 bash $B/queue.sh true > /dev/null
    note "begin $name try $try"; t0=$(date +%s)
    "$@" > "$ROOT/$name.log" 2>&1 < /dev/null & pid=$!
    taint=""
    while kill -0 $pid 2>/dev/null; do
      if w=$(others | sed "s/gpu=\[$pid[^]]*\]/gpu=[]/"); then
        # our own python is the one GPU process: only flag other GPU pids, pytest or busy CPU
        other=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | while read p; do
          [ -n "$p" ] && ! pstree -p $pid 2>/dev/null | grep -q "($p)" && [ "$p" != "$pid" ] && echo $p; done | paste -sd, -)
        suite=$(pgrep -f "[p]ytest" | paste -sd, -)
        busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20' | wc -l)
        [ -n "$other$suite" ] || [ "$busy" != 0 ] && taint="gpu=[$other] pytest=[$suite] busy=$busy"
      fi
      sleep 15
    done
    wait $pid; code=$?
    note "end $name try $try exit=$code wall $(( $(date +%s) - t0 )) s${taint:+ TAINTED $taint}"
    grep -h -E 'b/a' "$ROOT/$name.log" | sed "s/^/  $name try $try /" | tee -a "$log"
    [ $code != 0 ] && { note "STOP: $name failed"; exit 1; }
    [ -z "$taint" ] && return 0
    mv "$ROOT/$name.log" "$ROOT/$name-tainted$try.log"
  done
  note "STOP: $name tainted 4 times"; exit 3
}
note "binned start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for cell in 16x16x16 20x20x20; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    clean "ab-$cell-$order" python tools/ab_rewrites.py --output "$ROOT/ab-$cell-$order" --checkpoint "$CK" \
      --model omat0-medium --structures "$ST@$cell" --warmup 200 --block 128 --blocks 6 --order "$order" \
      --pattern "$pattern" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --a-constant $K:BINNED_SCREEN=False --b-constant $K:BINNED_SCREEN=True
  done
done
clean parts python tools/bench/rebuild_parts.py --structure "$ST" --repeats 16 20 --binned 0 1 --rounds 3 \
  --output "$ROOT/parts.json"
tail -12 "$ROOT/parts.log" | tee -a "$log"
note DONE
