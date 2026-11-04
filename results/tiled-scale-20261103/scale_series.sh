#!/bin/bash
# Where the per-atom gap between the 1M and the 98-149M tiled calls comes from (3 November 2026): SrTiO3, OMAT-0
# medium, compiled, the same source (main b301ee58), one GPU (leto, RTX 4000 Ada). Each run times its last call's
# sweeps (--sweep-times: device seconds of sweeps 1-3, fill and scatter per tile) next to the call's search_s.
#   1M  device state, R 16,384   the reference (the 1M speed pages)
#   1M  device state, R  8,192   the receiver cap the 98-149M calls ran at
#   1M  host state + wavefront, R 8,192  the large-cell state path at 1M
#   20M and 40M  host state, wavefront auto, R 8,192  size alone
# scale_series.sh HOST [STEP...]; every step runs through run.sh (idle check first, stops the series if busy).
W=/mnt/homes/zwingel/fm-tiled; H=$1; shift; S=$W/src-prof-b301ee58
A="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --mode auto --tile-list --species-index --no-budget --sweep-times"
steps=${*:-1m-dev-r16 1m-dev-r8 1m-host-wave-r8 20m-host-r8 40m-host-r8}
for p in $steps; do
  case $p in
    1m-dev-r16) X="--repeat 58x58x58 --calls 3 --receivers 16384 --wavefront off";;
    1m-dev-r8) X="--repeat 58x58x58 --calls 3 --receivers 8192 --wavefront off";;
    1m-host-wave-r8) X="--repeat 58x58x58 --calls 3 --receivers 8192 --host-state --alias-state --wavefront on";;
    20m-host-r8) X="--repeat 159x159x159 --calls 2 --receivers 8192 --host-state --alias-state";;
    40m-host-r8) X="--repeat 200x200x200 --calls 2 --receivers 8192 --host-state --alias-state";;
    *) echo "unknown step $p"; exit 2;;
  esac
  SRC=$S bash $W/run.sh $H-scale-$p $A $X > /dev/null || exit $?
done
for d in $W/runs/$H-scale-*; do python3 - $d <<PY
import json,glob,sys
f=glob.glob(sys.argv[1]+"/tiled-*.json")
if not f: print(sys.argv[1], "no json"); sys.exit()
r=json.load(open(f[0]))
print(sys.argv[1].split("/")[-1], r["atoms"], r.get("valid"), (r.get("error") or "")[:300], (r.get("check") or {}).get("max_abs_force"))
for c in r.get("calls",[]):
    print("  %.2f s %s R%s tiles %s search %.2f tiles_s %.2f wavefront %s peak %.2f" % (c["seconds"], c["mode"],
          c["receivers"], c.get("tiles"), c.get("search_s") or 0, c.get("tiles_s") or 0,
          (c.get("wavefront") or {}).get("on"), c["total_peak"]/2**30))
st=r.get("sweep_times") or {}
print("   wall %.2f timed %.2f" % (st.get("wall_s", 0), st.get("timed_s", 0)), {k: round(v, 2) for k, v in st.get("seconds", {}).items()})
PY
done
