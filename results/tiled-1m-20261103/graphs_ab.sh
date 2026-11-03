#!/bin/bash
# 1M A/B of TileConfig.compiled_graphs (fe7b0a7b), off/on/on/off, same source. graphs_ab.sh HOST
W=/mnt/homes/zwingel/fm-tiled; H=$1; S=$W/src-all-fe7b0a7b
A="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 58x58x58 --mode auto --calls 4 --tile-list --species-index --receivers 16384 --no-budget --sweep-times"
for p in off-a on-a on-b off-b; do
  X=""; case $p in on-*) X="--compiled-graphs";; esac
  SRC=$S bash $W/run.sh $H-graphs-$p $A $X >/dev/null || exit $?
done
for d in $W/runs/$H-graphs-*-?; do python3 - $d <<PY
import json,glob,sys
f=glob.glob(sys.argv[1]+"/tiled-*.json")
if not f: print(sys.argv[1], "no json"); sys.exit()
r=json.load(open(f[0]))
print(sys.argv[1].split("/")[-1], r.get("valid"), (r.get("error") or "")[:300], r.get("tiled",{}).get("energy_eV"), (r.get("check") or {}).get("max_abs_force"))
for c in r.get("calls",[]):
    print("  %.2f s %s R%s peak %.2f alloc %.2f %s" % (c["seconds"], c["mode"], c["receivers"], c["total_peak"]/2**30, c["allocated"]/2**30, c.get("lanes")))
print("  ", {k: round(v, 3) for k, v in (r.get("sweep_times") or {}).get("seconds", {}).items()})
PY
done
