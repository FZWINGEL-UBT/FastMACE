#!/bin/bash
# Local RTX 5060 Ti: compiled tiles with and without compiled_graphs, SrTiO3 20x20x20.
S=/mnt/d/Margraf/FastMACE/.claude/worktrees/tiled-backend-fa577c
OUT=~/fm-runs/graphs-smoke; mkdir -p $OUT
ENV=~/environments/mace_cueq_oeq; site=$ENV/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$S
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule FASTMACE_MODELS=~/models FASTMACE_CACHE=~/fm-cache
export TORCHINDUCTOR_CACHE_DIR=~/fm-cache/inductor TRITON_CACHE_DIR=~/fm-cache/triton TMPDIR=~/tmp
cd $S
for p in plain graphs; do
  X=""; [ $p = graphs ] && X="--compiled-graphs"
  rm -rf $OUT/$p
  $ENV/bin/python tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 20x20x20 --mode compiled \
    --calls 4 --tile-list --species-index --receivers 4096 --no-budget $X --output $OUT/$p > $OUT/$p.log 2>&1
  echo "$p exit $?"; tail -3 $OUT/$p.log
done
python3 - <<PY
import json,glob
for p in ("plain","graphs"):
    f=glob.glob("/home/fzwin/fm-runs/graphs-smoke/%s/tiled-*.json"%p)
    if not f: print(p,"no json"); continue
    r=json.load(open(f[0]))
    print(p, r.get("valid"), (r.get("error") or "")[:300], r.get("tiled",{}).get("energy_eV"), r.get("check"))
    for c in r.get("calls",[]): print("  %.3f s %s R%s peak %.2f lanes %s" % (c["seconds"], c["mode"], c["receivers"], c["total_peak"]/2**30, c.get("lanes")))
PY
