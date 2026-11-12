#!/bin/bash
# Offline NVRTC census of the sources the RTX 4000 Ada compiled (matrix DUMP_SOURCES, 12 November 2026): w' backward
# (3 renders), spline forward, sc_generated forward, fused folded forward, each by NVRTC 13.0 / 13.2 / 13.4 / 13.5 for
# sm_89 and sm_120 (CENSUS_ARCHS). CPU only (run on leto, not on the timing host).
set -u
B=/mnt/homes/zwingel/t215; C=$B/census; S=$B/src-b7c19b19; O=$C/out89; mkdir -p $C/src89 $O
cp $B/runs/leeloo/matrix/sources/a/*.cu $C/src89/ && rm -f $C/src89/oeq_projected_forward-*.cu
export CUDA_VISIBLE_DEVICES= TMPDIR=$B/tmp CENSUS_ARCHS="${CENSUS_ARCHS:-89 120}"
PY=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
export PATH=$(dirname $PY)/../lib/python3.10/site-packages/triton/backends/nvidia/bin:$PATH
cd $S
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do
  lib=$C/nvrtc-$v/nvidia/cu13/lib/libnvrtc.so.13
  LD_LIBRARY_PATH=$(dirname $lib) PYTHONPATH=$S $PY -W ignore $B/jobs.p11/census_nvrtc.py $lib $O/census-$v.json $C/src89/*.cu > $O/census-$v.log 2>&1 &
done; wait
$PY - <<'P'
import json
rows = {}
for v in ("13.0.88", "13.2.86", "13.4.92", "13.5.37"):
    for r in json.load(open(f"/mnt/homes/zwingel/t215/census/out89/census-{v}.json"))["rows"]:
        rows.setdefault((r["source"][:46], r["arch"]), {})[v[:4]] = r
for (s, a), d in sorted(rows.items()):
    print(f"{s:46s} sm{a}", "  ".join(f"{v}: R{r.get('registers')} L{r.get('local')} I{r.get('instructions')} F{r.get('ffma')} E{r.get('per_edge')}" + (f" ERR {r['error'][:60]}" if r.get('error') else "") for v, r in d.items()))
P
