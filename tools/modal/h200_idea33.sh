#!/usr/bin/env bash
# H200 ideas plan, idea 33 (6 November 2026): timing bound for CUTLASS 3.x native Hopper TF32 wgmma GEMMs (x3 for
# 3xTF32, and pre-split single GEMMs) against today's sm_80 mma.sync CUTLASS 3xTF32 GEMMs on the force call's own
# shapes (Benchmark_loup: OMAT-0 medium, SrTiO3 10x10x10). The wgmma kernels come from the prebuilt lab library
# tools/bench/idea33/libidea33_wgmma.so (the job image has no nvcc; build: tools/bench/idea33/build.sh).
# REPEAT (default 2) runs the bench in that many fresh processes.
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/h200_idea33.sh" --out ~/fm-runs/idea33-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(tail -n 1 "$out/$name.log" | cut -c1-300)"
  return $code
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
note "glibc $(ldd --version | head -n 1)"
for t in $(seq 1 "${REPEAT:-2}"); do
  step "bench-$t" python tools/bench/idea33/bench.py --checkpoint "$ck" --structure "$st" --output "$out/idea33-$t.json"
  grep -h -E '"ratio_|verdict' "$out/bench-$t.log" | tr -s ' ' | tr '\n' ' ' | cut -c1-400 >> "$log"
  echo >> "$log"
done
note done
