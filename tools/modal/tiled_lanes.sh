#!/usr/bin/env bash
# Two sweep-2 lanes on a large GPU (3 November 2026; docs/validation/tiled/tiled-concurrent-tiles.md measured 1.09x per
# sweep-2 pair on the RTX 4000 Ada, where TileConfig.lanes "auto" is on; at 149M its second workspace did not fit
# the 20 GB card). 1M SrTiO3, OMAT-0 medium, the large-cell path (host state aliased, wavefront on, R 8,192),
# compiled, both sides under cudaMallocAsync; lanes off, on, on, off, one process each.
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/tiled_lanes.sh" --out ~/fm-runs/tiled-lanes-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
echo "$(date -u +%FT%TZ) Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)" \
  >> "$out/run.log"
args="--structure SrTiO3 --model omat0-medium --repeat 58x58x58 --mode auto --calls 3 --tile-list --host-state
  --alias-state --species-index --receivers 8192 --no-budget --wavefront on --sweep-times"
for run in lanes-off-a lanes-on-a lanes-on-b lanes-off-b; do
  case $run in
    lanes-off-*) extra="--lanes off" ;;
    lanes-on-*) extra="--lanes on" ;;
  esac
  # shellcheck disable=SC2086
  python tools/bench/tiled_capacity.py $args $extra --output "$out/$run" > "$out/$run.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) $run exit=$? $(tail -n 1 "$out/$run.log" | cut -c1-200)" >> "$out/run.log"
done
python - "$out" <<'PY'
import glob, json, sys
for d in sorted(glob.glob(sys.argv[1] + "/lanes-*")):
    for f in glob.glob(d + "/tiled-*.json"):
        r = json.load(open(f))
        st = r.get("sweep_times") or {}
        print(d.split("/")[-1], r.get("valid"), [(round(c["seconds"], 2), c["mode"], c.get("lanes")) for c in r["calls"]],
              {k: round(v, 2) for k, v in st.get("seconds", {}).items()})
PY
