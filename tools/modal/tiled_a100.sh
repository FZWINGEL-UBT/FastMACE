#!/usr/bin/env bash
# Tiled capacity path (docs/validation/capacity-tiled-t1t2.md) on a Modal A100-80GB: NbCl5 cells growing until one
# fails or is invalid (tiled_capacity.py checks the edges per atom and the forces of every point). The neighbour
# list is built on the host (matscipy, O(N)): the GPU screen is O(N^2), about 200 s at 1.73M atoms on an RTX 4000
# Ada, which would be most of the budget at several million atoms. MH-1 float32 hybrid, R = 4,096.
# Usage (from WSL, Modal client venv; the guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 70 --tarball <tar> \
#     --command "bash tools/modal/tiled_a100.sh" --out ~/fastmace-runs/tiled-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
for rep in ${REPEATS:-24x24x72 28x28x84 32x32x96 34x34x102}; do
  note "start NbCl5 $rep"
  python tools/bench/tiled_capacity.py --structure NbCl5 --repeat "$rep" --receivers 4096 --calls 1 \
    --neighbours gpu_neighbour_list=off --output "$out/points" > "$out/NbCl5-$rep.log" 2>&1 < /dev/null
  code=$?
  note "end NbCl5 $rep exit=$code"
  python - "$out/points" "$rep" >> "$log" 2>&1 <<'EOF'
import glob, json, sys
for path in sorted(glob.glob(sys.argv[1] + "/tiled-NbCl5-*.json")):
    r = json.load(open(path))
    if "x".join(map(str, r["repeat"])) != sys.argv[2]:
        continue
    c = (r.get("calls") or [{}])[-1]
    print(r["atoms"], r["phase"], r.get("valid"), (r.get("error") or "")[:120],
          {k: c.get(k) for k in ("total_peak", "allocated_before", "peak_above_before", "seconds")},
          r.get("check"))
EOF
  grep -q '"phase": "done"' "$out"/points/tiled-NbCl5-*"$(echo "$rep" | awk -Fx '{print 72*$1*$2*$3}')"-*.json 2>/dev/null || break
done
note done
