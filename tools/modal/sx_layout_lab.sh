#!/usr/bin/env bash
# sx-layout lab (owner, 7 November 2026): Symmetrix-XL's interaction-2 forward layout with w from a cubic spline,
# from our 6-term table, from a w buffer, or w = 1, against the production fused forward; and the w' backward with
# w and its slope from the spline instead of the table. One process, one capture (OMAT-0 medium, Benchmark_loup
# SrTiO3 10x10x10, 300 equilibration steps), 15 rounds (tools/sx_layout_lab.py, results/sx-layout-lab-20261107).
# Usage (from WSL, Modal client venv, slim tarball):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 10 --tarball <slim.tar> \
#     --command "bash tools/modal/sx_layout_lab.sh" --out <dir>
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
note "start lab"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --structure "$st@10x10x10" --kernels wprime forward --rounds 15 \
  --variants ${VARIANTS-base sx-wbuf sx-table6 sx-spline256 sx-spline2048 sx-spline4096 sx-F-w1 B-spline256 B-spline2048 B-spline4096} \
  --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
note "end lab exit=$?"
grep -h -E "^(forward|wprime) |^sx-|^sx-layout" "$out/lab.log" | cut -c1-260 >> "$log"
note done
