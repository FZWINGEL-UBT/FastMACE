#!/bin/bash
# The bundled MH-1 H200 job (fastmace/mh1-catchup, prepared 12 November 2026; NOT launched: the Modal account was at
# $29.49 of its $30 monthly cap). From WSL, at the repository root of a checkout of COMMIT:
#   bash results/mh1-catchup-20261112/h200/launch_h200.sh plan            # check the packs and timeouts (offline)
#   bash results/mh1-catchup-20261112/h200/launch_h200.sh start COMMIT    # launch (only with room under the cap)
# Three containers: the A/B pack on NbCl5 3x3x3 and on NaCl 3x3x3 (a = the H200's MH-1 defaults, 7 entries, both
# build orders each), and one gate pack (default + 3 combos, the 3x float32 gate on NaCl, NaNbO3, NbCl5 3x3x3).
set -euo pipefail
mode=${1:?plan or start}; commit=${2:-}
H=results/mh1-catchup-20261112/h200
common=(--env MODEL=mh1-r2scan --env CHECKPOINT=/data/models/mace-mh-1.model)
run() { if [ "$mode" = start ]; then MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 tools/jobs/launch.py start "$@" --commit "${commit:?COMMIT}" --repo /mnt/d/Margraf/FastMACE; else python3 tools/jobs/launch.py plan "$@"; fi; }
for s in NbCl5 NaCl; do
  run "$H/ab.csv" --batch "mh1-h200-ab-$s" "${common[@]}" --env "STRUCTURE=$s" --env CELL=3x3x3
done
run "$H/gate.csv" --batch mh1-h200-gate "${common[@]}" --env STRUCTURES=NaCl,NaNbO3,NbCl5
