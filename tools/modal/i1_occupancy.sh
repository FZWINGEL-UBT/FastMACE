#!/usr/bin/env bash
# Edges per (receiver, sender species, piece) group of the interaction tables on a rattled SrTiO3 10x10x10
# (tools/i1_occupancy.py; CPU only). Environment: OUT (the job sets it).
set -uo pipefail
out=${OUT:?the job sets OUT}
mkdir -p "$out"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
python tools/i1_occupancy.py --checkpoint "$ck" --output "$out/occupancy.json" --threads "${SLURM_CPUS_ON_NODE:-8}" \
  2>&1 | tee "$out/occupancy.log"
exit "${PIPESTATUS[0]}"
