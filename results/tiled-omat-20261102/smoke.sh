#!/bin/bash
# Tiled smoke with MACE-OMAT-0 medium on cubic SrTiO3 (a = 3.905 A), plus MH-1 NaCl and NbCl5 rows: the tiled call
# against the whole-system call of the same configuration (rattled, strained), eager against eager and auto against
# the default compiled plan. On murphy (SRC set by the caller); one job at a time through run.sh's idle check.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
TAG=${TAG:-smoke}
ONLY=${ONLY:-}
common="--rattle 0.05 --strain 0.01 --calls 2 --tile-list --host-state --alias-state"
row() {  # name structure model repeat receivers mode compare [extra]
  local name=$1 structure=$2 model=$3 repeat=$4 receivers=$5 mode=$6 compare=$7
  shift 7
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && return
  bash $R murphy-omat-$TAG-$name tools/bench/tiled_capacity.py --structure $structure --model $model \
    --repeat $repeat --receivers $receivers --mode $mode --compare $compare $common "$@" > /dev/null 2>&1
  echo "$name exit $?"
  python3 - /mnt/homes/zwingel/fm-tiled/runs/murphy-omat-$TAG-$name <<'EOF'
import glob, json, sys
files = glob.glob(sys.argv[1] + "/tiled-*.json")
if not files:
    print("  no record")
    sys.exit()
r = json.load(open(files[0]))
ref = r.get("reference") or {}
calls = r.get("calls") or []
last = calls[-1] if calls else {}
print("  atoms", r.get("atoms"), "phase", r.get("phase"), "valid", r.get("valid"), "error", str(r.get("error"))[:300])
print("  mode", last.get("mode"), "R", last.get("receivers"), "tiles", last.get("tiles"), "edges", last.get("edges"),
      "reuse", last.get("list_reuse"), "seconds", [round(c.get("seconds", 0), 2) for c in calls])
print("  vs", ref.get("kind"), "dE meV/atom", ref.get("energy_diff_meV_per_atom"), "max dF", ref.get("max_force_diff"),
      "max |F|", ref.get("max_abs_force"), "max dS", ref.get("max_stress_diff"))
EOF
}
row srtio3-eager SrTiO3 omat0-medium 10x10x10 512 eager eager
row srtio3-auto SrTiO3 omat0-medium 10x10x10 512 auto compiled --species-index
row srtio3-auto-tiles SrTiO3 omat0-medium 12x12x12 1024 auto compiled --species-index --no-budget
row nacl-eager NaCl mh1-r2scan 6x6x6 256 eager eager
row nacl-auto NaCl mh1-r2scan 6x6x6 256 auto compiled --species-index
row nbcl5-eager NbCl5 mh1-r2scan 4x4x12 512 eager eager
row nbcl5-auto NbCl5 mh1-r2scan 4x4x12 512 auto compiled --species-index
