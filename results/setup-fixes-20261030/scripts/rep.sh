#!/bin/bash
# F5 first-build cost: fresh processes on warm caches, alternating after / noF5, NaCl and NbCl5.
set -u -o pipefail
FM=$1; S=/mnt/c/Users/fzwin/AppData/Local/Temp/claude/D--Margraf-FastMACE--claude-worktrees-opt10-gate-epilogue-1a0a88/02a46571-f82d-48ad-81f9-31b49063be14/scratchpad
BASE=$HOME/fastmace-capacity/setupfix; ROOT=$BASE/rep; mkdir -p $ROOT/tmp
ST=$FM/results/setup-across-compositions/structures
PY=$HOME/environments/mace_cueq_oeq/bin/python
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models TORCHINDUCTOR_COMPILE_THREADS=4 TMPDIR=$ROOT/tmp
cache_env() { export FASTMACE_CACHE=$1 TORCHINDUCTOR_CACHE_DIR=$1/inductor TRITON_CACHE_DIR=$1/triton MACE_COMPILE_CACHE=$1/compile XDG_CACHE_HOME=$1/xdg; }
declare -A CIF=([NaCl]=mp-22862_NaCl [NbCl5]=mp-23307_NbCl5)
for i in 1 2 3; do for p in NaCl NbCl5; do for side in after noF5; do
  name=$side-$p-$i; cache_env "$BASE/$side/cache-chain"
  if [ $side = noF5 ]; then C=fastmace.model.compile:MARK_ATOMS_DYNAMIC=False; else C=fastmace.model.compile:MARK_ATOMS_DYNAMIC=True; fi
  echo "$(date -u +%T) start $name" >> $ROOT/run.log
  "$PY" -W ignore $S/with_constants.py $C "$FM/tools/bench/setup_compositions.py" --output "$ROOT/$name" \
    --checkpoint $HOME/models/mace-mh-1.model --label "rep-$name" --repeat 3x3x3 --structures "$p=$ST/${CIF[$p]}.cif" > "$ROOT/$name.log" 2>&1
  echo "$(date -u +%T) exit $? $name" >> $ROOT/run.log
done; done; done
python3 - <<'PY'
import json,glob,os
for f in sorted(glob.glob(os.path.expanduser("~/fastmace-capacity/setupfix/rep/*/setup.json"))):
    s=json.load(open(f))["structures"][0]
    print(f.split("/")[-2], round(s["simulation_s"],2), round(s["stages"]["engine: first compiled force call"]["total_s"],2))
PY
