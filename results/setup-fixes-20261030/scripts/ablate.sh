#!/bin/bash
# Setup fixes: one switch off at a time on the after-source; chain-cold, chain-warm (and the size case for F5).
set -u -o pipefail
FM=$1; S=/mnt/c/Users/fzwin/AppData/Local/Temp/claude/D--Margraf-FastMACE--claude-worktrees-opt10-gate-epilogue-1a0a88/02a46571-f82d-48ad-81f9-31b49063be14/scratchpad
BASE=$HOME/fastmace-capacity/setupfix
ST=$FM/results/setup-across-compositions/structures
PY=$HOME/environments/mace_cueq_oeq/bin/python
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models TORCHINDUCTOR_COMPILE_THREADS=4
cache_env() { export FASTMACE_CACHE=$1 TORCHINDUCTOR_CACHE_DIR=$1/inductor TRITON_CACHE_DIR=$1/triton MACE_COMPILE_CACHE=$1/compile XDG_CACHE_HOME=$1/xdg; }
declare -A CIF=([NaCl]=mp-22862_NaCl [NbCl5]=mp-23307_NbCl5 [TaCl5]=mp-29831_TaCl5)
CH=("NaCl=$ST/${CIF[NaCl]}.cif" "NbCl5=$ST/${CIF[NbCl5]}.cif" "TaCl5=$ST/${CIF[TaCl5]}.cif" "NbCl5-0.95=$ST/${CIF[NbCl5]}.cif:0.95")
declare -A OFF=([noF2]=fastmace.model.forces:SPLIT_FORCE_FRAME=False [noF1radial]=fastmace.rewrites.radial:PIECE_BUCKET=1 [noF5]=fastmace.model.compile:MARK_ATOMS_DYNAMIC=False)
for tag in noF2 noF1radial noF5; do
  ROOT=$BASE/$tag; mkdir -p "$ROOT/tmp"; export TMPDIR=$ROOT/tmp
  run() { local name=$1; shift; [ -e "$ROOT/$name" ] && return 0
    echo "$(date -u +%T) start $name" >> "$ROOT/run.log"
    "$PY" -W ignore $S/with_constants.py "${OFF[$tag]}" "$FM/tools/bench/setup_compositions.py" --output "$ROOT/$name" \
      --checkpoint $HOME/models/mace-mh-1.model --label "$tag-$name" --repeat "${REPEAT:-3x3x3}" --structures "$@" > "$ROOT/$name.log" 2>&1
    echo "$(date -u +%T) exit $? $name" >> "$ROOT/run.log"; }
  echo "$(date -u +%T) $tag ${OFF[$tag]} fastmace=$(cat $FM/.source_commit)" >> "$ROOT/run.log"
  cache_env "$ROOT/cache-chain"; run chain-cold "${CH[@]}"; run chain-warm "${CH[@]}"
  if [ $tag = noF5 ]; then
    [ -e "$ROOT/cache-NbCl5" ] || cp -a "$BASE/after/cache-NbCl5" "$ROOT/cache-NbCl5"
    cache_env "$ROOT/cache-NbCl5"; REPEAT=3x3x4 run size-NbCl5-3x3x4 "NbCl5=$ST/${CIF[NbCl5]}.cif"
  fi
done
