#!/usr/bin/env bash
# MACE-OFF / POLAR-1 A/Bs and force gates on a Modal A100-80GB (branch fastmace/off-polar-alex, 7 November 2026):
# the cells that the Alex queue left over. TASKS is a list of kind:set:model, e.g.
#   TASKS="ab:radtf32:off23-large gate:radtf32:off23-medium ab:sharedgeo:polar1-s"
# Every A/B runs in both build orders (tools/offpolar_ab.sh) and stops after 4 blocks per side when b/a is more
# than 3 % from 1 (STOP_AFTER). The structures are those of the Alex jobs: OFF water_216 (gates: urea,
# naphthalene, paracetamol, water_216), POLAR NaCl and NbCl5. Exits 3 on any other card.
# Usage (from WSL, Modal client venv; models and Structures/Organic in the profile's volume):
#   MODAL_PROFILE=fifth modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 180 \
#     --commit <sha> --command 'TASKS="..." bash tools/modal/offpolar_tasks.sh' --out ~/fm-runs/<name>
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "A100.*80GB" || exit 3
ORG=$FASTMACE_STRUCTURES/Structures/Organic
mkdir -p "$OUT/sets"
switch() {  # set -> the b side's ab_rewrites arguments
  case $1 in
    radtf32) echo "--b-constant fastmace.kernels.triton.skinny_linear:RADIAL_TF32X3=True" ;;
    stockskip) echo "--b-constant fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True" ;;
    offtables) echo "--b radial_tables=on" ;;
    e3nnlin) echo "--b-constant fastmace.rewrites.linear:E3NN_BLOCK_LINEAR=True" ;;
    sharedgeo) echo "--b-constant fastmace.longrange.polar:SHARED_GEOMETRY=True" ;;
    *) echo "unknown set $1" >&2; return 1 ;;
  esac
}
gate_args() {  # set -> force_stress_gate.py production arguments of the candidate
  case $1 in
    offtables) echo "--rewrites radial_tables=on" ;;
    *) switch "$1" | sed 's/--b-constant/--production-constant/' ;;
  esac
}
checkpoint() {
  case $1 in
    off23-small) echo MACE-OFF23_small.model ;; off23-medium) echo MACE-OFF23_medium.model ;;
    off23-large) echo MACE-OFF23_large.model ;; off23b-medium) echo MACE-OFF23b_medium.model ;;
    off24-medium) echo MACE-OFF24_medium.model ;; polar1-s) echo MACE-POLAR-1-S.model ;;
    polar1-m) echo MACE-POLAR-1-M.model ;; polar1-l) echo MACE-POLAR-1-L.model ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}
for task in ${TASKS:?}; do
  IFS=: read -r kind set model <<< "$task"
  ckpt=$(checkpoint "$model") || continue
  run=$OUT/$kind-$set-$model
  echo "$(date -u +%FT%TZ) begin $task"
  if [ "$kind" = ab ]; then
    sets=$OUT/sets/$set.tsv  # set before OUT=$run below, which changes $OUT for the rest of that line
    printf '%s\t%s\n' "$set" "$(switch "$set")" > "$sets"
    if [ "${model%%-*}" = polar1 ]; then
      OUT=$run MODEL=$model CKPT=$ckpt STRUCTS="NaCl NbCl5" SETS_FILE=$sets BLOCK=400 BLOCKS=12 \
        STOP_AFTER=${STOP_AFTER:-4} bash tools/offpolar_ab.sh
    else
      OUT=$run MODEL=$model CKPT=$ckpt STRUCTS="$ORG/water_216.extxyz@1x1x1" SETS_FILE=$sets \
        WARMUP=500 BLOCK=1000 BLOCKS=16 STOP_AFTER=${STOP_AFTER:-4} bash tools/offpolar_ab.sh
    fi
    grep -h "b/a" "$run"/ab-*.log | sed "s|^|RATIO $set $model |"
  else
    if [ "${model%%-*}" = polar1 ]; then gs="NaCl NbCl5"
    elif [ "$model" = off23-large ]; then
      gs="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif@2x2x2 $ORG/water_216.extxyz@1x1x1"
    else
      gs="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1"
    fi
    OUT=$run MODEL=$model CKPT=$ckpt STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=default GATE_ARGS="--seed 20261015" \
      bash tools/offpolar_ab.sh
    OUT=$run MODEL=$model CKPT=$ckpt STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=candidate \
      GATE_ARGS="--seed 20261015 $(gate_args "$set")" bash tools/offpolar_ab.sh
    grep -H " summary " "$run"/gate-*.log | sed "s|^|GATE $set $model |" | cut -c1-400
  fi
  echo "$(date -u +%FT%TZ) end $task"
done
