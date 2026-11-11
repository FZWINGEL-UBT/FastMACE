#!/usr/bin/env bash
# The Ada set on the other models (conductor, 11 November 2026): one A/B of the whole set against today's Ada defaults
# (a = defaults, b = the set), both orders, on POLAR-1-M (NaCl, NbCl5), MACE-OFF23 medium (water_216) and POLAR-1-L
# (NbCl5); the 3x float32 gate with the set on for POLAR-1-M (NaCl, NbCl5 3x3x3) and OFF23 medium (water_216).
# On murphy after ada_final.sh, on src-screen.
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-screen
until grep -q DONE $B/runs/ada-final/run.log 2>/dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-models
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
SET="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $g:SC_FACTORED=True $w:BUTTERFLY=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD=True"
W=$STRUCT/Structures/Organic/water_216.extxyz
note "models start fastmace=$(cat .source_commit)"
run() {  # NAME MODEL CKFILE WARMUP BLOCK BLOCKS STRUCTURE...
  local name=$1 model=$2 ck=$MODELS/$3 warm=$4 block=$5 blocks=$6; shift 6
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $name $order: $why"; exit 3; fi
    note "begin $name $order"; t0=$(date +%s)
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$ROOT/ab-$name-$order" --checkpoint "$ck" --model "$model" \
      --structures "$@" --warmup $warm --block $block --blocks $blocks --order "$order" --pattern "$pattern" \
      --b-constant $SET > "$ROOT/ab-$name-$order.log" 2>&1 < /dev/null
    note "end $name $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$ROOT/ab-$name-$order.log" | sed "s/^/  $name $order /" | tee -a "$log"
  done
}
run polar1-m polar1-m MACE-POLAR-1-M.model 300 200 8 NaCl NbCl5
run off23-medium off23-medium MACE-OFF23_medium.model 500 1000 16 "$W@1x1x1"
# shellcheck disable=SC2086
bash $B/jobs/gate.sh $FM polar1-m polar1-m MACE-POLAR-1-M.model "NaCl NbCl5" "3 3 3" $SET
# shellcheck disable=SC2086
bash $B/jobs/gate.sh $FM off23-medium off23-medium MACE-OFF23_medium.model "$W@1x1x1" "1 1 1" $SET
run polar1-l polar1-l MACE-POLAR-1-L.model 300 200 8 NbCl5
note DONE
