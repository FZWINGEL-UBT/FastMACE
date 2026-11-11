#!/usr/bin/env bash
# The 5060 Ti set F (F_MODELS) on the other models, as ada_models.sh on the Ada: a = today's 5060 Ti defaults, b = F,
# both orders, POLAR-1-M (NaCl, NbCl5) and OFF23-medium (water_216, W); gates with F for both. Inside local_gpu.py.
#   HOST=local ROOT=<dir> F_MODELS=... W=<water_216.extxyz> bash local_models.sh SRC_DIR
FM=$1
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
SET=$F_MODELS
note "models start fastmace=$(cat .source_commit)"
run() {  # NAME MODEL CKFILE WARMUP BLOCK BLOCKS STRUCTURE...
  local name=$1 model=$2 ck=$MODELS/$3 warm=$4 block=$5 blocks=$6; shift 6
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
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
bash "$(dirname "${BASH_SOURCE[0]}")/gate.sh" $FM polar1-m polar1-m MACE-POLAR-1-M.model "NaCl NbCl5" "3 3 3" $SET
# shellcheck disable=SC2086
bash "$(dirname "${BASH_SOURCE[0]}")/gate.sh" $FM off23-medium off23-medium MACE-OFF23_medium.model "$W@1x1x1" "1 1 1" $SET
note DONE
