#!/usr/bin/env bash
# POLAR-1-L with the Ada set without STAGE_DOUT against today's Ada defaults (the set was 1.0033 / 1.0021; leave-one-out:
# every switch within 0.15 %, STAGE_DOUT off 0.9986 / 0.9987), both orders, NbCl5 (11 November 2026).
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-screen
until grep -q DONE $B/runs/ada-polar-confirm/run.log 2>/dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-polarl-confirm
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
o=fastmace.kernels.triton.tabled_outer_sum
SET="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $g:SC_FACTORED=True $w:BUTTERFLY=True $w:STAGE_DOUT=False $w:STAGE_SH=True $w:STAGE_EDGES=True $o:POWER_BACKWARD=True"
note "polarl-confirm start fastmace=$(cat .source_commit)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  if why=$(others); then note "STOP before $order: $why"; exit 3; fi
  note "begin polar1-l $order"; t0=$(date +%s)
  # shellcheck disable=SC2086
  python tools/ab_rewrites.py --output "$ROOT/ab-polar1-l-$order" --checkpoint $MODELS/MACE-POLAR-1-L.model \
    --model polar1-l --structures NbCl5 --warmup 300 --block 200 --blocks 8 --order "$order" --pattern "$pattern" \
    --b-constant $SET > "$ROOT/ab-polar1-l-$order.log" 2>&1 < /dev/null
  note "end polar1-l $order exit=$? wall $(( $(date +%s) - t0 )) s"
  grep -h -E 'b/a' "$ROOT/ab-polar1-l-$order.log" | sed "s/^/  polar1-l $order /" | tee -a "$log"
done
note DONE
