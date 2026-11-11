#!/usr/bin/env bash
# POLAR-1-L NbCl5 leave-one-out of the Ada set (conductor, 11 November 2026): with the whole set on, POLAR-1-L NbCl5
# was slower than today's Ada defaults in both orders (1.0033 / 1.0021). Only the four switches that fire for POLAR
# (models-firing.md): a = the set; b = the set with one off (b/a < 1: it slows this model). After ada_polar_loo.sh.
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-screen
until grep -q DONE $B/runs/ada-polar-loo/run.log 2>/dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-polarl-loo
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
o=fastmace.kernels.triton.tabled_outer_sum
SET="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $g:SC_FACTORED=True $w:BUTTERFLY=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $o:POWER_BACKWARD=True"
P1="no-sbl|$k:SORT_BY_LENGTH=False;no-bfly|$w:BUTTERFLY=False;no-sdout|$w:STAGE_DOUT=False;no-ssh|$w:STAGE_SH=False"
note "polarl-loo start fastmace=$(cat .source_commit)"
for p in P1; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $p $order: $why"; exit 3; fi
    note "begin $p $order"; t0=$(date +%s)
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$ROOT/ab-$p-$order" --checkpoint $MODELS/MACE-POLAR-1-L.model \
      --model polar1-l --structures NbCl5 --warmup 300 --block 200 --blocks 8 --order "$order" --pattern "$pattern" \
      --a-constant $SET --b-constant $SET --pack "${!p}" > "$ROOT/ab-$p-$order.log" 2>&1 < /dev/null
    note "end $p $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$ROOT/ab-$p-$order.log" | sed "s/^/  $p $order /" | tee -a "$log"
  done
done
note DONE
