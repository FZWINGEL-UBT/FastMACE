#!/usr/bin/env bash
# POLAR-1-M NaCl leave-one-out of the Ada set (conductor, 11 November 2026): with the whole set on, POLAR-1-M NaCl was
# slower than today's Ada defaults in both orders (1.0061 / 1.0077). a = the set; b = the set with one switch off
# (b/a < 1: that switch slows this model). Packs of at most four, both orders. On murphy after ada_models.sh, src-screen.
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-screen
until grep -q DONE $B/runs/ada-models/run.log 2>/dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-polar-loo
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
o=fastmace.kernels.triton.tabled_outer_sum
SET="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $g:SC_FACTORED=True $w:BUTTERFLY=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $o:POWER_BACKWARD=True"
P1="no-p8|$s:PACKED8=False;no-sbl|$k:SORT_BY_LENGTH=False;no-scg|$g:SC_GENERATED=False $g:SC_FACTORED=False;no-scf|$g:SC_FACTORED=False"
P2="no-bfly|$w:BUTTERFLY=False;no-sdout|$w:STAGE_DOUT=False;no-ssh|$w:STAGE_SH=False;no-wse|$w:STAGE_EDGES=False"
P3="no-power|$o:POWER_BACKWARD=False"
note "polar-loo start fastmace=$(cat .source_commit)"
for p in P1 P2 P3; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    if why=$(others); then note "STOP before $p $order: $why"; exit 3; fi
    note "begin $p $order"; t0=$(date +%s)
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$ROOT/ab-$p-$order" --checkpoint $MODELS/MACE-POLAR-1-M.model \
      --model polar1-m --structures NaCl --warmup 300 --block 200 --blocks 8 --order "$order" --pattern "$pattern" \
      --a-constant $SET --b-constant $SET --pack "${!p}" > "$ROOT/ab-$p-$order.log" 2>&1 < /dev/null
    note "end $p $order exit=$? wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$ROOT/ab-$p-$order.log" | sed "s/^/  $p $order /" | tee -a "$log"
  done
done
note DONE
