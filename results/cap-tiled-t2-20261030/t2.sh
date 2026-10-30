#!/bin/bash
# T2 on murphy: the tiled call keeping linear_up(h1) and skip_tp(h1) instead of h1. Accuracy against the
# whole-system eager call and the compiled default plan, then the NbCl5 and NaCl limits (growing until one
# fails). Each job goes through run.sh (idle check first).
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
for c in eager compiled; do
  bash $R t2-cmp-$c $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --compare $c --rattle 0.05 --strain 0.02 \
    > /dev/null 2>&1
  echo "cmp $c exit $?"
done
bash $R t2-cap-NbCl5-16x16x48 $T --structure NbCl5 --repeat 16x16x48 --receivers 4096 --calls 1 > /dev/null 2>&1
echo "capacity NbCl5 16x16x48 exit $?"
for rep in 18x18x54 19x19x57 20x20x60 21x21x63 22x22x66; do
  bash $R t2-cap-NbCl5-$rep $T --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/t2-cap-NbCl5-$rep/tiled-*.json || break
done
for rep in 50x50x50 54x54x54 57x57x57 60x60x60 63x63x63; do
  bash $R t2-cap-NaCl-$rep $T --structure NaCl --repeat $rep --receivers 4096 --calls 1 > /dev/null 2>&1
  echo "capacity NaCl $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/t2-cap-NaCl-$rep/tiled-*.json || break
done
