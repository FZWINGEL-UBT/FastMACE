#!/usr/bin/env bash
# ideas D2 step-level probe: float64 package with w_tables attached vs off (forces, then ABBA steps)
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/home/fzwin/tmp/fm-ideas/src-1ac81e02:/home/fzwin/tmp/fm-ideas/src-1ac81e02/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
W=/home/fzwin/tmp/fm-ideas/src-1ac81e02/tools/bench/ideas_d2_probe.py
log() { echo "$(date -u +%FT%TZ) $*" >> progress10.txt; }
for s in NbCl5 NaNbO3; do
  D2_MODE=off timeout 900 $P $W f64pkg_acc.py d2off $s > p10-acc-$s-off.log 2>&1; log "acc $s off $?"
  D2_MODE=on timeout 900 $P $W f64pkg_acc.py d2on $s > p10-acc-$s-on.log 2>&1; log "acc $s on $?"
done
A="--checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
run() { rm -rf p10-$1; D2_MODE=$2 timeout 900 $P $W sb64b.py $A --output p10-$1 --structure $3 > p10-$1.log 2>&1; log "$1 exit $?"; }
for s in NbCl5 NaNbO3 NaCl; do run $s-a1 off $s; run $s-b1 on $s; run $s-b2 on $s; run $s-a2 off $s; done
log P10DONE
