#!/usr/bin/env bash
# ideas D3: forces with transcendentals saved vs recomputed (float64 package), NaNbO3 ABBA steps
cd ~/tmp/fm-ideas
S=/home/fzwin/tmp/fm-ideas/src-dcc86ed6
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=$S:$S/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
P=~/environments/mace_cueq_oeq/bin/python
W=$S/tools/bench/ideas_d3_norecompute.py
log() { echo "$(date -u +%FT%TZ) $*" >> progress11.txt; }
for s in NbCl5 NaNbO3; do
  D3_MODE=base timeout 900 $P $W f64pkg_acc.py d3base $s > p11-acc-$s-base.log 2>&1; log "acc $s base $?"
  D3_MODE=save timeout 900 $P $W f64pkg_acc.py d3save $s > p11-acc-$s-save.log 2>&1; log "acc $s save $?"
done
A="--checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
run() { D3_MODE=$2 timeout 1200 $P $W $A --output p11-$1 --structure $3 > p11-$1.log 2>&1; log "$1 exit $?"; }
run NaNbO3-a1 base NaNbO3; run NaNbO3-b1 save NaNbO3; run NaNbO3-b2 save NaNbO3; run NaNbO3-a2 base NaNbO3
log P11DONE
