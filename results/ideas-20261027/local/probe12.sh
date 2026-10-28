#!/usr/bin/env bash
# ideas: the records through the planner (w_tables, save_transcendentals) and the combination with 56-bit emulation
cd ~/tmp/fm-ideas
S=/home/fzwin/tmp/fm-ideas/src-3988f4be
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=$S:$S/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
M=CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT
log() { echo "$(date -u +%FT%TZ) $*" >> progress12.txt; }
B=$S/tools/bench
for s in NbCl5 NaNbO3; do
  timeout 900 $P $B/f64pkg_acc.py rec-w $s rewrite.w_tables=on > p12-acc-$s-w.log 2>&1; log "acc $s w $?"
  timeout 900 $P $B/f64pkg_acc.py rec-st $s rewrite.save_transcendentals=on > p12-acc-$s-st.log 2>&1; log "acc $s st $?"
  env $M=56 timeout 900 $P $B/f64pkg_acc.py rec-all $s rewrite.w_tables=on rewrite.save_transcendentals=on > p12-acc-$s-all.log 2>&1; log "acc $s all $?"
done
A="--checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
run() { if [ "$2" = all ]; then env $M=56 timeout 900 $P $B/sb64b.py $A --output p12-$1 --structure $3 --rewrites w_tables=on save_transcendentals=on > p12-$1.log 2>&1; else timeout 900 $P $B/sb64b.py $A --output p12-$1 --structure $3 > p12-$1.log 2>&1; fi; log "$1 exit $?"; }
for s in NbCl5 NaNbO3 NaCl; do run $s-a1 pkg $s; run $s-b1 all $s; run $s-b2 all $s; run $s-a2 pkg $s; done
env $M=56 timeout 2400 $P $B/nve_drift.py --structure NaCl --precision float64 --ps 20 --output p12-nve-NaCl-all --rewrites w_tables=on save_transcendentals=on > p12-nve-NaCl-all.log 2>&1; log "nve all $?"
log P12DONE
