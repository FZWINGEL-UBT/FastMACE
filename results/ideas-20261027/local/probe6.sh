#!/usr/bin/env bash
# ideas D1 local checks: ABBA step runs (separate processes; the emulation setting is process-wide) and 20 ps NaCl NVE drift
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-ideas:/mnt/d/Margraf/fm-ideas/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
M=CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT
log() { echo "$(date -u +%FT%TZ) $*" >> progress6.txt; }
SB="sb64b.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
run() { rm -rf p6-$1; if [ "$2" = m56 ]; then env $M=56 timeout 900 $P $SB --output p6-$1 --structure $3 > p6-$1.log 2>&1; else timeout 900 $P $SB --output p6-$1 --structure $3 > p6-$1.log 2>&1; fi; log "$1 exit $?"; }
for s in NbCl5 NaNbO3 NaCl; do
  run $s-a1 pkg $s; run $s-b1 m56 $s; run $s-b2 m56 $s; run $s-a2 pkg $s
done
T=/mnt/d/Margraf/fm-ideas/tools/bench/nve_drift.py
timeout 2400 $P $T --structure NaCl --precision float64 --ps 20 --output p6-nve-NaCl-pkg > p6-nve-NaCl-pkg.log 2>&1; log "nve pkg exit $?"
env $M=56 timeout 2400 $P $T --structure NaCl --precision float64 --ps 20 --output p6-nve-NaCl-m56 > p6-nve-NaCl-m56.log 2>&1; log "nve m56 exit $?"
log P6DONE
