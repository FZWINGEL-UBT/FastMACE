#!/usr/bin/env bash
# fm-ideas probe 1: float64 emulation mantissa (fixed vs dynamic), forces and step; float32 step breakdown
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/FastMACE PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
E="env CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager"
log() { echo "$(date -u +%FT%TZ) $*" >> progress.txt; }
rm -f gemm.jsonl progress.txt
log start
for r in 1 2; do
  $E $P gemm_emu.py eager-dyn-r$r >> gemm.jsonl 2>>gemm.err
  for b in 56 52 48; do $E CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=$b $P gemm_emu.py eager-m$b-r$r >> gemm.jsonl 2>>gemm.err; done
done
log gemm done
STOCK_RW="rewrite.radial_tables=off rewrite.species_tables=off rewrite.fold_readout_tail=off rewrite.fold_product_linear=off"
for s in NbCl5 NaNbO3; do
  $P f64pkg_acc.py stock $s $STOCK_RW engine.fp64_emulation=off >> acc.txt 2> acc-$s-stock.err
  $P f64pkg_acc.py pkg $s >> acc.txt 2> acc-$s-pkg.err
  for b in 56 48; do CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=$b $P f64pkg_acc.py pkg-m$b $s >> acc.txt 2> acc-$s-m$b.err; done
done
log acc done
SB="sb64b.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
for r in 1 2; do for s in NbCl5 NaCl; do
  rm -rf sb-$s-pkg-r$r sb-$s-m56-r$r
  $P $SB --output sb-$s-pkg-r$r --structure $s > sb-$s-pkg-r$r.log 2>&1
  CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56 $P $SB --output sb-$s-m56-r$r --structure $s > sb-$s-m56-r$r.log 2>&1
done; done
log sb64 done
for s in NbCl5 NaNbO3 NaCl; do
  rm -rf sb32-$s
  $P /mnt/d/Margraf/FastMACE/tools/step_breakdown.py --output sb32-$s --checkpoint $HOME/models/mace-mh-1.model --structure $s --temperature-K 300 > sb32-$s.log 2>&1
done
log ALLDONE
