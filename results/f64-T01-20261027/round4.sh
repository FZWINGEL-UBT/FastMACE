#!/usr/bin/env bash
# step times with float64 tables (worktree fm-f64); fresh output directories
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-f64 PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
RW="species_tables=on fold_readout_tail=on fold_product_linear=on"
EMU="env CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="env CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager"
SB="sb64.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
rm -rf out4-* log4-* round4.txt
for s in NaCl NbCl5; do
  $P $SB --output out4-$s-native --structure $s > log4-$s-native.txt 2>&1
  $P $SB --output out4-$s-tables --structure $s --rewrites radial_tables=on > log4-$s-tables.txt 2>&1
  $EAGER $P $SB --output out4-$s-all --structure $s --rewrites radial_tables=on $RW > log4-$s-all.txt 2>&1
  $EMU $P $SB --output out4-$s-emuall --structure $s --rewrites radial_tables=on $RW > log4-$s-emuall.txt 2>&1
  echo "$s done" >> round4.txt
done
$P df64_bench.py > df64.json 2> df64.err
echo ALLDONE >> round4.txt
