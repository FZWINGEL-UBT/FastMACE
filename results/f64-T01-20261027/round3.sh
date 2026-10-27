#!/usr/bin/env bash
# float64 radial tables (worktree fm-f64), alone and with emulation and the rounding-class rewrites
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-f64 PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
RW="species_tables=on fold_readout_tail=on fold_product_linear=on"
EMU="env CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="env CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager"
SB="sb64.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
rm -f round3.txt
for s in NaCl NbCl5; do
  $P f64acc.py tables $s hybrid radial_tables=on >> round3.txt 2> acc-$s-tables.err
  $EAGER $P f64acc.py all $s hybrid radial_tables=on $RW >> round3.txt 2> acc-$s-all.err
  $P $SB --output out-$s-tables --structure $s --rewrites radial_tables=on > log3-$s-tables.txt 2>&1

  $EAGER $P $SB --output out-$s-all --structure $s --rewrites radial_tables=on $RW > log3-$s-all.txt 2>&1
  $EMU $P $SB --output out-$s-emuall --structure $s --rewrites radial_tables=on $RW > log3-$s-emuall.txt 2>&1
  echo "$s done" >> round3.txt
done
echo ALLDONE >> round3.txt
