#!/usr/bin/env bash
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/FastMACE PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
RW="species_tables=on fold_readout_tail=on fold_product_linear=on"
EMU="env CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="env CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager"
SB="sb64.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
rm -f round2.txt
for s in NaCl NbCl5; do
  # accuracy (one rattled cell)
  $P f64acc.py cueq $s cueq >> round2.txt 2> acc-$s-cueq.err
  $P f64acc.py native $s hybrid >> round2.txt 2> acc-$s-native.err
  $EMU $P f64acc.py emu $s hybrid >> round2.txt 2> acc-$s-emu.err
  $EAGER $P f64acc.py emueager $s hybrid >> round2.txt 2> acc-$s-emueager.err
  $P f64acc.py rewrites $s hybrid $RW >> round2.txt 2> acc-$s-rewrites.err
  $EMU $P f64acc.py emu-rewrites $s hybrid $RW >> round2.txt 2> acc-$s-emurw.err
  # timing
  $EMU $P $SB --output out-$s-emu --structure $s > log2-$s-emu.txt 2>&1
  $EAGER $P $SB --output out-$s-emueager --structure $s > log2-$s-emueager.txt 2>&1
  $P $SB --output out-$s-rw --structure $s --rewrites $RW > log2-$s-rw.txt 2>&1
  $EMU $P $SB --output out-$s-emurw --structure $s --rewrites $RW > log2-$s-emurw.txt 2>&1
  echo "$s done" >> round2.txt
done
echo ALLDONE >> round2.txt
