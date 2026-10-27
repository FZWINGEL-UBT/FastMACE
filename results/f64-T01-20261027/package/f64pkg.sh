#!/usr/bin/env bash
# Decision 63 package check (fm-f64b): accuracy (forces, stress), MD step times and NVE drift, package against stock.
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-f64b PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
STOCK_RW="rewrite.radial_tables=off rewrite.species_tables=off rewrite.fold_readout_tail=off rewrite.fold_product_linear=off"
STOCK_SB="radial_tables=off species_tables=off fold_readout_tail=off fold_product_linear=off"
SB="sb64b.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
O=pkg-results; mkdir -p $O; rm -f $O/summary.txt
for s in NaCl NbCl5 NaNbO3; do
  $P f64pkg_acc.py stock $s $STOCK_RW engine.fp64_emulation=off >> $O/summary.txt 2> $O/acc-$s-stock.err
  $P f64pkg_acc.py pkg $s >> $O/summary.txt 2> $O/acc-$s-pkg.err
  rm -rf $O/sb-$s-stock $O/sb-$s-pkg
  $P $SB --output $O/sb-$s-stock --structure $s --rewrites $STOCK_SB --engine fp64_emulation=off > $O/sb-$s-stock.log 2>&1
  $P $SB --output $O/sb-$s-pkg --structure $s > $O/sb-$s-pkg.log 2>&1
  echo "$s done" >> $O/summary.txt
done
T=/mnt/d/Margraf/fm-f64b/tools/bench/nve_drift.py
$P $T --structure NaCl --precision float64 --ps 20 --output $O/nve-NaCl-stock --rewrites $STOCK_SB --engine fp64_emulation=off > $O/nve-NaCl-stock.log 2>&1
$P $T --structure NaCl --precision float64 --ps 20 --output $O/nve-NaCl-pkg > $O/nve-NaCl-pkg.log 2>&1
$P $T --structure NbCl5 --precision float64 --ps 3 --output $O/nve-NbCl5-stock --rewrites $STOCK_SB --engine fp64_emulation=off > $O/nve-NbCl5-stock.log 2>&1
$P $T --structure NbCl5 --precision float64 --ps 3 --output $O/nve-NbCl5-pkg > $O/nve-NbCl5-pkg.log 2>&1
echo ALLDONE >> $O/summary.txt
