#!/usr/bin/env bash
# OPT4-T03 stage 1 on a Colab A100 (sm_80): tf32x3_gemms against the default plan. Untimed. One VM, stopped at the end
# (Colab uncapped; record the compute units from colab.json). From WSL, in the main checkout:
#   bash results/opt4-T03/colab-a100.sh COMMIT
# Commands, one after another on the VM:
#   1 tests/gpu/test_tf32x3_gemm_cuda.py
#   2 tools/tf32x3_accuracy.py: MH-1 NaCl, NbCl5; OMOL-0 XL NaCl (real operands)
#   3 NVE drift (tools/bench/nve_drift.py, 300 K, 1 fs): MH-1 NaCl and NbCl5 3x3x3, 20 ps, default and tf32x3
#   4 OMOL-0 XL: force gate on NaCl 3x3x3, tf32x3 and default; NVE drift NaCl, 10 ps, default and tf32x3
set -uo pipefail
commit=${1:?commit}
driver=${DRIVER:-tools/colab/fastmace_colab.py}
out=$HOME/fastmace-opt4/t03-colab-a100-$(date -u +%Y%m%dT%H%M)
TEST='python -m pytest -p no:cacheprovider -q -rf tests/gpu/test_tf32x3_gemm_cuda.py > $OUT/pytest.log 2>&1; echo "pytest $?"; tail -3 $OUT/pytest.log'
ACC='python tools/tf32x3_accuracy.py --output $OUT/mh1 --structures NaCl NbCl5 > $OUT/mh1.log 2>&1; echo "mh1 $?"
python tools/tf32x3_accuracy.py --output $OUT/omol --model omol0-xl --structures NaCl > $OUT/omol.log 2>&1; echo "omol $?"'
DRIFT='for s in NaCl NbCl5; do for r in default tf32x3; do
  extra=""; [ $r = tf32x3 ] && extra="--rewrites tf32x3_gemms=on"
  python tools/bench/nve_drift.py --structure $s --ps 20 --output $OUT/$r $extra > $OUT/$s-$r.log 2>&1; echo "$s $r $?"; tail -1 $OUT/$s-$r.log
done; done'
OMOL='ck=$FASTMACE_MODELS/MACE-omol-0-extra-large-1024.model
for r in tf32x3 default; do
  extra=""; [ $r = tf32x3 ] && extra="--rewrites tf32x3_gemms=on"
  python tools/force_stress_gate.py --output $OUT/gate-$r --engine fastmace --checkpoint $ck --model omol0-xl \
    --structures NaCl --structures-root $FASTMACE_STRUCTURES $extra > $OUT/gate-$r.log 2>&1; echo "gate $r $?"; tail -3 $OUT/gate-$r.log
  python tools/bench/nve_drift.py --model omol0-xl --structure NaCl --ps 10 --output $OUT/drift-$r $extra > $OUT/drift-$r.log 2>&1
  echo "drift $r $?"; tail -1 $OUT/drift-$r.log
done'
python3 "$driver" --gpu A100 --commit "$commit" --out "$out" --models mh-1,omol-0-xl --session "fm-a100-opt4t03-$(date -u +%H%M%S)" \
  --timeout-min 120 --command "$TEST" --command "$ACC" --command "$DRIFT" --command "$OMOL"
echo "out=$out"
