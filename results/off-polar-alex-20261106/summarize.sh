#!/usr/bin/env bash
# Regenerates ab_summary.txt and gate_summary.txt from runs/ (run from this folder).
for f in runs/*/ab-*.log; do d=$(basename $(dirname $f) | sed -E 's/-[0-9]{7}$//'); n=$(basename $f .log)
  grep -h "b/a" $f | sed "s#^#$d $n #"; done | grep -v "^smoke" | sort -u > ab_summary.txt
python jobs/gatesum.py runs/gate-*/gate-*.log runs/merged-*/gate-*.log 2>/dev/null | sed -E 's#runs/##; s/-[0-9]{7}//' > gate_summary.txt
# Modal A100-80GB runs (7 November 2026): modal/<job>/out/<kind-set-model>/
for f in modal/*/out/ab-*/ab-*.log; do d=$(basename $(dirname $f)); n=$(basename $f .log)
  grep -h "b/a" $f | sed "s#^#modal $d $n #"; done | sort -u > modal_ab_summary.txt
python jobs/gatesum.py modal/*/out/gate-*/gate-*.log 2>/dev/null | sed -E 's#modal/##' > modal_gate_summary.txt
