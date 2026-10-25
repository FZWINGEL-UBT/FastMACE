#!/bin/bash
# 6-T13 item 6a: plan corpus of fastmace-5 against the freeze.
T=$HOME/tmp/6-T13; F=${FREEZE:-$T/freeze2}
mkdir -p "$T/corpus"; cd "$T"
for t in base "$(basename "$F")"; do
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/$t \
    ~/environments/fastmace_cpu/bin/python "$F/tools/plan_corpus.py" run --root "$T/$t" --output "$T/corpus/$t.json" 2>&1 | tail -2
done
~/environments/fastmace_cpu/bin/python "$F/tools/plan_corpus.py" compare "$T/corpus/base.json" "$T/corpus/$(basename "$F").json" > "$T/corpus/compare.txt"
echo "compare exit $?"; tail -5 "$T/corpus/compare.txt"
