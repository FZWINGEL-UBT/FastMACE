#!/bin/bash
# 6-T13 item 1: lint and the CPU suite from a freeze archive (one suite at a time on the PC).
#   setsid nohup bash tools/acceptance/phase6_cpu_suite.sh SRC NAME &
src=$(cd "${1:?SRC}" && pwd); name=${2:?NAME}
T=$HOME/tmp/6-T13
if pgrep -f 'python -m pytest' >/dev/null; then echo "another pytest runs; not starting" > "$T/$name.log"; exit 3; fi
cd "$src"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$src FASTMACE_BASELINE=$HOME/fastmace-baseline/1a-T18
~/environments/fastmace_cpu/bin/ruff check fastmace tests > "$T/$name-ruff.log" 2>&1; echo "ruff exit $?" >> "$T/$name-ruff.log"
~/environments/fastmace_cpu/bin/python -m pytest -n 8 tests/cpu tests/gpu -q --junitxml "$T/$name-junit.xml" > "$T/$name.log" 2>&1
echo "pytest exit $?" >> "$T/$name.log"
