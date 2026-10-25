#!/bin/bash
# 6-T13 item 5: the examples' smoke commands (examples/README.md) on the RTX 5060 Ti, from the freeze archive.
# Run under the reservation: python3 tools/ci/local_gpu.py run --purpose "6-T13 examples" -- bash ...
set -u
src=$(cd "${1:?SRC}" && pwd)
out=$HOME/tmp/6-T13/examples; mkdir -p "$out"
cache=$HOME/fastmace-cache/6-T13; mkdir -p "$cache"/{tmp,fastmace,inductor,triton}
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace \
  TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton FASTMACE_MODELS=$HOME/models
M=$HOME/models/mace-mh-1.model S=$HOME/fastmace-baseline/1a-T18/Structures/Benchmark/R2SCAN
PY=$HOME/environments/mace_cueq_oeq/bin/python
cd "$src"
fail=0
ex() { local name=$1; shift; local t0=$(date +%s)
  "$PY" "examples/$name" --output "$out/${name%.py}" "$@" > "$out/${name%.py}.log" 2>&1; local s=$?
  echo "$name exit=$s wall=$(( $(date +%s) - t0 ))s"; [ $s = 0 ] || fail=1; }
ex 01_nvt_anneal_nacl.py --structure $S/NaCl.cif --checkpoint $M --steps 200
ex 02_npt_d3_melt_quench_nbcl5.py --structure $S/NbCl5.cif --checkpoint $M --steps 50 --relax-steps 20
ex 03_swap_mc_anneal.py --checkpoint $M --steps 700 --repeat 1 1 1
ex 04_gpu_relaxations.py --structure $S/NaCl.cif --checkpoint $M --steps 50
ex 05_port_ase_script.py --structure $S/NaCl.cif --checkpoint $M --steps 100
echo "examples $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
