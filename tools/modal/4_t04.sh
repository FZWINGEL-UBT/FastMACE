#!/usr/bin/env bash
# 4-T04 on Modal (owner's second batch, 23 October 2026). One argument:
# - "trajectories": tests/gpu/test_fix_atoms_gpu.py::test_trajectories_with_fixed_atoms_are_bitwise_equal[nve-csvr]
#   (NVE and CSVR with FixAtoms, pipelined against checked), one pytest process, junit and log in $OUT;
# - "pairs": tools/modal/4_t04_pairs.py, pipelined against checked without FixAtoms, one JSON per run in
#   $OUT/pairs (run it on a tarball of main with this script and 4_t04_pairs.py added).
# Usage (from WSL, Modal client venv; guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/4_t04.sh trajectories" --out ~/fastmace-runs/4-t04/modal-L4-trajectories \
#     --limit-usd <the shared ceiling>
set -uo pipefail
out=${OUT:?the job sets OUT}
export FASTMACE_SOURCE_COMMIT=$(cat .source_commit)
export FASTMACE_RESULTS=$out
log=$out/run.log
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $gpu; source $FASTMACE_SOURCE_COMMIT host $(hostname)"
case "${1:?trajectories or pairs}" in
  trajectories)
    note "start pytest"
    python -m pytest -s -p no:cacheprovider \
      "tests/gpu/test_fix_atoms_gpu.py::test_trajectories_with_fixed_atoms_are_bitwise_equal[nve-csvr]" \
      --junitxml "$out/junit.xml" > "$out/pytest.log" 2>&1 < /dev/null
    code=$?
    tail -n 3 "$out/pytest.log";;
  pairs)
    note "start pairs"
    python tools/modal/4_t04_pairs.py --output "$out/pairs" > "$out/pairs.log" 2>&1 < /dev/null
    code=$?
    tail -n 5 "$out/pairs.log";;
  *) echo "unknown job $1"; exit 2;;
esac
note "end exit=$code"
exit $code
