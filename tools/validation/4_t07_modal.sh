#!/usr/bin/env bash
# 4-T07 on Modal (one container, both sides on the same GPU): variable-cell relaxations,
# tools/bench/optimizer_frames.py --cell-filter frechet, ASE on fm.calc then the GPU against it, per job in the
# order below, until the container's timeout. Run through tools/modal/fastmace_modal.py's job entry point:
#   modal run tools/modal/fastmace_modal.py::job --repo <checkout> --gpu L4 --timeout-min 44 \
#       --limit-usd <shared ceiling> --command "bash tools/validation/4_t07_modal.sh [JOB...]" --out <dir>
# Results: $OUT/<job>.json, .npz and .log (each relaxation is written as it finishes).
set -uo pipefail
out=${OUT:?the Modal job sets OUT}
export PYTHONDONTWRITEBYTECODE=1
MH1="--checkpoint /data/models/mace-mh-1.model"
POLAR="--checkpoint /data/models/MACE-POLAR-1-S.model --model polar1-s --head none"
COMMON="--cell-filter frechet --modes graph --skins 0.6 --structures-root /data/baseline/1a-T18"
A="NaCl NaNbCl6 NaTaCl6 NaNbO3 NaTaO3"
B="NbCl3O TaCl3O NbCl2O NbCl5 TaCl5"
SIDES=${SIDES:-ase gpu}
# ASE_FROM=<dir>: the gpu side compares against <dir>/ase-<name>.json (a committed earlier run) instead of $out.
run_pair() {  # name optimizer models structures [extra...]
  local name=$1 optimizer=$2 models=$3 structures=$4
  shift 4
  for side in $SIDES; do
    local compare=""
    [[ $side == gpu ]] && compare="--compare ${ASE_FROM:-$out}/ase-$name.json"
    echo "== $side-$name $(date -Is)"
    python tools/bench/optimizer_frames.py --side "$side" --optimizer "$optimizer" $models $COMMON \
      --structures $structures "$@" $compare --output "$out/$side-$name" >"$out/$side-$name.log" 2>&1
    echo "   exit $? $(date -Is)"
    tail -n 14 "$out/$side-$name.log"
  done
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for job in ${@:-lbfgs-a lbfgs-b lbfgs-d3 fire polar}; do
  case $job in
    lbfgs-a) run_pair lbfgs-a lbfgs "$MH1" "$A" ;;
    lbfgs-b) run_pair lbfgs-b lbfgs "$MH1" "$B" ;;
    lbfgs-d3) run_pair lbfgs-d3 lbfgs "$MH1" NbCl5 --dispersion d3bj ;;
    lbfgs-d3-gpu) SIDES=gpu ASE_FROM=results/4-T07-modal-L4-57da786/out \
                  run_pair lbfgs-d3 lbfgs "$MH1" NbCl5 --dispersion d3bj ;;
    fire) run_pair fire fire "$MH1" "NaCl NbCl5" ;;
    polar) run_pair polar lbfgs "$POLAR" NbCl5 ;;
    *) echo "unknown job $job" ;;
  esac
done
