#!/usr/bin/env bash
# 4-T07 on Modal (one container, both sides on the same GPU): variable-cell relaxations,
# tools/bench/optimizer_frames.py --cell-filter frechet, ASE on fm.calc then the GPU against it, per job in the
# order below, until the container's timeout. Run through tools/modal/fastmace_modal.py's job entry point:
#   modal run tools/modal/fastmace_modal.py::job --repo <checkout> --gpu L4 --timeout-min 44 \
#       --limit-usd <metered + share> --command "bash tools/validation/4_t07_modal.sh" --out <dir>
# Results: $OUT/<job>.json, .npz and .log (each relaxation is written as it finishes).
set -uo pipefail
out=${OUT:?the Modal job sets OUT}
export PYTHONDONTWRITEBYTECODE=1
MH1="--checkpoint /data/models/mace-mh-1.model"
POLAR="--checkpoint /data/models/MACE-POLAR-1-S.model --model polar1-s --head none"
COMMON="--cell-filter frechet --modes graph --skins 0.6 --structures-root /data/baseline/1a-T18"
A="NaCl NaNbCl6 NaTaCl6 NaNbO3 NaTaO3"
B="NbCl3O TaCl3O NbCl2O NbCl5 TaCl5"
run_pair() {  # name optimizer models structures [extra...]
  local name=$1 optimizer=$2 models=$3 structures=$4
  shift 4
  for side in ase gpu; do
    local compare=""
    [[ $side == gpu ]] && compare="--compare $out/ase-$name.json"
    echo "== $side-$name $(date -Is)"
    python tools/bench/optimizer_frames.py --side "$side" --optimizer "$optimizer" $models $COMMON \
      --structures $structures "$@" $compare --output "$out/$side-$name" >"$out/$side-$name.log" 2>&1
    echo "   exit $? $(date -Is)"
    tail -n 14 "$out/$side-$name.log"
  done
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
run_pair lbfgs-a lbfgs "$MH1" "$A"
run_pair lbfgs-b lbfgs "$MH1" "$B"
run_pair lbfgs-d3 lbfgs "$MH1" NbCl5 --dispersion d3bj
run_pair fire fire "$MH1" "NaCl NbCl5"
run_pair polar lbfgs "$POLAR" NbCl5
