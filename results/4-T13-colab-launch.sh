#!/usr/bin/env bash
# 4-T13: the phase 4 acceptance's untimed Colab legs (owner, 23 October 2026: untimed parts on the fastest GPU for
# their type; Colab uncapped; VMs always stopped). One call per VM: bash launch.sh g4|a100-equivalence|a100-engine|l4
set -uo pipefail
leg=${1:?leg}
commit=269674154d504e732ea3f19f25d9d1a4901c0aa9
driver=${DRIVER:-$HOME/fastmace-4-T13/colab-driver-e1bbb96/tools/colab/fastmace_colab.py}
baseline=$HOME/fastmace-4-T13/colab-baseline/1a-T18
out=$HOME/fastmace-acceptance/4-T13-colab-$leg-$(date -u +%Y%m%d)
PT='export PYTHONPATH=$PYTHONPATH:$PWD/tools/acceptance FASTMACE_RESULTS=$OUT/results; python -m pytest -p no:cacheprovider -p resultlog --result-log=$OUT/results.jsonl --junitxml=$OUT/junit.xml -rfEs --durations=40'
# Run locally (preflight: stale-expectation reruns, research-engine comparisons; every label's loading)
LOCAL="--ignore=tests/gpu/test_reproducible_schemes_gpu.py --ignore=tests/gpu/test_subprocess_worker_gpu.py
  --ignore=tests/gpu/test_canonical_restart_gpu.py --ignore=tests/gpu/test_reproducible_gpu.py
  --ignore=tests/gpu/test_planner_gpu.py --ignore=tests/gpu/test_simulation_gpu.py --ignore=tests/gpu/test_selfcheck_gpu.py
  --ignore=tests/gpu/test_gate_resources_gpu.py --ignore=tests/gpu/test_isolation_gpu.py
  --ignore=tests/gpu/test_fused_integrator_gpu.py --ignore=tests/gpu/test_engine_gpu.py --ignore=tests/gpu/test_registry_gpu.py
  --ignore=tests/gpu/test_fixed_costs_gpu.py --ignore=tests/gpu/test_polar_gpu.py --ignore=tests/gpu/test_reference_tier_gpu.py
  --ignore=tests/gpu/test_reference_gpu.py --ignore=tests/gpu/test_loading_gpu.py"
# Float64-heavy files: the A100 legs
A100="--ignore=tests/gpu/test_nve_csvr_berendsen_gpu.py --ignore=tests/gpu/test_nose_hoover_mtk_mh1_gpu.py
  --ignore=tests/gpu/test_ase_equivalence_gpu.py --ignore=tests/gpu/test_fix_atoms_gpu.py
  --ignore=tests/gpu/test_schemes_engine_ase_gpu.py"
LOCAL=$(echo $LOCAL); A100=$(echo $A100)
MH1='ckpt=$FASTMACE_MODELS/mace-mh-1.model; S=$FASTMACE_STRUCTURES; A=$S/anneal-minima'
OPTIMISERS="$MH1"'
pair() { name=$1; opt=$2; shift 2; mkdir -p $OUT/$name
  python tools/bench/optimizer_frames.py --side ase --optimizer $opt --checkpoint $ckpt --modes graph --skins auto "$@" --output $OUT/$name/ase > $OUT/$name/ase.log 2>&1; echo "$name ase $?"
  python tools/bench/optimizer_frames.py --side gpu --optimizer $opt --checkpoint $ckpt --modes graph --skins auto "$@" --compare $OUT/$name/ase.json --output $OUT/$name/gpu > $OUT/$name/gpu.log 2>&1; echo "$name gpu $?"; }
for opt in lbfgs fire; do pair fixed-f32-$opt $opt --traj NaCl=$A/NaCl/captured.traj NbCl5=$A/NbCl5/captured.traj --frames 0 10 20; done
pair cell-lbfgs lbfgs --cell-filter frechet --structures NaCl NaNbCl6 NaTaCl6 NaNbO3 NaTaO3 NbCl3O TaCl3O NbCl2O NbCl5 TaCl5 --structures-root $S
pair cell-d3 lbfgs --cell-filter frechet --structures NbCl5 --structures-root $S --dispersion d3bj
pair cell-fire fire --cell-filter frechet --structures NaCl NbCl5 --structures-root $S
mkdir -p $OUT/relax-cli; python -m fastmace.cli relax $S/Structures/Benchmark/R2SCAN/NaCl.cif --model mh1-r2scan --repeat 3 3 3 --optimizer gpu-LBFGS --fmax 0.01 --output $OUT/relax-cli/NaCl-relaxed.extxyz > $OUT/relax-cli/relax.log 2>&1; echo "relax-cli $?"'
MINIMA="$MH1"'
for c in NaCl NaNbCl6 NaTaCl6 NaNbO3 NaTaO3 NbCl3O TaCl3O NbCl2O NbCl5 TaCl5; do mkdir -p $OUT/minima/$c; cp $S/minima-ase/$c/ase.json $S/minima-ase/$c/ase.npz $OUT/minima/$c/
  python tools/bench/optimizer_frames.py --side gpu --checkpoint $ckpt --traj $c=$A/$c/captured.traj --frames all --modes graph --skins auto --compare $OUT/minima/$c/ase.json --output $OUT/minima/$c/gpu > $OUT/minima/$c/gpu.log 2>&1; echo "$c gpu $?"; done
python tools/validation/anneal_minima.py evaluate --data $A --runs $OUT/minima > $OUT/evaluate.log 2>&1; echo "evaluate $?"; tail -30 $OUT/evaluate.log'
SOFTCELL="$MH1"'
pair() { name=$1; opt=$2; shift 2; mkdir -p $OUT/$name
  python tools/bench/optimizer_frames.py --side ase --optimizer $opt --checkpoint $ckpt --modes graph --skins auto "$@" --output $OUT/$name/ase > $OUT/$name/ase.log 2>&1; echo "$name ase $?"
  python tools/bench/optimizer_frames.py --side gpu --optimizer $opt --checkpoint $ckpt --modes graph --skins auto "$@" --compare $OUT/$name/ase.json --output $OUT/$name/gpu > $OUT/$name/gpu.log 2>&1; echo "$name gpu $?"; }
pair cell-lbfgs-f002 lbfgs --cell-filter frechet --structures NaNbO3 --structures-root $S --fmax 0.002
pair cell-d3-f002 lbfgs --cell-filter frechet --structures NbCl5 --structures-root $S --dispersion d3bj --fmax 0.002'
D3PROBE="$MH1"'
pair() { name=$1; opt=$2; shift 2; mkdir -p $OUT/$name
  python tools/bench/optimizer_frames.py --side ase --optimizer $opt --checkpoint $ckpt --modes graph --skins auto "$@" --output $OUT/$name/ase > $OUT/$name/ase.log 2>&1; echo "$name ase $?"
  python tools/bench/optimizer_frames.py --side gpu --optimizer $opt --checkpoint $ckpt --modes graph --skins auto "$@" --compare $OUT/$name/ase.json --output $OUT/$name/gpu > $OUT/$name/gpu.log 2>&1; echo "$name gpu $?"; }
for f in 0.01 0.002; do pair cell-d3-f64-$f lbfgs --cell-filter frechet --structures NbCl5 --structures-root $S --dispersion d3bj --precision float64 --fmax $f; grep -h "frame 0" $OUT/cell-d3-f64-$f/*.log; done'
PROTOCOL='bash results/3-T14-protocol/run.sh start $OUT/short npt_anneal_NbCl5_short 13500 && bash results/3-T14-protocol/run.sh resume $OUT/short 13500'
NVE='for s in NaCl NbCl5; do python tools/bench/nve_drift.py --structure $s --output $OUT; done'
PROBES="$MH1"'
probe() { name=$1; opt=$2; crystal=$3; shift 3; mkdir -p $OUT/$name
  for side in ase gpu; do python tools/validation/anneal_frame_probe.py run --side $side --optimizer $opt --checkpoint $ckpt --traj $crystal=$A/$crystal/captured.traj --frame 10 --precision float64 "$@" --output $OUT/$name/$side > $OUT/$name/$side.log 2>&1; echo "$name $side $?"; done
  python tools/validation/anneal_frame_probe.py compare $OUT/$name/ase $OUT/$name/gpu > $OUT/$name/compare.json 2>&1; cat $OUT/$name/compare.json; }
for opt in lbfgs fire; do for c in NaCl NbCl5; do probe fixed-f64-$opt-$c $opt $c; done; probe fix-atoms-$opt $opt NbCl5 --fix-every 3 --ase-calculator mace; done'
common=(--commit "$commit" --baseline "$baseline" --timeout-min 480 --session fm-4t13-$leg)
case $leg in
  g4) python3 "$driver" --gpu G4 "${common[@]}" --models mh-1,medium,polar-1-s,off23-small --out "$out" \
        --command "$PT tests/gpu -m acceptance $LOCAL $A100" \
        --command "$PT tests/gpu -m 'slow and not acceptance' $LOCAL $A100" \
        --command "$OPTIMISERS" --command "$MINIMA" --command "$PROTOCOL" --command "$NVE" \
        --command 'python tools/acceptance/compat.py --provider colab --label "Colab G4" --output $OUT' ;;
  a100-equivalence) python3 "$driver" --gpu A100 "${common[@]}" --out "$out" \
        --command "$PT tests/gpu/test_nve_csvr_berendsen_gpu.py tests/gpu/test_nose_hoover_mtk_mh1_gpu.py tests/gpu/test_ase_equivalence_gpu.py tests/gpu/test_fix_atoms_gpu.py -m 'slow or acceptance'" ;;
  a100-engine) python3 "$driver" --gpu A100 "${common[@]}" --out "$out" \
        --command "$PT tests/gpu/test_schemes_engine_ase_gpu.py" --command "$PROBES" ;;
  g4-engine) python3 "$driver" --gpu G4 "${common[@]}" --out "$out" \
        --command "$PT tests/gpu/test_schemes_engine_ase_gpu.py" --command "$PROBES" ;;
  l4-combo) python3 "$driver" --gpu L4 --commit 636787bd5612366bf7c55552858f1362c94be571 --baseline "$baseline" \
        --timeout-min 120 --session fm-4t13-l4-combo --out "$out" \
        --command "$PT tests/gpu/test_combo_kernels_cuda.py" ;;
  l4) python3 "$driver" --gpu L4 "${common[@]}" --out "$out" \
        --command 'python tools/acceptance/compat.py --provider colab --label "Colab L4" --output $OUT' ;;
  g4-fix) python3 "$driver" --gpu G4 --commit ba1d5be27b93dd9a7f5c762c0317ff241807c582 --baseline "$baseline" --timeout-min 480 \
        --session fm-4t13-g4-fix --models mh-1 --out "$out" \
        --command "$PT tests/gpu/test_mc_evaluator_gpu.py tests/gpu/test_template_cache_gpu.py" \
        --command "$SOFTCELL" --command "$MINIMA" ;;
  a100-d3probe) python3 "$driver" --gpu A100 --commit d6d2f9d8b26e6a8743e6e30e7b1044632850cf26 --baseline "$baseline" --timeout-min 240 \
        --session fm-4t13-a100-d3probe --models mh-1 --out "$out" --command "$D3PROBE" ;;
  *) echo "unknown leg $leg" >&2; exit 2 ;;
esac
