#!/bin/bash
# murphy: 1M A/B of the trims (aa5dbbfc) against main (7e3e8cd5), order base trim trim base; then the GPU tests.
W=/mnt/homes/zwingel/fm-tiled
A="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 58x58x58 --mode auto --calls 3 --tile-list --species-index --receivers 16384 --no-budget --sweep-times"
for p in base-a trim-a trim-b base-b; do
  case $p in base-*) S=$W/src-all-7e3e8cd5;; trim-*) S=$W/src-all-aa5dbbfc;; esac
  SRC=$S bash $W/run.sh murphy-trims-$p $A >/dev/null || exit $?
done
for d in $W/runs/murphy-trims-*-?; do python3 - $d <<PY
import json,glob,sys
f=glob.glob(sys.argv[1]+"/tiled-*.json")
if not f: print(sys.argv[1], "no json"); sys.exit()
r=json.load(open(f[0]))
print(sys.argv[1].split("/")[-1], r.get("valid"), (r.get("error") or "")[:200], r.get("tiled",{}).get("energy_eV"), r.get("tiled",{}).get("max_force_eV_A"))
for c in r.get("calls",[]):
    print("  %.2f s %s R%s peak %.2f" % (c["seconds"], c["mode"], c["receivers"], c["total_peak"]/2**30))
print("  ", {k: round(v, 3) for k, v in (r.get("sweep_times") or {}).get("seconds", {}).items()})
PY
done
# GPU tests (the OEq environment has no pytest: a stub runner), after the idle check.
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " ")
[ -n "$pids" ] && { echo "GPU busy, tests skipped"; exit 3; }
ENV=/mnt/homes/zwingel/environments/mace_cueq_oeq; site=$ENV/lib/python3.10/site-packages; S=$W/src-all-aa5dbbfc
mkdir -p $W/runs/murphy-trims-gputests/tmp
cd $S && PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$W/cache TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor \
  TRITON_CACHE_DIR=$W/cache/triton XDG_CACHE_HOME=$W/cache/xdg TMPDIR=$W/runs/murphy-trims-gputests/tmp \
  LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib $ENV/bin/python $W/run_tests_nopytest.py \
  tests/gpu/test_radial_table_cuda.py::test_kernels_equal_their_pytorch_twin,test_without_density_the_folded_input_and_its_gradient_are_unchanged \
  tests/gpu/test_kernels_oeq_streaming_cuda.py::test_message_and_gradients_match_oeq,test_chunked_folded_backward_matches_the_unchunked_path,test_a_tiles_real_rows_equal_every_nodes_first_rows \
  > $W/runs/murphy-trims-gputests/log.txt 2>&1
grep -E "PASS|FAIL|SKIP" $W/runs/murphy-trims-gputests/log.txt
