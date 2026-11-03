#!/bin/bash
# murphy: 1M A/B of the reassociation layout (e2903496) against main (a7159afd), order base new new base; GPU tests.
W=/mnt/homes/zwingel/fm-tiled
A="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 58x58x58 --mode auto --calls 3 --tile-list --species-index --receivers 16384 --no-budget --sweep-times"
for p in base-a new-a new-b base-b; do
  case $p in base-*) S=$W/src-all-a7159afd;; new-*) S=$W/src-all-e2903496;; esac
  SRC=$S bash $W/run.sh murphy-reassoc-$p $A >/dev/null || exit $?
done
for d in $W/runs/murphy-reassoc-*-?; do python3 - $d <<PY
import json,glob,sys
f=glob.glob(sys.argv[1]+"/tiled-*.json")
if not f: print(sys.argv[1], "no json"); sys.exit()
r=json.load(open(f[0]))
print(sys.argv[1].split("/")[-1], r.get("valid"), (r.get("error") or "")[:200], r.get("tiled",{}).get("energy_eV"))
for c in r.get("calls",[]):
    print("  %.2f s %s R%s peak %.2f" % (c["seconds"], c["mode"], c["receivers"], c["total_peak"]/2**30))
print("  ", {k: round(v, 3) for k, v in (r.get("sweep_times") or {}).get("seconds", {}).items()})
print("  ", [x for x in r.get("rewrites", []) if "reassociate" in x][:1])
PY
done
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " ")
[ -n "$pids" ] && { echo "GPU busy, tests skipped"; exit 3; }
ENV=/mnt/homes/zwingel/environments/mace_cueq_oeq; site=$ENV/lib/python3.10/site-packages; S=$W/src-all-e2903496
R=$W/runs/murphy-reassoc-gputests; mkdir -p $R/tmp
cd $S && PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$W/cache TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor \
  TRITON_CACHE_DIR=$W/cache/triton XDG_CACHE_HOME=$W/cache/xdg TMPDIR=$R/tmp \
  LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib $ENV/bin/python - > $R/log.txt 2>&1 <<PY
import sys, importlib.util, torch
exec(open("$W/run_tests_nopytest.py").read().split("failed = 0")[0])
spec = importlib.util.spec_from_file_location("t", "tests/gpu/test_kernels_triton_cuda.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.backends.cuda.matmul.allow_tf32 = False
for seg in (False, True):
    m.test_inner_layout_is_the_plain_layout_transposed(seg); print("PASS inner layout segmented=%s" % seg)
m.test_triton_segmented_forward_matches_reference(); print("PASS segmented forward")
for dt, tol in ((torch.float32, 1e-4), (torch.float64, 1e-12)):
    m.test_triton_backward_matches_reference(dt, tol); print("PASS backward", dt)
PY
echo "gpu tests exit $?"; grep -E "PASS|Error" $R/log.txt
