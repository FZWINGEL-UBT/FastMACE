#!/usr/bin/env bash
# fastmace/spline-registry-fix (merged with main; tree src-regfix) on the Ada, one job at a time:
# 1. the CPU registry test and the GPU multi-engine test (three engines of different sizes; one captures after a
#    later equal build, that build closed and its blocks refilled), against each engine alone;
# 2. the same multi-engine layout also on the timing source 7d18028d (before the fix), CUDA_LAUNCH_BLOCKING=1, against each
#    alone (compute-sanitizer 2026.2 aborts this stack on the host: "double free or corruption", runs/cs-probe);
# 3. the full CPU suite (-n 8 -m "not acceptance", GPU hidden).
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-regfix
QUIET=3 bash $B/jobs/queue.sh true
mkdir $B/runs/regfix 2>/dev/null || { echo "claimed elsewhere"; exit 0; }
hostname > $B/runs/regfix/host
export HOST=ada ROOT=$B/runs/regfix
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
note "regfix start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# pytest lives in mace_cueq_latest (the oeq environment has none); the GPU test's child runs directly in mace_cueq_oeq
# (the hybrid backend's kernels), extracted with ast, and the comparison below is the test's own (same tolerances).
PYT=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
CUDA_VISIBLE_DEVICES= $PYT -m pytest -p no:cacheprovider -v tests/cpu/test_spline_forward_registry.py > $ROOT/tests.log 2>&1 < /dev/null
note "end cpu test exit=$? $(tail -1 $ROOT/tests.log)"
python - > $ROOT/child.py <<'PY'
import ast
tree = ast.parse(open("tests/gpu/test_spline_registry_engines_cuda.py").read())
print([n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CHILD"][0].value.value)
PY
if why=$(others); then note "STOP before the before-fix runs: $why"; exit 3; fi
for tree in src-regfix src-7d18028d; do
  tag=$([ $tree = src-regfix ] && echo after || echo before)
  if why=$(others); then note "STOP before the $tag runs: $why"; exit 3; fi
  (cd $B/$tree && export PYTHONPATH=$B/$tree CUDA_LAUNCH_BLOCKING=1
   python $ROOT/child.py > $ROOT/$tag-together.log 2>&1 < /dev/null; echo "together exit=$?" > $ROOT/$tag.exits
   for name in small large middle; do python $ROOT/child.py $name > $ROOT/$tag-$name.log 2>&1 < /dev/null; echo "$name exit=$?" >> $ROOT/$tag.exits; done)
  python - $ROOT $tag > $ROOT/$tag-compare.txt 2>&1 <<'PY'
import json, sys
import numpy as np
root, tag = sys.argv[1:3]
def last(path):
    try:
        return json.loads(open(path).read().strip().splitlines()[-1])
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error)[:200]}
together = last(f"{root}/{tag}-together.log")
print("shared", together.get("shared"), "registered", together.get("registered"), together.get("skip", ""))
ok = bool(together.get("shared")) and bool(together.get("registered"))
for name in ("small", "large", "middle"):
    alone = last(f"{root}/{tag}-{name}.log")
    if name in together and name in alone:
        p = np.abs(np.asarray(together[name]["positions"]) - np.asarray(alone[name]["positions"])).max()
        f = np.abs(np.asarray(together[name]["forces"]) - np.asarray(alone[name]["forces"])).max()
        ok &= p <= 1e-4 and f <= 1e-3 and together[name]["step"] == alone[name]["step"]
        print(name, "positions %.3g forces %.3g" % (p, f))
    else:
        ok = False
        print(name, "missing", together.get("error"), alone.get("error"))
print("PASS" if ok else "FAIL")
PY
  note "$tag-fix $(tr '
' ' ' < $ROOT/$tag.exits) | $(tr '
' ' ' < $ROOT/$tag-compare.txt | cut -c1-500) | $(grep -m2 -hE 'illegal|Error' $ROOT/$tag-together.log | tr '
' ' ' | cut -c1-200)"
done
if why=$(others); then note "STOP before suite: $why"; exit 3; fi
CUDA_VISIBLE_DEVICES= $PYT -m pytest -p no:cacheprovider -n 8 -m "not acceptance" tests/cpu -q > $ROOT/suite.log 2>&1 < /dev/null
note "end suite exit=$? $(tail -1 $ROOT/suite.log)"
note DONE
