#!/bin/bash
# Clean-install check for `pip install fastmace[oeq]` without a C++ compiler (plan task 6-T04, 6-T03).
#
#   tools/release/clean_install.sh WHEEL_DIR [PYTHON_VERSION]      (default 3.10)
#   CLEAN_INSTALL_STAGE=install|check|all (default all): install needs no GPU; check is the GPU part.
#
# 1. A fresh venv under ~/tmp/6-T04/clean-<python> (removed first), made with uv.
# 2. Installs torch 2.14.1+cu130, cuEquivariance's CUDA 13 ops, then FastMACE from this checkout with
#    the [oeq] extra, taking fastmace-oeq-ext from WHEEL_DIR (--find-links; nothing comes from or goes to
#    PyPI for it).
# 3. Every later step runs with g++, c++, ninja and nvcc shadowed by sentinels that log and fail, ahead of
#    the venv's bin (which holds the `ninja` wheel OpenEquivariance depends on). gcc (C) stays: Triton
#    compiles its small C launcher with it, which is outside this task (docs/release/oeq-wheels.md).
# 4. `fastmace doctor`, then a 64-atom NaCl NVT smoke (MH-1, 20 steps) that must plan the hybrid backend,
#    report the host extension as "prebuilt", and leave the sentinel log empty.
#
# The GPU part must run under the local reservation:
#   python3 tools/ci/local_gpu.py run --purpose "6-T04 clean install" -- bash tools/release/clean_install.sh DIR
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
wheels=$(cd "${1:?usage: clean_install.sh WHEEL_DIR [PYTHON_VERSION]}" && pwd)
python_version=${2:-3.10}
root=${CLEAN_INSTALL_ROOT:-$HOME/tmp/6-T04}
venv=$root/clean-py$python_version
shims=$root/no-compiler-bin
log=$root/compiler-attempts-py$python_version.log
models=${FASTMACE_MODELS:-$HOME/models}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv} PYTHONDONTWRITEBYTECODE=1 TMPDIR=$root/tmp
stage=${CLEAN_INSTALL_STAGE:-all}
# 6-T13: a built sdist or wheel instead of the checkout, and never the checkout through PYTHONPATH (Colab sets it).
fastmace_source=${CLEAN_INSTALL_FASTMACE:-$repo}
case $fastmace_source in /*) ;; *) fastmace_source=$(cd "$(dirname "$fastmace_source")" && pwd)/$(basename "$fastmace_source") ;; esac
unset PYTHONPATH
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$root" "$TMPDIR"
py=$venv/bin/python

if [ "$stage" != check ]; then
rm -rf "$venv" "$shims" "$log"
uv venv -q --python "$python_version" "$venv"
uv pip install -q --python "$py" --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch==2.14.1+cu130"
uv pip install -q --python "$py" --find-links "$wheels" \
  "cuequivariance-ops-torch-cu13" "fastmace[oeq] @ file://$fastmace_source" "torch==2.14.1+cu130" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip check --python "$py"
"$py" -m pip --version >/dev/null 2>&1 || true
echo "== installed"
uv pip list --python "$py" | grep -Ei "^(torch|openequivariance|fastmace|fastmace-oeq-ext|cuequivariance|mace-torch|ninja) "
fi
[ "$stage" = install ] && exit 0

mkdir -p "$shims"
for tool in g++ c++ x86_64-linux-gnu-g++ ninja nvcc; do
  printf '#!/bin/sh\necho "%s $*" >> "%s"\necho "%s blocked by clean_install.sh" >&2\nexit 97\n' \
    "$tool" "$log" "$tool" > "$shims/$tool"
  chmod +x "$shims/$tool"
done
: > "$log"
export PATH="$shims:$venv/bin:/usr/local/bin:/usr/bin:/bin"
export CXX="$shims/g++"
export FASTMACE_CACHE=$root/cache-py$python_version FASTMACE_MODELS=$models FASTMACE_OEQ_PREBUILT=1
rm -rf "$FASTMACE_CACHE" "$TMPDIR/nacl-smoke"
echo "== which: g++=$(command -v g++) ninja=$(command -v ninja) nvcc=$(command -v nvcc)"

echo "== fastmace doctor"
(cd "$root" && "$venv/bin/fastmace" doctor --no-hash) || echo "(doctor exit $?: required checks failed; see above)"

echo "== NaCl smoke"
cd "$root"
"$py" - <<'EOF'
import json, os, sys
import fastmace as fm
from fastmace import _patches
from ase.build import bulk

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
atoms.rattle(0.05, seed=17)
result = fm.md(atoms, model="mh1-r2scan", temperature_K=300, steps=20, seed=1,
               output=os.path.join(os.environ["TMPDIR"], "nacl-smoke"))
record = _patches.record_of("oeq_runtime")
import numpy as np
committed = result.simulation.committed()
summary = {"backend": result.plan.engine.backend, "oeq_runtime": None if record is None else record.detail,
           "step": int(committed.step), "finite": bool(np.isfinite(committed.positions).all()),
           "python": sys.version.split()[0], "fastmace": fm.__version__, "fastmace_file": fm.__file__}
print(json.dumps(summary, indent=1))
assert summary["backend"] == "hybrid" and summary["step"] == 20 and summary["finite"], summary
assert record is not None and "host extension: prebuilt" in record.detail, summary
assert fm.__file__.startswith(sys.prefix), summary  # the installed copy, not a checkout
EOF
if [ -s "$log" ]; then echo "FAIL: a blocked compiler was called:"; cat "$log"; exit 1; fi
echo "CLEAN INSTALL OK (python $python_version): hybrid backend, prebuilt OEq host extension, no g++/ninja/nvcc call"
