#!/bin/bash
# 6-T13 item 3b: the clean install on a Colab VM, on the VM's own Python, from the artefacts shipped in the
# payload's baseline directory (dist/: fastmace 1.0.0 wheel and the fastmace-oeq-ext wheels). Run from the source
# root by tools/colab/fastmace_colab.py with $OUT set.
set -u
dist=$FASTMACE_BASELINE/dist
# The VM's system Python, not the driver's 3.10 venv that remote.sh puts first on PATH.
py=${1:-$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')}  # or a uv-managed one
echo "VM python3: $(/usr/bin/python3 --version) -> clean venv on $py; g++: $(command -v g++); nvcc: $(command -v nvcc)"
command -v uv >/dev/null || python3 -m pip install -q uv
whl=$(ls "$dist"/fastmace-1.0.0-*.whl)
mkdir -p "$dist/oeq" && cp "$dist"/fastmace_oeq_ext-*.whl "$dist/oeq/"
CLEAN_INSTALL_ROOT=/content/fm/tmp/clean CLEAN_INSTALL_FASTMACE=$whl \
  bash tools/release/clean_install.sh "$dist/oeq" "$py" 2>&1 | tee "$OUT/clean-install-py$py.log"
status=${PIPESTATUS[0]}
cp /content/fm/tmp/clean/compiler-attempts-py$py.log "$OUT/" 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv > "$OUT/gpu.csv" 2>&1
echo "clean install exit $status"
exit "$status"
