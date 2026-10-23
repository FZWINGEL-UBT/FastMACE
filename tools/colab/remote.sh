#!/usr/bin/env bash
# FastMACE on a Colab VM: the VM side of tools/colab/fastmace_colab.py. Usage on the VM:
#   bash /content/fm/remote.sh <stage> [arg]
# Stages: probe | setup | restore | run <n> | pack
# Expects /content/fm/payload.tar (gzipped source, baseline export, requirements, models list) from the driver,
# and for `restore` the store's cache parts /content/fm/cache-in-<nnn>.tar.gz. Records go to /content/fm/out (downloaded by the driver).
# Mirrors tools/modal/fastmace_modal.py: Python 3.10 (uv-managed), requirements-hybrid.txt with the cu130
# torch index, g++ for the OEq host extension (no nvcc), TORCHINDUCTOR_COMPILE_THREADS=4, caches under one root.
set -uo pipefail
FM=/content/fm
SRC=$FM/src
VENV=$FM/venv
PY=$VENV/bin/python
OUT=$FM/out
CACHE=$FM/cache
mkdir -p "$OUT" "$CACHE" "$FM/tmp" "$FM/models"
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH=$SRC FASTMACE_MODELS=$FM/models FASTMACE_STRUCTURES=$FM/baseline/1a-T18 \
  FASTMACE_BASELINE=$FM/baseline/1a-T18 FASTMACE_HYBRID_PYTHON=$PY FASTMACE_CACHE=$CACHE/fastmace \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/nv \
  CUDA_CACHE_MAXSIZE=4294967296 MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$FM/tmp \
  UV_CACHE_DIR=$FM/uv-cache UV_PYTHON_INSTALL_DIR=$FM/uv-python
# Colab keeps the driver in /usr/lib64-nvidia, not among FastMACE's DRIVER_CANDIDATES (kernels/_oeq_runtime.py)
export OEQ_CUDA_DRIVER_LIBRARY=/usr/lib64-nvidia/libcuda.so.1
export PATH=$VENV/bin:$PATH
stage=${1:?stage}
now() { date +%s.%N; }
since() { python3 -c "import sys; print(round($(now) - $1, 1))"; }
cd "$FM"
case $stage in
probe)
  # The GPU as the driver keys the compile caches by it; no venv needed yet.
  python3 - <<'EOF' > "$OUT/probe.json"
import json, os, subprocess
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
name, driver, memory, cap = [s.strip() for s in sh(
    "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader").split(",")]
cuda = sh("nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9.]+' | grep -oE '[0-9.]+$'")
mem_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
print(json.dumps(dict(gpu_name=name, driver=driver, gpu_memory=memory, capability=cap, cuda_driver=cuda,
                      cpus=os.cpu_count(), ram_gib=mem_gib)))
EOF
  cat "$OUT/probe.json"
  ;;
setup)
  t0=$(now)
  rm -rf "$SRC" "$FM/baseline" "$FM/staging" && mkdir -p "$SRC" "$FM/baseline" "$FM/staging"
  tar -xf payload.tar -C "$FM/staging"
  tar -xzf "$FM/staging/source.tar.gz" -C "$SRC"
  tar -xzf "$FM/staging/baseline.tar.gz" -C "$FM/baseline"
  echo "source $(cat "$SRC/.source_commit")"
  t1=$(now)
  if [ -x "$PY" ] && cmp -s "$FM/staging/requirements.txt" "$FM/venv.requirements.txt"; then
    echo "venv already built from these requirements (kept session)"
  else
    rm -rf "$VENV"
    uv python install -q 3.10 || exit 1
    uv venv -q -p 3.10 "$VENV" || exit 1
    uv pip install -q --python "$PY" -r "$FM/staging/requirements.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match || exit 1
    cp "$FM/staging/requirements.txt" "$FM/venv.requirements.txt"
  fi
  t2=$(now)
  # Released checkpoints from their official URLs, accepted only with the label table's SHA-256.
  "$PY" - "$FM/staging/models.txt" "$FM/models" <<'EOF' || exit 1
import hashlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastmace.model import labels
names = [line.strip() for line in open(sys.argv[1]) if line.strip()]
target = Path(sys.argv[2])
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
def fetch(name):
    c = labels.CHECKPOINTS[name]
    path = target / c.url.rsplit("/", 1)[-1]
    if not (path.exists() and digest(path) == c.sha256):
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(c.url, part)
        if digest(part) != c.sha256:
            raise SystemExit(f"{name}: SHA-256 mismatch for {c.url}")
        part.rename(path)
    return f"{name}: {path.name}"
with ThreadPoolExecutor(8) as pool:
    for line in pool.map(fetch, names):
        print(line)
EOF
  t3=$(now)
  uv pip freeze --python "$PY" > "$OUT/pip-freeze.txt"
  "$PY" - "$OUT/setup.json" "$(python3 -c "print(round($t1-$t0,1))")" "$(python3 -c "print(round($t2-$t1,1))")" \
    "$(python3 -c "print(round($t3-$t2,1))")" <<'EOF' || exit 1
import hashlib, importlib.metadata as md, json, sys
import torch
freeze = open("/content/fm/out/pip-freeze.txt").read()
def version(name):
    for candidate in (name, name.replace("_", "-")):
        try:
            return md.version(candidate)
        except md.PackageNotFoundError:
            pass
packages = ("torch", "triton", "cuequivariance", "cuequivariance_torch", "openequivariance", "mace-torch", "e3nn",
            "ase", "numpy", "pytest", "pytest-xdist")
record = dict(python=sys.version.split()[0], versions={p: version(p) for p in packages}, torch_cuda=torch.version.cuda,
              cuda_available=torch.cuda.is_available(), pip_freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest(),
              unpack_s=float(sys.argv[2]), install_s=float(sys.argv[3]), models_s=float(sys.argv[4]))
json.dump(record, open(sys.argv[1], "w"), indent=2)
print(json.dumps(record))
EOF
  echo "setup $(since "$t0") s"
  ;;
restore)
  # Compile caches from an earlier run on the same capability and stack; the listing tells `pack` what changed.
  t0=$(now)
  # The store's parts in order (the first run's caches, then each later run's new files).
  for part in $(ls cache-in-*.tar.gz 2>/dev/null | sort); do tar -xzf "$part" -C "$CACHE" && rm -f "$part"; done
  find "$CACHE" -type f -printf '%s %T@ %P\0' > "$FM/cache-before.lst"
  echo "restored $(du -sh "$CACHE" | cut -f1), $(since "$t0") s"
  ;;
run)
  n=${2:?command number}
  dir=$OUT/cmd$n
  mkdir -p "$dir"
  cd "$SRC"
  # $OUT is the command's own results directory, as on Modal.
  OUT=$dir bash -c "$(cat "$FM/staging/command$n.sh")"
  ;;
pack)
  t0=$(now)
  # Only the files that are new or changed since `restore` go back, as the store's next part.
  rm -f cache-out.tar.gz
  find "$CACHE" -type f -printf '%s %T@ %P\0' > "$FM/cache-after.lst"
  python3 - "$FM/cache-before.lst" "$FM/cache-after.lst" "$FM/cache-delta.lst" <<'EOF'
import sys
def entries(path):
    return set(open(path, "rb").read().split(b"\0")) - {b""}
new = entries(sys.argv[2]) - entries(sys.argv[1])
with open(sys.argv[3], "wb") as f:
    f.write(b"".join(entry.split(b" ", 2)[2] + b"\0" for entry in sorted(new)))
print(f"{len(new)} new or changed cache files")
EOF
  if [ -s "$FM/cache-delta.lst" ]; then
    tar -czf cache-out.tar.gz -C "$CACHE" --null -T "$FM/cache-delta.lst" \
      && echo "caches changed: $(du -sh cache-out.tar.gz | cut -f1)"
  else
    echo "caches unchanged"
  fi
  tar -czf out.tar.gz -C "$FM" out
  echo "pack $(since "$t0") s"
  ;;
*) echo "unknown stage $stage"; exit 2 ;;
esac
