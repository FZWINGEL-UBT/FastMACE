#!/bin/bash -l
#SBATCH --job-name=fm-cgcheck
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:05:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledcap/runs/slurm-%x-%j.out
B=/home/atuin/v130cb/v130cb18/fm-tiledcap
cat /proc/self/cgroup
mkdir -p $TMPDIR/t && tar -xf $B/src-bafc2396.tar -C $TMPDIR/t fastmace/model/tiled.py
python3 - $TMPDIR/t/fastmace/model/tiled.py <<'PY'
import os, sys
src = open(sys.argv[1]).read()
start = src.index("def _cgroup_room"); end = src.index("\n\n\n", start)
ns = {}; exec(src[start:end], ns)
print("room", ns["_cgroup_room"]())
path = open("/proc/self/cgroup").read().split(":", 2)[2].strip()
parts = [p for p in path.split("/") if p]
for d in range(len(parts), -1, -1):
    dd = os.path.join("/sys/fs/cgroup", *parts[:d])
    print(dd, os.path.isdir(dd), [open(os.path.join(dd, f)).read().strip() if os.path.exists(os.path.join(dd, f)) else None for f in ("memory.max", "memory.current")])
PY
