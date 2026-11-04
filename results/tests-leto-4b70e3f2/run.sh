#!/usr/bin/env bash
# CPU suite + ruff for fastmace/omat0-strided-tables at 4b70e3f2 on leto (NAS only, no /tmp).
set -u
D=/mnt/homes/zwingel/fastmace-branches/omat0-strided-4b70e3f2
OUT=$D/../omat0-strided-4b70e3f2-tests
ENV=/mnt/homes/zwingel/environments/mace_cueq_latest
SP=$ENV/lib/python3.10/site-packages
mkdir -p "$OUT/tmp" "$OUT/basetemp"
export TMPDIR="$OUT/tmp" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH="$D"
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
export XDG_CACHE_HOME="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor"
cd "$D"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv,noheader
  ps -eo user,pcpu,etime,comm --sort=-pcpu | head -8; echo "source $(cat .source_commit)"; } > "$OUT/idle_check.txt"
"$ENV/bin/python" -m ruff check . > "$OUT/ruff.log" 2>&1; r=$?
"$ENV/bin/python" -m pytest -n 16 --dist=load --maxschedchunk=1 -p no:cacheprovider --basetemp="$OUT/basetemp" tests/cpu > "$OUT/pytest-cpu.log" 2>&1; p=$?
{ echo "ruff exit=$r"; echo "cpu exit=$p $(tail -n 1 "$OUT/pytest-cpu.log")"; echo done; } > "$OUT/summary.txt"
