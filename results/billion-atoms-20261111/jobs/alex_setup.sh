#!/bin/bash -l
# Billion-atom plan, "Before step 4" (11 November 2026): the setup speed-ups on one A100 (not a whole node).
# 1. The chunked supercell builder (tools/bench/tiled_capacity.py ``supercell``), CPU only: bitwise ASE's
#    (``supercell_ase``) at 300^3 (135M SrTiO3 atoms, rattle 0.05), then 430^3, 520^3 and 585^3 alone (10^9 atoms at
#    585^3), each with its time and the cgroup's peak.
# 2. TILE_FIRST_CALL_PROBE A/B: mode auto's measuring first call whole (off) against its 8 largest tiles (on), SrTiO3
#    OMAT-0 medium, the go/no-go's call (tile lists, aliased host state, wavefront on, species index, host positions),
#    58^3 and 160^3, one process each: the probe's mode, receiver cap and workspace against the whole call's.
# Usage: sbatch alex_setup.sh SHA
#SBATCH --job-name=fm-setup
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/setup-$SLURM_JOB_ID
mkdir -p $OUT
log=$OUT/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
SRC=$TMPDIR/src
mkdir -p $SRC && tar -xf $B/src-$SHA.tar -C $SRC
env_dir=/home/atuin/v130cb/v130cb18/fm-a100omat/env
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_MODELS=/home/atuin/v130cb/v130cb18/fm-a100omat/models FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_CACHE=$TMPDIR/cache TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export XDG_CACHE_HOME=$TMPDIR/xdg MALLOC_MMAP_THRESHOLD_=1048576
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $SRC || exit 4
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed 's#/step_batch.*##')
note "job $SLURM_JOB_ID on $(hostname); cgroup limit $(cat $cg/memory.max 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; nproc $n"
( while true; do echo "$(date +%T) cg $(( $(cat $cg/memory.current 2>/dev/null || echo 0) / 1048576 )) MiB peak $(( $(cat $cg/memory.peak 2>/dev/null || echo 0) / 1048576 )) MiB gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"; sleep 30; done >> $OUT/mem.log ) &
sampler=$!
cat > $TMPDIR/supercell.py <<'PY'
import json, resource, sys, time
sys.path.insert(0, "tools/bench")
import numpy as np
import tiled_capacity as tc

mode, n = sys.argv[1], int(sys.argv[2])
rec = dict(mode=mode, repeat=n)
t = time.perf_counter()
got = tc.supercell("SrTiO3", (n, n, n), 0.05, 0.0)
rec.update(atoms=len(got), builder_s=time.perf_counter() - t,
           maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
           positions_dtype=str(got.positions.dtype), abs_max=float(np.abs(got.positions).max()))
if mode == "ase":
    t = time.perf_counter()
    want = tc.supercell_ase("SrTiO3", (n, n, n), 0.05, 0.0)
    rec.update(ase_s=time.perf_counter() - t,
               bitwise=bool(np.array_equal(got.positions, want.positions)
                            and np.array_equal(got.numbers, want.numbers)
                            and np.array_equal(got.cell.array, want.cell.array)))
print(json.dumps(rec))
PY
note "supercell ase 300"
python $TMPDIR/supercell.py ase 300 >> $OUT/supercell.jsonl 2> $OUT/supercell-300.err; note "exit=$? $(tail -1 $OUT/supercell.jsonl)"
for r in 430 520 585; do
  note "supercell $r"
  python $TMPDIR/supercell.py new $r >> $OUT/supercell.jsonl 2> $OUT/supercell-$r.err; note "exit=$? $(tail -1 $OUT/supercell.jsonl)"
done
for rep in 58 160; do
  for probe in off on; do
    note "probe $probe $rep"
    python tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat ${rep}x${rep}x${rep} \
      --mode auto --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto \
      --host-positions --first-call-probe $probe --output $OUT/probe-$probe-$rep > $OUT/probe-$probe-$rep.log 2>&1 < /dev/null
    note "exit=$? $(tail -1 $OUT/probe-$probe-$rep.log | cut -c1-600)"
  done
done
kill $sampler 2>/dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
