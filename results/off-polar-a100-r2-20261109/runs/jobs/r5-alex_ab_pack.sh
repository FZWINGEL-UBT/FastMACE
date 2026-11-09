#!/bin/bash -l
# alex_ab_pack v2: A/B packs on one NHR@FAU Alex A100 (docs/nhr-cluster.md §6.3). The Modal ab_pack starts one
# process per switch and order. Here one process runs each model and build order: tools/ab_rewrites.py --pack
# builds side a (the GPU's default plan) once per structure, and each switch once, instead of a and b for every
# switch. On Alex a build of POLAR-1-M is about 4 min of CPU-bound Dynamo tracing, Inductor and graph capture.
#SBATCH --job-name=fm-abpack
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#
# Usage, on a frontend:  sbatch --output=<OUT>/slurm-%j.out [--time=HH:MM:SS] tools/jobs/alex_ab_pack.sh <job.env>
# <job.env> is a bash file the job sources. It sets:
#   FM_ROOT     directory with env/ (the venv), uv-python/ (its interpreter), models/ and baseline/ (structures)
#   SRC         the unpacked slim source (tools/jobs/slim_tar.py; holds .source_commit)
#   OUT         results directory
#   PACK        "name|ITEM ...;name|ITEM ..."  one entry per switch, at most 8. ITEM: MODULE:NAME=VALUE (a module
#               constant) or FIELD=VALUE (a RewriteConfig field), as tools/ab_rewrites.py --pack takes them
#   MODELS      "label=checkpoint ..." (checkpoint relative to $FM_ROOT/models; default polar1-m=MACE-POLAR-1-M.model)
#   STRUCTURES  default "NaCl NbCl5"; WARMUP 200, BLOCK 256, BLOCKS 6 (per side); ORDERS default "ab ba"
#   EXTRA       more tools/ab_rewrites.py arguments (e.g. "--timestep-fs 1 --precision float64")
#   CACHE       FASTMACE_CACHE (default $FM_ROOT/cache). A warm cache saves Inductor's compile, not Dynamo's tracing.
#               With STAGE=1 the job reads it through a node-local directory whose oeq/ is its own (concurrent jobs)
#   STAGE       1 (default): copy the interpreter, venv and source to node-local $TMPDIR first; 0: run them from $WORK
#   ENV_TAR     with STAGE=1: a tar holding uv-python/<interpreter>/ and env/ (as made by `tar -cf X -C <root>
#               uv-python env`), unpacked instead of copying the venv file by file (docs/nhr-cluster.md §6.3)
# Results: $OUT/run.log (one b/a line per switch, model, structure and order; SETUP lines), $OUT/ab-<model>-<order>.log
# and $OUT/ab-<model>-<order>-<name>.json (tools/ab_rewrites.py's format, setup_s.a the shared build).
# ba holds every switch's engine and side a on the GPU at once: keep packs of large models small.
unset SLURM_EXPORT_ENV
set -uo pipefail
source "${1:?usage: sbatch tools/jobs/alex_ab_pack.sh <job.env>}"
out=${OUT:?set OUT}; mkdir -p "$out"
log=$out/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
module purge >/dev/null 2>&1
root=${FM_ROOT:?set FM_ROOT}
src=${SRC:?set SRC}
IFS=';' read -r -a entries <<< "${PACK:?set PACK}"
[ "${#entries[@]}" -le 8 ] || { note "PACK has ${#entries[@]} entries, at most 8"; exit 2; }
py_dir=$(readlink -f "$root/env/bin/python"); py_dir=${py_dir%/bin/*}  # .../uv-python/cpython-X-linux-x86_64-gnu
env_dir=$root/env
if [ "${STAGE:-1}" = 1 ]; then
  local_root=$TMPDIR/fm; mkdir -p "$local_root/uv-python"; t0=$(date +%s)
  if [ -n "${ENV_TAR:-}" ]; then  # one sequential read instead of ~54k small files
    tar -xf "$ENV_TAR" -C "$local_root" && cp -a "$src" "$local_root/src" || { note "STAGE tar failed"; exit 4; }
  else
    cp -a "$py_dir" "$local_root/uv-python/" && cp -a "$env_dir" "$local_root/" && cp -a "$src" "$local_root/src" ||
      { note "STAGE copy failed"; exit 4; }
  fi
  ln -sfn "$local_root/uv-python/$(basename "$py_dir")/bin/$(basename "$(readlink -f "$root/env/bin/python")")" \
    "$local_root/env/bin/python"
  sed -i "s#^home = .*#home = $local_root/uv-python/$(basename "$py_dir")/bin#" "$local_root/env/pyvenv.cfg"
  env_dir=$local_root/env src=$local_root/src
  note "STAGE copied to $local_root in $(( $(date +%s) - t0 )) s"
fi
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=${CACHE:-$root/cache} FASTMACE_MODELS=$root/models FASTMACE_STRUCTURES=$root/baseline
# Thread pools sized to the job's cores (16 per A100; Alex refuses more per GPU)
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
mkdir -p "$FASTMACE_CACHE"
if [ "${STAGE:-1}" = 1 ]; then
  # The staged env is node-local, and every process points <cache>/oeq/cuda-runtime (OEq's stand-in CUDA home: links
  # to the env's nvidia/cu13 headers and libraries) at its own env. Concurrent staged jobs sharing one cache would leave
  # links into another node's $TMPDIR, and OEq's host-extension build fails ("cuda_runtime_api.h: No such file";
  # 4963282, 8 November 2026). So the job gets a node-local cache: links to the shared cache's entries (Inductor,
  # Triton, radial tables stay warm and shared) except oeq/, which is the job's own (its extension build, about 1 min).
  shared=$FASTMACE_CACHE local_cache=$TMPDIR/fm/cache
  mkdir -p "$local_cache/oeq" "$shared/inductor" "$shared/triton" "$shared/radial-tables"
  for entry in "$shared"/* "$shared"/.[!.]*; do
    [ -e "$entry" ] && [ "$(basename "$entry")" != oeq ] && ln -sfn "$entry" "$local_cache/$(basename "$entry")"
  done
  export FASTMACE_CACHE=$local_cache
  note "cache $shared shared through $local_cache, oeq/ per job"
fi
cd "$src" || exit 4
note "alex_ab_pack v2 job ${SLURM_JOB_ID:-?} on $(hostname) ($(nproc) cores, $(lscpu | sed -n 's/^Model name: *//p'));" \
  "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
failed=0
for model in ${MODELS:-polar1-m=MACE-POLAR-1-M.model}; do
  label=${model%%=*}; ck=$FASTMACE_MODELS/${model#*=}
  for order in ${ORDERS:-ab ba}; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$label-$order (${#entries[@]} switches)"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$out/ab-$label-$order" --checkpoint "$ck" --model "$label" \
      --structures ${STRUCTURES:-NaCl NbCl5} --warmup "${WARMUP:-200}" --block "${BLOCK:-256}" --blocks "${BLOCKS:-6}" \
      --order "$order" --pattern "$pattern" --pack "$PACK" ${EXTRA:-} > "$out/ab-$label-$order.log" 2>&1 < /dev/null
    code=$?
    [ "$code" = 0 ] || failed=$((failed + 1))
    note "end ab-$label-$order exit=$code wall $(( $(date +%s) - t0 )) s"
    grep -h -E 'b/a' "$out/ab-$label-$order.log" | sed "s/^/  $label $order /" | tee -a "$log"
    python - "$out" "$label" "$order" <<'P' | tee -a "$log"
import json, sys
from pathlib import Path
out, label, order = sys.argv[1:]
for path in sorted(Path(out).glob(f"ab-{label}-{order}-*.json")):
    for name, entry in json.loads(path.read_text())["structures"].items():
        print(f"  SETUP {path.stem} {name} " + " ".join(f"{k} {v:.0f} s" for k, v in entry["setup_s"].items()))
P
  done
done
note "done failed=$failed"
[ "$failed" = 0 ]
