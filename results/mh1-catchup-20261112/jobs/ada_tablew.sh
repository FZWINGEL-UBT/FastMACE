#!/usr/bin/env bash
# MH-1 TABLE_W on the RTX 4000 Ada (fastmace/mh1-catchup, 12 November 2026): the workstation screen before the A100
# job (owner via conductor: "it should just work on workstations"). Same-process MD-step A/B, a = today's Ada MH-1
# defaults (w' on, slope table), b = TABLE_W=True (+ the Ada's HORNER; SKIP_OUTSIDE is H200/A100 only); NaCl and
# NbCl5 3x3x3, both build orders, 6 x 256 steps after 200 warm-up; then the 3x float32 gate with TABLE_W=True.
#   ROOT=/mnt/homes/zwingel/fm-mh1/ada-tablew setsid nohup bash ada_tablew.sh SRC_DIR > log 2>&1 &
set -u -o pipefail
FM=$1
ROOT=${ROOT:?set ROOT}
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
C=/mnt/homes/zwingel/fm-mh1/cache-$(hostname)
MODELS=/mnt/homes/zwingel/ResearchModule/Models
STRUCT=/mnt/homes/zwingel/ResearchModule
SITE=$E/lib/python3.10/site-packages
mkdir -p "$ROOT/tmp" "$C"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$C FASTMACE_MODELS=$MODELS \
  FASTMACE_STRUCTURES=$STRUCT TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 FASTMACE_HYBRID_PYTHON=$E/bin/python \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$FM
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" | tee -a "$log"; }
gpu_busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -; }
W=fastmace.kernels.oeq_streaming.wprime_backward
cd "$FM" || exit 4
note "start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for order in ab ba; do
  b=$(gpu_busy); [ -n "$b" ] && { note "STOP before $order: GPU busy [$b]"; exit 3; }
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output "$ROOT/ab-$order" --checkpoint "$MODELS/mace-mh-1.model" --model mh1-r2scan \
    --structures NaCl NbCl5 --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --pack "tablew|$W:TABLE_W=True" > "$ROOT/ab-$order.log" 2>&1 < /dev/null
  note "end $order exit=$? $(grep -h 'b/a' "$ROOT/ab-$order.log" | tr '\n' ' ' | cut -c1-300)"
done
b=$(gpu_busy); [ -n "$b" ] && { note "STOP before gate: GPU busy [$b]"; exit 3; }
python tools/force_stress_gate.py --output "$ROOT/gate-tablew" --engine fastmace --model mh1-r2scan \
  --checkpoint "$MODELS/mace-mh-1.model" --structures-root "$STRUCT" --structures NaCl NaNbO3 NbCl5 \
  --production-constant "$W:TABLE_W=True" > "$ROOT/gate-tablew.log" 2>&1 < /dev/null
note "gate exit=$? $(grep -h -iE 'verdict' "$ROOT/gate-tablew.log" | tail -n 1 | cut -c1-200)"
note "done"
