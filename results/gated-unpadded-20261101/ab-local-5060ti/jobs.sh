set -u
export PYTHONPATH=/home/fzwin/fastmace-capacity/src-unpadded2 PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18   TMPDIR=/home/fzwin/fastmace-runs/unpadded-ab-20261101/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_CACHE=/home/fzwin/fastmace-cache/unpadded-local TORCHINDUCTOR_CACHE_DIR=/home/fzwin/fastmace-cache/unpadded-local/inductor   TRITON_CACHE_DIR=/home/fzwin/fastmace-cache/unpadded-local/triton MACE_COMPILE_CACHE=/home/fzwin/fastmace-cache/unpadded-local/compile XDG_CACHE_HOME=/home/fzwin/fastmace-cache/unpadded-local/xdg   FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python
PY=$HOME/environments/mace_cueq_oeq/bin/python
CK=$HOME/models/mace-mh-1.model
ab() {
  local job=$1 structure=$2 order=$3 pattern=$4; shift 4
  "$PY" /home/fzwin/fastmace-capacity/src-unpadded2/tools/ab_rewrites.py --output /home/fzwin/fastmace-runs/unpadded-ab-20261101/$job --checkpoint "$CK" --model mh1-r2scan --structures "$structure"     --prepare --a-constant 'fastmace.kernels.triton.gated_products:UNPADDED="off"'     --b-constant 'fastmace.kernels.triton.gated_products:UNPADDED="on"' --order $order --pattern $pattern "$@"     > /home/fzwin/fastmace-runs/unpadded-ab-20261101/$job.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $job exit=$?" >> /home/fzwin/fastmace-runs/unpadded-ab-20261101/run.log
}
echo "$(date -u +%FT%TZ) fastmace=$(cat /home/fzwin/fastmace-capacity/src-unpadded2/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> /home/fzwin/fastmace-runs/unpadded-ab-20261101/run.log
ab mid-ab NbCl5@5x5x8 ab abab --warmup 100 --block 100 --blocks 12
ab mid-ba NbCl5@5x5x8 ba abba --warmup 100 --block 100 --blocks 12
echo "$(date -u +%FT%TZ) DONE" >> /home/fzwin/fastmace-runs/unpadded-ab-20261101/run.log
