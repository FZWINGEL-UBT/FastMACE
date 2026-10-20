#!/usr/bin/env bash
# OPT-T01 step 2: FP32 GEMM precision modes on the production shapes of 2-T20a, local RTX 5060 Ti (WSL2).
#   setsid -f python3 tools/ci/local_gpu.py run --purpose "OPT-T01 GEMM precision" -- \
#     bash results/opt-T01-gemm-precision-20261020/run.sh ROOT BREAKDOWN >/dev/null 2>&1
# BREAKDOWN: a 2-T20a run directory holding NaCl/gemms.csv and NbCl5/gemms.csv. One process after an idle
# check (no WSL compute process, utilisation <= 5 % in both nvidia-smi views).
set -u -o pipefail
ROOT=$1
BREAKDOWN=$2
FM=/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t20-5ac2bd
CACHE=$HOME/fastmace-cache/opt-T01
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$(dirname "$ROOT")" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=$CACHE/triton XDG_CACHE_HOME=$CACHE/xdg
for i in $(seq 12); do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
  [ -z "$pids" ] && [ "$util" -le 5 ] && [ "$wutil" -le 5 ] && break
  [ "$i" -eq 12 ] && { echo "$(date -u +%FT%TZ) NOT IDLE util=$util win=$wutil" > "$ROOT.idle"; exit 1; }
  sleep 5
done
echo "$(date -u +%FT%TZ) idle util=$util win=$wutil" > "$ROOT.idle"
"$HOME/environments/mace_cueq_oeq/bin/python" "$FM/tools/gemm_precision_bench.py" --output "$ROOT" \
  --gemms "$BREAKDOWN/NaCl/gemms.csv" "$BREAKDOWN/NbCl5/gemms.csv" > "$ROOT.log" 2>&1
echo "$(date -u +%FT%TZ) exit=$?" >> "$ROOT.idle"
