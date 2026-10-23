#!/usr/bin/env bash
# 4-T09 GPU smoke on the local RTX 5060 Ti, under tools/ci/local_gpu.py run:
#
#   bash results/4-T09-smoke/run.sh e2e    # npt_anneal_NbCl5_short end to end, relax and quench with gpu-LBFGS
#   bash results/4-T09-smoke/run.sh copy   # fastmace quench --into gpu on a copy of 3b-T07's short run,
#                                          # beside that run's ASE LBFGS quench of the same 21 frames
#   bash results/4-T09-smoke/run.sh all    # both, in that order
#
# Run directories under ~/fastmace-runs/4-T09/; every command appends to <run>.log.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
checkout="$(cd "$here/../.." && pwd)"
python="${FASTMACE_HYBRID_PYTHON:-$HOME/environments/mace_cueq_oeq/bin/python}"
export PYTHONPATH="$checkout" PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS="${FASTMACE_MODELS:-$HOME/models}"
export TORCHINDUCTOR_COMPILE_THREADS=4
runs="$HOME/fastmace-runs/4-T09"
mkdir -p "$runs"

fastmace() {
    local log="$1"; shift
    # An app worktree's .git names a Windows path that WSL's git cannot follow: the caller may pass the commit.
    local commit="${FASTMACE_COMMIT:-$(git -C "$checkout" rev-parse HEAD 2>/dev/null || echo unknown)}"
    echo "=== $(date -u +%FT%TZ) fastmace $* (FastMACE $commit)" >>"$log"
    "$python" -m fastmace.cli "$@" 2>&1 | tee -a "$log"
}

e2e() {
    # The shipped short protocol as YAML (this Python 3.10 has no tomllib), with gpu-LBFGS for both stages.
    local config="$runs/npt_anneal_NbCl5_short_gpu.yaml" data="$checkout/fastmace/protocols/data"
    python3 - "$data/npt_anneal_NbCl5_short.toml" "$config" "$data" <<'PY'
import json, os, sys, tomllib
source, out, data = sys.argv[1:]
with open(source, "rb") as handle:
    config = tomllib.load(handle)
config["structure"] = os.path.join(data, config["structure"])
config["relax"]["optimizer"] = config["quench"]["optimizer"] = "gpu-LBFGS"
with open(out, "w") as handle:
    json.dump(config, handle, indent=1)
PY
    rm -rf "$runs/e2e"
    fastmace "$runs/e2e.log" run "$config" --output "$runs/e2e"
}

copy() {
    local source="$HOME/fastmace-runs/3b-T07/short" target="$runs/short-copy"
    rm -rf "$target"
    cp -r "$source" "$target"
    fastmace "$runs/short-copy.log" quench "$target" --into gpu --optimizer gpu-LBFGS
}

case "${1:-}" in
    e2e) e2e ;;
    copy) copy ;;
    all) e2e; copy ;;
    *) echo "usage: run.sh e2e | copy | all" >&2; exit 2 ;;
esac
