#!/usr/bin/env bash
# 3-T14: a swap-MC protocol run in pieces on the local GPU, under tools/ci/local_gpu.py run.
#
#   bash results/3-T14-protocol/run.sh start  RUN_DIR PROTOCOL STEPS   # relaxation + the first STEPS steps
#   bash results/3-T14-protocol/run.sh resume RUN_DIR STEPS            # the next STEPS steps (and the quench)
#   bash results/3-T14-protocol/run.sh smoke  RUN_DIR                  # smoke.toml: start 3000, resume the rest
#
# PROTOCOL is a shipped protocol name or a TOML file. Every piece appends to RUN_DIR/pieces.log.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
checkout="$(cd "$here/../.." && pwd)"
python="${FASTMACE_HYBRID_PYTHON:-$HOME/environments/mace_cueq_oeq/bin/python}"
export PYTHONPATH="$checkout" PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS="${FASTMACE_MODELS:-$HOME/models}"
export TORCHINDUCTOR_COMPILE_THREADS=4

fastmace() {
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    # An app worktree's .git names a Windows path that WSL's git cannot follow: the caller may pass the commit.
    local commit="${FASTMACE_COMMIT:-$(git -C "$checkout" rev-parse HEAD 2>/dev/null || echo unknown)}"
    echo "=== $(date -u +%FT%TZ) fastmace $* (FastMACE $commit)" >>"$log"
    "$python" -m fastmace.cli "$@" 2>&1 | tee -a "$log"
}

case "$1" in
    start)
        mkdir -p "$(dirname "$2")"
        fastmace "$2.pieces.log" run "$3" --output "$2" --steps "$4"
        ;;
    resume)
        fastmace "$2.pieces.log" run --resume "$2" --steps "$3"
        ;;
    smoke)
        config="$2.smoke.toml"
        mkdir -p "$(dirname "$2")"
        { echo "structure = \"$checkout/fastmace/protocols/data/NaNbOCl4_I4m_Ma2026.cif\""; cat "$here/smoke.toml"; } >"$config"
        fastmace "$2.pieces.log" run "$config" --output "$2" --steps 3000
        fastmace "$2.pieces.log" run --resume "$2"
        ;;
    *)
        echo "usage: run.sh start RUN_DIR PROTOCOL STEPS | resume RUN_DIR STEPS | smoke RUN_DIR" >&2
        exit 2
        ;;
esac
