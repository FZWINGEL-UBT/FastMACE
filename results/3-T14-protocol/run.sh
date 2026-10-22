#!/usr/bin/env bash
# 3-T14: a swap-MC protocol run in pieces on the local GPU, under tools/ci/local_gpu.py run.
#
#   bash results/3-T14-protocol/run.sh start  RUN_DIR PROTOCOL STEPS   # relaxation + the first STEPS steps
#   bash results/3-T14-protocol/run.sh resume RUN_DIR STEPS            # the next STEPS steps (and the quench)
#   bash results/3-T14-protocol/run.sh smoke  RUN_DIR                  # smoke.toml: start 3000, resume the rest
#   bash results/3-T14-protocol/run.sh yaml   PROTOCOL OUT.yaml CHECKOUT  # for a host without tomllib/tomli:
#       the protocol as YAML, its shipped structure path made absolute under CHECKOUT (that host's copy)
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

# The config fastmace run reads: the TOML itself where $python reads TOML (tomllib, tomli), else the same
# mapping as YAML (JSON is YAML; fastmace run reads .yaml with safe_load), converted by a Python >= 3.11.
# A shipped protocol name becomes its file; its relative structure path is a file shipped with it.
config_for() {
    local source="$1" out="$2" data="$checkout/fastmace/protocols/data"
    case "$source" in *.yaml|*.yml) echo "$source"; return ;; esac  # made elsewhere (yaml_protocol below)
    [ -f "$source" ] || source="$data/$source.toml"
    if "$python" -c "import tomllib" 2>/dev/null || "$python" -c "import tomli" 2>/dev/null; then
        if [ "$(dirname "$(realpath "$source")")" = "$(realpath "$data")" ]; then
            echo "$(basename "$source" .toml)"
        else
            echo "$source"
        fi
        return
    fi
    python3 - "$source" "$out.yaml" "$data" <<'PY'
import json, os, sys, tomllib
source, out, data = sys.argv[1:]
with open(source, "rb") as handle:
    config = tomllib.load(handle)
shipped = os.path.realpath(os.path.dirname(source)) == os.path.realpath(data)
if shipped and "structure" in config and not os.path.isabs(config["structure"]):
    config["structure"] = os.path.join(data, config["structure"])
with open(out, "w") as handle:
    json.dump(config, handle, indent=1)
PY
    echo "$out.yaml"
}

case "$1" in
    start)
        mkdir -p "$(dirname "$2")"
        fastmace "$2.pieces.log" run "$(config_for "$3" "$2.protocol")" --output "$2" --steps "$4"
        ;;
    resume)
        fastmace "$2.pieces.log" run --resume "$2" --steps "$3"
        ;;
    smoke)
        config="$2.smoke.toml"
        mkdir -p "$(dirname "$2")"
        { echo "structure = \"$checkout/fastmace/protocols/data/NaNbOCl4_I4m_Ma2026.cif\""; cat "$here/smoke.toml"; } >"$config"
        fastmace "$2.pieces.log" run "$(config_for "$config" "$2.smoke")" --output "$2" --steps 3000
        fastmace "$2.pieces.log" run --resume "$2"
        ;;
    yaml)
        source="$2"; data="$checkout/fastmace/protocols/data"
        [ -f "$source" ] || source="$data/$source.toml"
        python3 - "$source" "$3" "$4/fastmace/protocols/data" <<'PY'
import json, os, sys, tomllib
source, out, target = sys.argv[1:]
with open(source, "rb") as handle:
    config = tomllib.load(handle)
if "structure" in config and not os.path.isabs(config["structure"]):
    config["structure"] = f"{target}/{config['structure']}"
with open(out, "w") as handle:
    json.dump(config, handle, indent=1)
PY
        ;;
    *)
        echo "usage: run.sh start RUN_DIR PROTOCOL STEPS | resume RUN_DIR STEPS | smoke RUN_DIR | yaml PROTOCOL OUT CHECKOUT" >&2
        exit 2
        ;;
esac
