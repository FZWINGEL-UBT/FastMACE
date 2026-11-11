#!/bin/bash
# Source snapshot of main b951e804 for the BINNED_SCREEN run (fastmace/, tools/, the SrTiO3 cell), outside every checkout.
set -eu
RUN=$HOME/fm-runs/binned-5060-20261111
rm -rf "$RUN/src"; mkdir -p "$RUN/src" "$RUN/out"
tar -xf "/mnt/c/Users/fzwin/AppData/Local/Temp/claude/06a4a8e8fc832e51/d904e1a1-8ed5-4484-a04f-77e04c5d61cc/scratchpad/binned-src.tar" -C "$RUN/src"
echo b951e804820d780b8f0dca921d9a212f2fbb580a > "$RUN/src/.source_commit"
ls "$RUN/src" && ls "$RUN/src/tools/bench/rebuild_parts.py" "$RUN/src/tools/ab_rewrites.py" "$RUN/src/results/loup-a100-20261103/SrTiO3.cif"
