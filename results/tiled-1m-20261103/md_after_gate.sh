#!/bin/bash
# After the capacity gate on leto: the 1M MD check of 3a83c308 (tiled branch without the edge headroom).
W=/mnt/homes/zwingel/fm-tiled
while pgrep -u zwingel -f "[g]ate2_leto.sh" > /dev/null; do sleep 30; done
SRC=$W/src-all-3a83c308 bash $W/run.sh leto-md1m-branch2 tools/bench/tiled_md.py --structure SrTiO3 --model omat0-medium --repeat 58x58x58 --mode auto --path tiled --tile-list --species-index --receivers 16384 --steps 60 --timestep-fs 1 --temperature 600
