cd /mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671
export PATH=/home/fzwin/environments/mace_cueq_oeq_polar/bin:$PATH PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671:/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671/tools PYTHONDONTWRITEBYTECODE=1
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 FASTMACE_MODELS=$HOME/models
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23-large:MACE-OFF23_large.model; do
  m=${p%%:*}; c=${p#*:}
  OUT=/home/fzwin/fm-runs/offlocal-ab-20261107/$m MODEL=$m CKPT=$c STRUCTS="/mnt/c/Users/fzwin/AppData/Local/Temp/claude/f88975f7f53aa3f2/90f7a089-dfeb-401d-92e7-193574c2ffdf/scratchpad/organic/water_216.extxyz@1x1x1" SETS_FILE=/home/fzwin/fm-runs/offlocal-ab-20261107/sets/off.tsv     WARMUP=500 BLOCK=1000 BLOCKS=16 STOP_AFTER=4 bash tools/offpolar_ab.sh
done
grep -H "b/a" /home/fzwin/fm-runs/offlocal-ab-20261107/*/ab-*.log
