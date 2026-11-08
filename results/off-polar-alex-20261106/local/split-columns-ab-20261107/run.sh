cd /mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 ~/environments/mace_cueq_latest/bin/python -m pytest -q -p no:cacheprovider tests/gpu/test_folded_tiles_cuda.py > /home/fzwin/fm-runs/split-columns-ab-20261107/gputest2.log 2>&1; tail -2 /home/fzwin/fm-runs/split-columns-ab-20261107/gputest2.log
export PATH=/home/fzwin/environments/mace_cueq_oeq_polar/bin:$PATH PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671:/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671/tools PYTHONDONTWRITEBYTECODE=1
export FASTMACE_STRUCTURES=/home/fzwin/fastmace-baseline/1a-T18 FASTMACE_MODELS=/home/fzwin/models
for p in off23-large:MACE-OFF23_large.model off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23b-medium:MACE-OFF23b_medium.model off24-medium:MACE-OFF24_medium.model; do
  m=${p%%:*}; c=${p#*:}
  OUT=/home/fzwin/fm-runs/split-columns-ab-20261107/$m MODEL=$m CKPT=$c STRUCTS="/mnt/c/Users/fzwin/AppData/Local/Temp/claude/f88975f7f53aa3f2/90f7a089-dfeb-401d-92e7-193574c2ffdf/scratchpad/organic/water_216.extxyz@1x1x1" SETS_FILE=/home/fzwin/fm-runs/split-columns-ab-20261107/sets/off.tsv     WARMUP=500 BLOCK=1000 BLOCKS=16 STOP_AFTER=4 bash tools/offpolar_ab.sh
done
grep -H "b/a" /home/fzwin/fm-runs/split-columns-ab-20261107/*/ab-*.log
