cd /mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671
export PATH=/home/fzwin/environments/mace_cueq_oeq_polar/bin:$PATH PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671:/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671/tools PYTHONDONTWRITEBYTECODE=1
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 FASTMACE_MODELS=$HOME/models
OUT=/home/fzwin/fm-runs/polar-switches-ab-20261107/polar1-m-mm-rerun2 MODEL=polar1-m CKPT=MACE-POLAR-1-M.model STRUCTS="NaCl NbCl5" SETS_FILE=/home/fzwin/fm-runs/polar-switches-ab-20261107/sets/polar-m-mm.tsv WARMUP=1200 BLOCK=400 BLOCKS=12 STOP_AFTER=4   bash tools/offpolar_ab.sh
OUT=/home/fzwin/fm-runs/polar-switches-ab-20261107/polar1-l-spinmm MODEL=polar1-l CKPT=MACE-POLAR-1-L.model STRUCTS="NaCl NbCl5" SETS_FILE=/home/fzwin/fm-runs/polar-switches-ab-20261107/sets/polar-l.tsv BLOCK=400 BLOCKS=12 STOP_AFTER=4   bash tools/offpolar_ab.sh
grep -H "b/a" /home/fzwin/fm-runs/polar-switches-ab-20261107/polar1-m-mm-rerun2/ab-*.log /home/fzwin/fm-runs/polar-switches-ab-20261107/polar1-l-spinmm/ab-*.log
