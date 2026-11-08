cd /mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671
export PATH=/home/fzwin/environments/mace_cueq_oeq_polar/bin:$PATH PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671:/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671/tools PYTHONDONTWRITEBYTECODE=1
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 FASTMACE_MODELS=$HOME/models
for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model; do
  m=${p%%:*}; c=${p#*:}
  OUT=/home/fzwin/fm-runs/fold-ab-20261107/$m MODEL=$m CKPT=$c STRUCTS="NaCl NbCl5" SETS_FILE=/home/fzwin/fm-runs/fold-ab-20261107/sets/polar.tsv BLOCK=400 BLOCKS=12 STOP_AFTER=4     bash tools/offpolar_ab.sh
done
grep -H "b/a" /home/fzwin/fm-runs/fold-ab-20261107/*/ab-*.log
