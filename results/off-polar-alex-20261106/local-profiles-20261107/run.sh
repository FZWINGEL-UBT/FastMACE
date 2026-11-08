cd /mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671
export PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671:/mnt/d/Margraf/FastMACE/.claude/worktrees/tc-gemm-screen-ba8671/tools PYTHONDONTWRITEBYTECODE=1 FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
WATER=/mnt/c/Users/fzwin/AppData/Local/Temp/claude/f88975f7f53aa3f2/90f7a089-dfeb-401d-92e7-193574c2ffdf/scratchpad/organic/water_216.extxyz@1x1x1
STOCK=fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True
SPIN=fastmace.longrange.polar:SPIN_BATCHED=True
one() {  # name model ckpt structure "constants" "rewrites"
  local n=$1 m=$2 c=$3 s=$4 k=$5 r=$6
  echo "=== $n $(date +%T)"
  /home/fzwin/environments/mace_cueq_oeq_polar/bin/python /mnt/c/Users/fzwin/AppData/Local/Temp/claude/f88975f7f53aa3f2/90f7a089-dfeb-401d-92e7-193574c2ffdf/scratchpad/with_constants.py $k -- tools/module_profile.py --output /home/fzwin/fm-runs/detailed-profiles-20261107/$n/module --checkpoint $HOME/models/$c     --model $m --structure $s ${r:+--rewrites $r} > /home/fzwin/fm-runs/detailed-profiles-20261107/$n.module.log 2>&1 || echo "module_profile failed: $n"
  /home/fzwin/environments/mace_cueq_oeq_polar/bin/python /mnt/c/Users/fzwin/AppData/Local/Temp/claude/f88975f7f53aa3f2/90f7a089-dfeb-401d-92e7-193574c2ffdf/scratchpad/graph_step_profile.py --checkpoint $HOME/models/$c --model $m --structure $s --output /home/fzwin/fm-runs/detailed-profiles-20261107/$n/step     ${k:+--constant $k} ${r:+--rewrites $r} > /home/fzwin/fm-runs/detailed-profiles-20261107/$n.step.log 2>&1 || echo "step profile failed: $n"
  grep -h '^{' /home/fzwin/fm-runs/detailed-profiles-20261107/$n.step.log | cut -c1-400
}
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23b-medium:MACE-OFF23b_medium.model off24-medium:MACE-OFF24_medium.model off23-large:MACE-OFF23_large.model; do
  m=${p%%:*}; c=${p#*:}
  one $m-default $m $c $WATER "" ""
  one $m-stock-tables $m $c $WATER "$STOCK" "radial_tables=on"
done
for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model; do
  m=${p%%:*}; c=${p#*:}
  for s in NaCl NbCl5; do
    one $m-$s-default $m $c $s "" ""
    one $m-$s-spin $m $c $s "$SPIN" ""
  done
done
