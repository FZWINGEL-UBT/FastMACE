#!/usr/bin/env bash
# The 3x float32 gate of the adopted 5060 Ti set (F of local_stage3.sh without SC_GENERATED + SC_FACTORED: X12
# removing them was a tie, 0.9995 / 0.9998, 0.9994 / 1.0026), OMAT-0 medium, NaCl, NbCl5, SrTiO3 2x2x2, through
# local_gpu.py after local_stage3.sh (whose own OMAT-0 gate did not start: gate.sh was missing in the WSL jobs dir).
# Supercells 2x2x2, not the Ada's 3x3x3: local_stage3's POLAR-1-M gate at 3x3x3 failed in its float64 NbCl5 reference
# ("CUDA driver error: device not ready / Failed to create GPU mapping", WSL), and the 16 GiB card must not be
# oversubscribed. Then the POLAR-1 adoption check (local_polar2.sh's body, gates also 2x2x2).
# RTX 5060 Ti POLAR-1 adoption check (11 November 2026), through local_gpu.py. Stage 1
# (polar.sh): the w' pair (wprime_backward + fused_folded_forward) faster for POLAR-1-M in both orders (NaCl 0.9597 /
# 0.9589, NbCl5 0.9380 / 0.9415), not for -L (NbCl5 1.0098 / 1.0097); gate_epilogue faster on NbCl5 for S, M, L (NaCl
# ties). Here the candidate defaults together with the adopted 5060 Ti constants A (with the pair on, POLAR-1-M's w'
# kernel also takes STAGE_EDGES/DOUT/SH): a = today's 5060 Ti defaults, b = candidate, both orders, NaCl + NbCl5; then
# the 3x float32 gate of each candidate.
J=$HOME/fm-ada5060/jobs S=$HOME/fm-ada5060/src-screen OUT=$HOME/fm-ada5060/runs/local-stage3
unset PYTORCH_CUDA_ALLOC_CONF
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin t=fastmace.rewrites.species
PW=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD RS=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
A="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $s:PACKED8=True $k:SORT_BY_LENGTH=True $t:GENERIC_FOLD_READOUT_TAIL=True $PW=True $RS=True $w:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True"
# shellcheck disable=SC2086
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen 5060 Ti set gate (d66)" -- \
  env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$OUT bash $J/gate.sh $S set-5060 omat0-medium mace-omat-0-medium.model \
  "NaCl NbCl5 results/loup-a100-20261103/SrTiO3.cif" "2 2 2" $A -- fold_product_linear=on
echo "$(date -u +%FT%TZ) gate-5060 exit=$?" >> $OUT/chain.log
OUT=$HOME/fm-ada5060/runs/local-polar2
mkdir -p $OUT
unset PYTORCH_CUDA_ALLOC_CONF
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin t=fastmace.rewrites.species
PW=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD RS=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
A="$s:SPLINE_W=True $s:R128=True $s:STAGE_EDGES=True $s:PACKED8=True $k:SORT_BY_LENGTH=True $t:GENERIC_FOLD_READOUT_TAIL=True $PW=True $RS=True $w:STAGE_EDGES=True $w:STAGE_DOUT=True $w:STAGE_SH=True"
cat > $OUT/job.sh <<EOF
source $J/common.sh
cd $S && export PYTHONPATH=$S
note "polar2 start fastmace=\$(cat .source_commit)"
for size in M S L; do
  model=polar1-\$(echo \$size | tr SML sml); ck=\$MODELS/MACE-POLAR-1-\$size.model
  if [ \$size = M ]; then RW="wprime_backward=on fused_folded_forward=on gate_epilogue=on"; else RW="gate_epilogue=on"; fi
  for order in ab ba; do
    pattern=\$([ "\$order" = ab ] && echo abab || echo abba)
    note "begin \$model \$order"; t0=\$(date +%s)
    python tools/ab_rewrites.py --output "\$ROOT/ab-\$model-\$order" --checkpoint "\$ck" --model "\$model" \
      --structures NaCl NbCl5 --warmup 300 --block 200 --blocks 8 --order "\$order" --pattern "\$pattern" \
      --b \$RW --b-constant $A > "\$ROOT/ab-\$model-\$order.log" 2>&1 < /dev/null
    note "end \$model \$order exit=\$? wall \$(( \$(date +%s) - t0 )) s"
    grep -h -E 'b/a' "\$ROOT/ab-\$model-\$order.log" | sed "s/^/  \$model \$order /" | tee -a "\$log"
  done
  bash $J/gate.sh $S \$model \$model MACE-POLAR-1-\$size.model "NaCl NbCl5" "2 2 2" $A -- \$RW
done
note DONE
EOF
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen POLAR-1 5060 Ti adoption (d66)" -- \
  env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$OUT bash $OUT/job.sh
echo "$(date -u +%FT%TZ) polar2 exit=$?" >> $HOME/fm-ada5060/runs/local-stage3/chain.log
