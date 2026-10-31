#!/usr/bin/env bash
# M1's Ada leg on a Colab L4 (sm_89, root on the VM): Nsight Compute over one eager force call per large cell, as
# run.sh's ncu pass. First finds or installs ncu and checks that it can read counters on one small kernel; stops if
# not. Run through tools/colab/fastmace_colab.py --gpu L4 --models mh-1 --command 'bash results/l2-gap-audit-20261031/colab_l4.sh'.
set -uo pipefail
out=${OUT:?the driver sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader); source $(cat .source_commit)"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
if [ -z "$NCU" ]; then
  note "no ncu: installing"
  (apt-get update && apt-get install -y nsight-compute || apt-get install -y "$(apt-cache search --names-only '^nsight-compute-[0-9]' | sort -V | tail -1 | cut -d' ' -f1)") \
    > "$out/apt.log" 2>&1
  NCU=$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
fi
note "ncu=$NCU $([ -n "$NCU" ] && "$NCU" --version | tail -1)"
[ -n "$NCU" ] || { note "STOP: ncu not available"; exit 3; }
cat > /tmp/probe.py <<'EOF'
import torch
x = torch.randn(1 << 20, device="cuda")
torch.cuda.synchronize()
(x * 2).sum()
torch.cuda.synchronize()
EOF
"$NCU" --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct python /tmp/probe.py > "$out/probe-ncu.log" 2>&1
if grep -q "ERR_NVGPUCTRPERM\|ERROR" "$out/probe-ncu.log" || ! grep -q "lts__t_sector_hit_rate" "$out/probe-ncu.log"; then
  note "STOP: ncu cannot read counters (probe-ncu.log)"; exit 4
fi
note "counters readable"
TOOL=tools/step_breakdown.py
COMMON=(--checkpoint "$FASTMACE_MODELS/mace-mh-1.model" --model mh1-r2scan --backend hybrid --temperature-K 1000
        --timestep-fs 1 --equilibrate-steps 100)
for spec in NbCl5-9216:NbCl5@4x4x8 NaNbO3-8640:NaNbO3@6x6x6; do
  name=${spec%%:*} structure=${spec#*:} dir=$out/$name
  mkdir -p "$dir"
  note "start $name main"
  python $TOOL --output "$dir" "${COMMON[@]}" --structure "$structure" --steps 300 > "$out/$name.log" 2>&1
  note "end $name main exit=$?"
  note "start $name ncu"
  "$NCU" --profile-from-start off --target-processes all --cache-control none --clock-control none \
    --metrics "$(python $TOOL --ncu-metrics),lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum" \
    --export "$dir/ncu" --force-overwrite \
    python $TOOL --pass ncu --output "$dir" "${COMMON[@]}" --structure "$structure" > "$out/$name-ncu.log" 2>&1
  note "end $name ncu exit=$?"
  "$NCU" --import "$dir/ncu.ncu-rep" --csv --page raw --print-units base > "$dir/ncu_raw.csv" 2> "$out/$name-csv.log"
  python $TOOL --output "$dir" --summarize-ncu "$dir/ncu_raw.csv" > "$out/$name-join.log" 2>&1
  note "join $name exit=$?"
  rm -f "$dir/ncu.ncu-rep" "$dir"/trace*.json
done
note done
