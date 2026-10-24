#!/bin/bash
# OPT4-T04: grouped_launch on generic MACE (plans/opt-4.md candidate 4), local RTX 5060 Ti (sm_120, WSL2).
# Usage: run.sh JOB...  Each job holds its own local_gpu.py reservation (<= 15 min); timed jobs (ab-*) hold a
# quiet one and run only in a window the conductor gave. SRC is a pinned export of main (git archive).
# Jobs:
#   gpu-test         tests/gpu/test_grouped_gemm_cuda.py (untimed)
#   gate-LABEL       tools/force_stress_gate.py --rewrites grouped_launch=on, NaCl and NbCl5 (untimed)
#   ab-LABEL-ab/ba   tools/ab_rewrites.py, a = the defaults, b = a + grouped_launch, 12 blocks, NaCl (STRUCTS),
#                    build order a-b with ABAB and b-a with ABBA (timed, quiet)
set -u
W=$HOME/fastmace-opt4/t04; mkdir -p $W/tmp
SRC=${SRC:-$HOME/fastmace-opt4/src-T04-13a1983}
STRUCTS=${STRUCTS:-NaCl}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export TMPDIR=$W/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_CACHE=$HOME/fastmace-cache/opt4-t04
export MACE_COMPILE_CACHE=$FASTMACE_CACHE/compile TORCHINDUCTOR_CACHE_DIR=$FASTMACE_CACHE/inductor TRITON_CACHE_DIR=$FASTMACE_CACHE/triton
P=$HOME/environments/mace_cueq_oeq/bin/python
PT=$HOME/environments/mace_cueq_latest/bin/python
GPU="python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --wait-minutes 600 --min-available-gib 6"
LAUNCH="bash $HOME/fastmace-baseline/1a-T18/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh $HOME/environments/mace_cueq_oeq $FASTMACE_CACHE/oeq-launcher $P"
declare -A CK=([omat0-medium]=mace-omat-0-medium.model [mp0-medium]=2023-12-03-mace-128-L1_epoch-199.model
  [mpa0-medium]=mace-mpa-0-medium.model [mp0b3-medium]=mace-mp-0b3-medium.model
  [matpes-pbe]=MACE-matpes-pbe-omat-ft.model [matpes-r2scan]=MACE-matpes-r2scan-omat-ft.model)
log() { echo "$(date -u +%FT%TZ) $*" >> $W/run.log; }
log "source $(cat $SRC/.source_commit) gpu $(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gpu-test)
      $GPU --purpose "OPT4-T04 grouped GEMM GPU test (<5 min)" -- bash -c \
        "cd $SRC && nice $PT -m pytest -q -p no:cacheprovider tests/gpu/test_grouped_gemm_cuda.py --basetemp=$W/tmp/gg -rfs" \
        > $W/$job.log 2>&1 ;;
    gate-*)
      label=${job#gate-}; rm -rf $W/$job
      $GPU --purpose "OPT4-T04 force gate $label NaCl NbCl5 (<15 min)" -- nice $P -W ignore $SRC/tools/force_stress_gate.py \
        --output $W/$job --engine fastmace --structures-root $HOME/fastmace-baseline/1a-T18 \
        --checkpoint $HOME/models/${CK[$label]} --model $label --structures NaCl NbCl5 --rewrites grouped_launch=on \
        --reference-command $PT --production-command "$LAUNCH" > $W/$job.log 2>&1 ;;
    ab-*-ab|ab-*-ba)
      order=${job##*-}; label=${job#ab-}; label=${label%-*}
      pattern=$([ $order = ab ] && echo abab || echo abba)
      $GPU --quiet --purpose "OPT4-T04 timed A/B $label $order $STRUCTS (<15 min)" -- $P $SRC/tools/ab_rewrites.py \
        --output $W/$job --checkpoint $HOME/models/${CK[$label]} --model $label --structures $STRUCTS \
        --a grouped_launch=off --b grouped_launch=on --order $order --pattern $pattern > $W/$job.log 2>&1 ;;
    abset-*)  # abset-LABEL1,LABEL2: both orders of each label in one quiet reservation
      labels=${job#abset-}; cmd=""
      for label in ${labels//,/ }; do
        for order in ab ba; do
          pattern=$([ $order = ab ] && echo abab || echo abba)
          cmd+="$P $SRC/tools/ab_rewrites.py --output $W/ab-$label-$order --checkpoint $HOME/models/${CK[$label]} \
            --model $label --structures $STRUCTS --a grouped_launch=off --b grouped_launch=on --order $order \
            --pattern $pattern > $W/ab-$label-$order.log 2>&1; echo \"\$(date -u +%FT%TZ) ab-$label-$order exit=\$?\" >> $W/run.log; "
        done
      done
      $GPU --quiet --purpose "OPT4-T04 timed A/Bs $labels $STRUCTS (<13 min)" -- bash -c "$cmd" > $W/$job.log 2>&1 ;;
    *) log "unknown job $job"; continue ;;
  esac
  log "$job exit=$?"
done
log "done $*"
