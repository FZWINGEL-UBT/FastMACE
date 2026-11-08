#!/bin/bash
# On Alex: writes one screen body per model and submits it on the given feature (a100_80 / a100_40).
# mkscreens.sh <feature> <tag> [models...]
set -euo pipefail
B=/home/atuin/v130cb/v130cb18/fm-offpolar
feat=$1; tag=$2; shift 2
declare -A CK=([off23-small]=MACE-OFF23_small.model [off23-medium]=MACE-OFF23_medium.model
  [off23-large]=MACE-OFF23_large.model [off23b-medium]=MACE-OFF23b_medium.model [off24-medium]=MACE-OFF24_medium.model
  [polar1-s]=MACE-POLAR-1-S.model [polar1-m]=MACE-POLAR-1-M.model [polar1-l]=MACE-POLAR-1-L.model)
models=${*:-${!CK[@]}}
for m in $models; do
  case $m in
    off*) structs='$ORG/water_216.extxyz@1x1x1'; sets=$B/sets/off.tsv; t=03:00:00; tim="WARMUP=500 BLOCK=1000 BLOCKS=16" ;;
    polar1-s) structs='NaCl NbCl5'; sets=$B/sets/polar_s.tsv; t=04:00:00; tim="BLOCK=400 BLOCKS=12" ;;
    polar*) structs='NaCl NbCl5'; sets=$B/sets/polar.tsv; t=06:00:00; tim="BLOCK=400 BLOCKS=12" ;;
  esac
  body=$B/jobs/screen-$tag-$m.body
  cat > $body <<EOF
OUT=\$B/runs/screen-$tag-$m-\$SLURM_JOB_ID MODEL=$m CKPT=${CK[$m]} STRUCTS="$structs" SETS_FILE=$sets $tim bash tools/offpolar_ab.sh
cat \$B/runs/screen-$tag-$m-\$SLURM_JOB_ID/run.log
EOF
  echo "$m $(bash $B/jobs/mkjob.sh screen-$tag-$m $feat $t $body)"
done
