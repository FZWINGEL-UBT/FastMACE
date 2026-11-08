#!/bin/bash
# On Alex: force/stress gates, default plan and candidate, one job per model.
# mkgates.sh <feature> <tag> <candidate-args> <models...>
set -euo pipefail
B=/home/atuin/v130cb/v130cb18/fm-offpolar
feat=$1; tag=$2; cand=$3; shift 3
declare -A CK=([off23-small]=MACE-OFF23_small.model [off23-medium]=MACE-OFF23_medium.model
  [off23-large]=MACE-OFF23_large.model [off23b-medium]=MACE-OFF23b_medium.model [off24-medium]=MACE-OFF24_medium.model
  [polar1-s]=MACE-POLAR-1-S.model [polar1-m]=MACE-POLAR-1-M.model [polar1-l]=MACE-POLAR-1-L.model)
for m in "$@"; do
  case $m in
    off23-large) gs='$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif@2x2x2 $ORG/water_216.extxyz@1x1x1' ;;
    off*) gs='$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1' ;;
    polar*) gs='NaCl NbCl5' ;;
  esac
  body=$B/jobs/gate-$tag-$m.body
  cat > $body <<EOF
R=\$B/runs/gate-$tag-$m-\$SLURM_JOB_ID
OUT=\$R MODEL=$m CKPT=${CK[$m]} STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh
OUT=\$R MODEL=$m CKPT=${CK[$m]} STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=candidate GATE_ARGS="--seed 20261015 $cand" bash tools/offpolar_ab.sh
cat \$R/run.log
EOF
  echo "$m $(bash $B/jobs/mkjob.sh gate-$tag-$m $feat 04:00:00 $body)"
done
