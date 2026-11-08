#!/usr/bin/env bash
# Which prerequisites put the spline / composed-skip paths in use on the Ada: tools/spline_w_check.py on SrTiO3 4x4x4
# per candidate set (in-use flags; exit 1 = spline not in use).   ROOT=<dir> bash probe.sh SRC_DIR [FIRST]
set -u -o pipefail
FM=$1; first=${2:-1}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
read -r -a A <<< "$ALL"
H=("$w:HORNER=True")
D=("$w:STAGE_DOUT=True")
T=(--rewrites tf32x3_gemms=on)
K=('fastmace.kernels.triton.tf32x3_gemm:CUTLASS_CAPABILITIES=((8, 0), (8, 9), (9, 0))')
sets=(1 2 3 4 5)
for i in "${sets[@]}"; do
  [ "$i" -lt "$first" ] && continue
  case $i in
    1) c=("${A[@]}") ;;
    2) c=("${A[@]}" "${H[@]}") ;;
    3) c=("${A[@]}" "${H[@]}" "${D[@]}") ;;
    4) c=("${A[@]}" "${H[@]}" "${T[@]}") ;;
    5) c=("${A[@]}" "${H[@]}" "${K[@]}" "${T[@]}") ;;
  esac
  if why=$(others); then note "STOP before probe $i: $why"; exit 3; fi
  python tools/spline_w_check.py --checkpoint "$CK" --structure "$ST@4x4x4" --steps 5 --output $ROOT/probe$i.json \
    --constant "${c[@]}" > $ROOT/probe$i.log 2>&1 < /dev/null
  code=$?
  note "probe $i exit=$code extra=[${c[*]:${#A[@]}}] $(python - $ROOT/probe$i.json <<'P'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("no json", e); raise SystemExit
print("spline", r.get("spline_in_use"), "compose", r.get("compose_skip_in_use"), "joined", r.get("joined_layout_in_use"),
      [(w.get("in_use"), w.get("forward_stage_edges"), w.get("wprime_stage_edges")) for w in r.get("wprime", [])])
P
) $(grep -hE "Error|error:" $ROOT/probe$i.log | tail -n 1 | cut -c1-200)"
done
