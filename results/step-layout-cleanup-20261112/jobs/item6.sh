#!/usr/bin/env bash
# Item 6 (conductor, 12 November 2026): SC_GENERATED / SC_FACTORED for products[1] (OMAT-0 medium: 128x0e output, one
# output segment, 13 weight segments). Ada workstation.
#   sources  tools/sc0_lab.py --product 1 --sources-only: product 1's contraction captured from a SrTiO3 10x10x10 call,
#            the generated monomial and factored sources written (GPU, seconds)
#   census   those sources compiled offline with the production NVRTC for sm_80, sm_89 and sm_90 (CPU only;
#            t215 census_nvrtc.py): registers, spills, SASS instructions, FFMA
#   lab      tools/sc0_lab.py --product 1: production's division + cuEq forward + backward + division backward
#            ("proxy") against the generated pairs, kernel time with L2 flushed
# Usage: setsid nohup bash item6.sh SRC OUT JOB... > OUT.out 2>&1 < /dev/null &
set -u -o pipefail
SRC=$1; OUT=$2; shift 2
H=/mnt/homes/zwingel
E=$H/environments/mace_cueq_oeq
CK=$H/ResearchModule/Models/mace-omat-0-medium.model
B=$H/fastmace-baseline/1a-T18
T=$H/t215
mkdir -p "$OUT/tmp"
log=$OUT/run.log
note() { echo "$(TZ=Europe/Berlin date +%FT%T%z) $*" | tee -a "$log"; }
idle() {
  local pids busy
  pids=$(LC_ALL=C nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(LC_ALL=C ps -eo pid:10,user:20,pcpu,comm --sort=-pcpu | awk -v me=$$ 'NR > 1 && $3 > 50 && $1 != me {print $2 ":" $4 ":" $3}' | paste -sd, -)
  if [ -n "$pids" ] || [ -n "$busy" ]; then note "$1 NOT IDLE gpu=[$pids] cpu=[$busy]"; return 1; fi
  note "$1 idle: load=$(cut -d' ' -f1-3 /proc/loadavg)"
}
site=$E/lib/python3.10/site-packages
prod_env() {
  export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib PYTHONPATH=$SRC:$SRC/tools \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTMACE_MODELS=$H/ResearchModule/Models FASTMACE_STRUCTURES=$B \
    FASTMACE_SOURCE_COMMIT=$(cat $SRC/.source_commit) FASTMACE_HYBRID_PYTHON=$E/bin/python TMPDIR=$OUT/tmp \
    FASTMACE_CACHE=$OUT/cache TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor TRITON_CACHE_DIR=$OUT/cache/triton \
    MACE_COMPILE_CACHE=$OUT/cache/compile XDG_CACHE_HOME=$OUT/cache/xdg
}
note "host=$(hostname) gpu=$(LC_ALL=C nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src=$(cat $SRC/.source_commit)"
[ -f $OUT/SrTiO3.cif ] || (prod_env; cd $SRC && $E/bin/python -c "
import sys; sys.path.insert(0, 'tools/bench'); import capacity_scan as cs; from ase.io import write
write('$OUT/SrTiO3.cif', cs.base_cell('SrTiO3')[1])" >> $OUT/cif.log 2>&1)
for job in "$@"; do
  [ $job = census ] || idle "$job" || exit 3
  note "start $job"
  case $job in
    sources) (prod_env; cd $SRC && $E/bin/python tools/sc0_lab.py --checkpoint $CK --structure $OUT/SrTiO3.cif@10x10x10 \
                --product 1 --sources-only --output $OUT/sources > $OUT/sources.log 2>&1 < /dev/null
              echo "sources exit=$? $(grep -h -E '^factored|^sources' $OUT/sources.log | cut -c1-400)") | tee -a "$log" ;;
    census) (cd $SRC && export CUDA_VISIBLE_DEVICES= CENSUS_ARCHS="80 89 90" \
               PATH=$site/triton/backends/nvidia/bin:$PATH
             lib=$site/nvidia/cu13/lib/libnvrtc.so.13
             LD_LIBRARY_PATH=$(dirname $lib) PYTHONPATH=$SRC $E/bin/python -W ignore $T/jobs/census_nvrtc.py $lib \
               $OUT/census.json $OUT/sources/*.cu > $OUT/census.log 2>&1
             echo "census exit=$?"
             $E/bin/python - "$OUT/census.json" <<'P'
import json, sys
for r in json.load(open(sys.argv[1]))["rows"]:
    print("census", r["source"], "sm", r["arch"], {k: r.get(k) for k in ("registers", "local", "instructions", "ffma", "ldl_stl", "error")})
P
            ) | tee -a "$log" ;;
    lab) (prod_env; cd $SRC && $E/bin/python tools/sc0_lab.py --checkpoint $CK --structure $OUT/SrTiO3.cif@10x10x10 \
            --product 1 --output $OUT/lab > $OUT/lab.log 2>&1 < /dev/null
          echo "lab exit=$? $(tail -4 $OUT/lab.log | tr '\n' ' ' | cut -c1-600)") | tee -a "$log" ;;
  esac
  note "end $job"
done
rm -rf $OUT/cache
note done
