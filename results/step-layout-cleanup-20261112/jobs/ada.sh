#!/usr/bin/env bash
# step-layout-cleanup on an RTX 4000 Ada workstation (12 November 2026), OMAT-0 medium float32, production plan:
#   ab-ab / ab-ba  the same-process MD-step A/B (tools/bench/ab_swap.py), SrTiO3 10x10x10 and NbCl5, 1 fs, skin 0.5,
#           6 x 256 steps. Side a runs main's code: no bound species plan (every forward sorts, inside the capture),
#           reshape_irreps as a copy, JOINED_LAYOUT off; side b the branch (plan bound, views, joined layout on the
#           Triton backend).
#   jl-ab / jl-ba  JOINED_LAYOUT alone (side a off, side b the Ada default on), the same protocol
#   ld-* / lds-*  item 5: side b the CUTLASS backend on sm_89 with LINEAR_DIV (lds: and SEGMENT_ROWS), a today's default
#   prof    (PROFC: extra --constant items, e.g. item 5's side b) the torch 2.15 attribution (t215 profile_point.py + attribution.py, PYSTACKS=1, cupti-python) of the
#           new step, SrTiO3 10x10x10 and NbCl5 3x3x3
# Usage: setsid nohup bash ada.sh SRC OUT JOB... > OUT.out 2>&1 < /dev/null &
# Before each job: no process on the GPU, and no process of another user or ours above 50 % CPU (LC_ALL=C).
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
  note "$1 idle: load=$(cut -d' ' -f1-3 /proc/loadavg) mem=$(LC_ALL=C nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
site=$E/lib/python3.10/site-packages
prod_env() {
  export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib PYTHONPATH=$SRC:$SRC/tests \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTMACE_MODELS=$H/ResearchModule/Models FASTMACE_STRUCTURES=$B \
    FASTMACE_SOURCE_COMMIT=$(cat $SRC/.source_commit) FASTMACE_HYBRID_PYTHON=$E/bin/python TMPDIR=$OUT/tmp \
    FASTMACE_CACHE=$OUT/cache TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor TRITON_CACHE_DIR=$OUT/cache/triton \
    MACE_COMPILE_CACHE=$OUT/cache/compile XDG_CACHE_HOME=$OUT/cache/xdg
}
note "host=$(hostname) gpu=$(LC_ALL=C nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) src=$(cat $SRC/.source_commit)"
# SrTiO3 (cubic, a = 3.905 A) as a file for read_structure (tools/bench/capacity_scan.py's cell)
[ -f $OUT/SrTiO3.cif ] || (prod_env; cd $SRC && $E/bin/python -c "
import sys; sys.path.insert(0, 'tools/bench'); import capacity_scan as cs; from ase.io import write
write('$OUT/SrTiO3.cif', cs.base_cell('SrTiO3')[1])" >> $OUT/cif.log 2>&1)
ST=$OUT/SrTiO3.cif
K="fastmace.kernels.triton.species_gemm:bound_plan=no_bound_plan fastmace.rewrites.generic.modules:_reshaped=reshape_copy"
A="fastmace.kernels.triton.compose_skip:JOINED_LAYOUT=False"
# item 5 (conductor, 12 November 2026): the CUTLASS tf32x3 backend on sm_89, as ada-5060-screen built it
KC="fastmace.kernels.triton.tf32x3_gemm:CUTLASS_CAPABILITIES=((8,0),(8,9),(9,0))"
for job in "$@"; do
  idle "$job" || exit 3
  note "start $job"
  case $job in
    ab-ab|ab-ba) order=${job#ab-}; pattern=$([ $order = ab ] && echo abab || echo abba)
           (prod_env; cd $SRC && $E/bin/python tools/bench/ab_swap.py --swap $K --output $OUT/$job --checkpoint $CK \
              --model omat0-medium --structures $ST@10x10x10 NbCl5 --warmup 200 --block 256 --blocks 6 \
              --order $order --pattern $pattern --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
              --a-constant $A > $OUT/$job.log 2>&1 < /dev/null
            echo "$job exit=$? $(grep -h -E 'b/a|wrong side' $OUT/$job.log | tr '\n' ' ' | cut -c1-600)") | tee -a "$log" ;;
    jl-ab|jl-ba) order=${job#jl-}; pattern=$([ $order = ab ] && echo abab || echo abba)
           (prod_env; cd $SRC && $E/bin/python tools/ab_rewrites.py --output $OUT/$job --checkpoint $CK               --model omat0-medium --structures $ST@10x10x10 NbCl5 --warmup 200 --block 256 --blocks 6               --order $order --pattern $pattern --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5               --a-constant $A > $OUT/$job.log 2>&1 < /dev/null
            echo "$job exit=$? $(grep -h -E 'b/a' $OUT/$job.log | tr '
' ' ' | cut -c1-600)") | tee -a "$log" ;;
    ld-ab|ld-ba|lds-ab|lds-ba) order=${job#*-}; pattern=$([ $order = ab ] && echo abab || echo abba)
           B5=$KC; B5="$B5 fastmace.kernels.cutlass.linear_div:LINEAR_DIV=True"
           [ ${job%%-*} = lds ] && B5="$B5 fastmace.kernels.cutlass.segment_rows:SEGMENT_ROWS=True"
           (prod_env; cd $SRC && $E/bin/python tools/ab_rewrites.py --output $OUT/$job --checkpoint $CK               --model omat0-medium --structures $ST@10x10x10 NbCl5 --warmup 200 --block 256 --blocks 6               --order $order --pattern $pattern --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5               --b-constant $B5 > $OUT/$job.log 2>&1 < /dev/null
            echo "$job exit=$? $(grep -h -E 'b/a' $OUT/$job.log | tr '
' ' ' | cut -c1-600)") | tee -a "$log" ;;
    prof) for st in SrTiO3:10x10x10 NbCl5:3x3x3; do s=${st%%:*}; c=$OUT/pcache-$s; rm -rf $c; mkdir -p $c
            (cd $SRC && export ENVROOT=$H/environments PYTHONPATH=$SRC FM_SRC=$SRC FASTMACE_MODELS=$H/ResearchModule/Models \
              FASTMACE_STRUCTURES=$H/ResearchModule TMPDIR=$OUT/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
              FASTMACE_SOURCE_COMMIT=$(cat $SRC/.source_commit) REFUSE_FALLBACK=1 MEMORY_FRACTION=0.9 CUDA_CACHE_DISABLE=1 \
              PYSTACKS=1 PROFILE_OUT=$OUT/profile PROFILE_STEPS=5 FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor \
              TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile XDG_CACHE_HOME=$c/xdg PROFILE_TAG=attr-$s
             timeout 1500 bash $T/jobs/py-c.sh $T/jobs/profile_point.py point --structure $s --repeat ${st#*:} \
              --model omat0-medium --precision float32 --output $OUT/points/$s --warmup 200 --steps 200 --timestep-fs 1 \
              --constant "torch._inductor.config:comment_origin=True" ${PROFC:-} > $OUT/attr-$s.log 2>&1
             echo "prof $s exit=$? $(grep -a '^point' $OUT/attr-$s.log | cut -c1-160)"
             ENVROOT=$H/environments bash $T/jobs/py-c.sh $T/jobs/attribution.py $OUT/profile/trace-attr-$s.json.gz \
               $OUT/profile/pystacks-attr-$s.json $c/inductor 5 $OUT/attribution-$s.json > $OUT/attribution-$s.log 2>&1
             echo "attribution $s exit=$?") | tee -a "$log"
          done ;;
  esac
  note "end $job"
done
rm -rf $OUT/cache $OUT/pcache-*/triton
note done
