#!/bin/bash
# OPT3-T03 local GPU jobs: correctness first (stops on failure), then the timed A/Bs, one structure and one
# build order per local_gpu.py reservation (fair share: at most 15 min each, released between jobs).
set -u
W=$HOME/opt3-t03; mkdir -p $W/tmp $W/ab
SRC=$W/src  # a pinned copy of the branch
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export TMPDIR=$W/tmp TORCHINDUCTOR_COMPILE_THREADS=4 MACE_COMPILE_CACHE=$HOME/compile-cache/opt3-t03 TORCHINDUCTOR_CACHE_DIR=$HOME/compile-cache/opt3-t03-inductor
export FASTMACE_RESULTS=$W/gpu-results FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python
P=$HOME/environments/mace_cueq_oeq/bin/python
PT=$HOME/environments/mace_cueq_latest/bin/python
GPU="python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --wait-minutes 240 --min-available-gib 6"
BASE="species_tables=on fold_readout_tail=on fold_product_linear=on"
log() { echo "$(date -u +%FT%TZ) $*" >> $W/ab.log; }
job() {  # name, command...
  local name=$1; shift
  $GPU --purpose "OPT3-T03 $name (<15 min)" -- "$@" > $W/ab/$name.out 2> $W/ab/$name.err
  local status=$?
  log "$name exit $status"
  return $status
}
T="cd $SRC && nice $PT -m pytest -q -p no:cacheprovider tests/gpu/test_fused_integrator_gpu.py --basetemp=$W/tmp/gpu -rfs"
job stage bash -c "$T -k fused_stage" || { log "stop: stage test failed"; exit 1; }
job md-nvt bash -c "$T -k 'trajectories and nvt'" || { log "stop: nvt test failed"; exit 1; }
job md-restart-npt bash -c "$T -k 'trajectories and restart'" || { log "stop: restart/npt test failed"; exit 1; }
ab() {  # name, temperature, structure, order
  job $1 $P -W ignore $SRC/tools/ab_rewrites.py --output $W/ab/$1 --checkpoint $HOME/models/mace-mh-1.model \
    --structures $3 --temperature-K $2 --order $4 --a $BASE --b $BASE \
    --a-engine fused_integrator=false --b-engine fused_integrator=true
}
for structure in NaCl NbCl5; do
  for order in ab ba; do
    ab fused-300K-$structure-$order 300 $structure $order
  done
done
log "all done"
