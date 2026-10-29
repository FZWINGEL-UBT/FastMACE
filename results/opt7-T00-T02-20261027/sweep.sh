#!/usr/bin/env bash
# LunaTP backward parameter sweep (scratch venv overrides; patch_luna.py). Direct backward kernel timing.
cd ~/tmp/opt7
export PATH=$HOME/environments/lunatp_screen/bin:/usr/local/cuda-13.3/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
  CUDA_HOME=/usr/local/cuda-13.3 TORCH_EXTENSIONS_DIR=$HOME/tmp/torch_ext_luna PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
F="ops-mh1-r2scan-NaCl.pt ops-mh1-r2scan-NbCl3O.pt ops-mh1-r2scan-NbCl5.pt ops-mh1-r2scan-NaNbO3.pt"
rm -f sweep.jsonl
run() {  # label, then env assignments
  label=$1; shift
  env "$@" LUNA_DEBUG=1 ~/environments/lunatp_screen/bin/python luna_bench.py $F 2> sweep-$label.err \
    | sed "s/^{/{\"config\": \"$label\", /" >> sweep.jsonl
}
run default
for rw in 8 16 24 32; do for th in 32 64 128; do
  run rw${rw}-th${th} LUNA_RW=$rw LUNA_THREADS=$th
done; done
run vec2-rw16-th64 LUNA_VEC=2 LUNA_RW=16 LUNA_THREADS=64
run vec2-rw8-th64 LUNA_VEC=2 LUNA_RW=8 LUNA_THREADS=64
echo ALLDONE >> sweep.jsonl
