# Offline NVRTC census (conductor, 12 Nov 2026): w' backward (A100 and H200 renders) and the w' forward (spline_forward0,
# A100 production capture) compiled with NVRTC 13.0 / 13.2 / 13.4 / 13.5 for sm_80 and sm_90; registers and SASS
# instructions (census_sass.py). CPU only. Each NVRTC is a pip wheel in its own --target dir, loaded through
# fastmace.kernels.oeq_streaming.runtime.NVRTC (a child process preloads that libnvrtc.so.13).
set -u
B=/mnt/homes/zwingel/t215; C=$B/census; S=$B/src-b7c19b19; mkdir -p $C/out
export UV_CACHE_DIR=$B/uv-cache TMPDIR=$B/tmp CUDA_VISIBLE_DEVICES=
PY=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do
  [ -d $C/nvrtc-$v ] || $B/uvpkg/bin/uv pip install -q --python $PY --target $C/nvrtc-$v --no-deps nvidia-cuda-nvrtc==$v
done
W=$S/results/wprime-h200-20261111/sources; F=$S/results/a100-colab-ideas-20261111/batch2/ghoist-prod/sources
mkdir -p $C/src; cp $W/wprime-a100.cu $C/src/wprime-a100.cu; cp $W/wprime-h200-skip+hoist.cu $C/src/wprime-h200.cu
cp $F/real-0.cu $C/src/spline-forward-a100.cu
cd $S
export PATH=$(dirname $PY)/../lib/python3.10/site-packages/triton/backends/nvidia/bin:$PATH
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do
  lib=$(ls $C/nvrtc-$v/nvidia/cu13/lib/libnvrtc.so.13)
  LD_LIBRARY_PATH=$(dirname $lib) PYTHONPATH=$S $PY -W ignore $B/jobs/census_nvrtc.py $lib $C/out/census-$v.json     $C/src/wprime-a100.cu $C/src/wprime-h200.cu $C/src/spline-forward-a100.cu > $C/out/census-$v.log 2>&1
  echo "== NVRTC $v exit=$?"
done
$PY - <<'P'
import json
for v in ("13.0.88", "13.2.86", "13.4.92", "13.5.37"):
    for r in json.load(open(f"/mnt/homes/zwingel/t215/census/out/census-{v}.json"))["rows"]:
        print(v, r["source"], "sm", r["arch"], {k: r.get(k) for k in ("registers", "local", "instructions", "ffma", "ldl_stl", "per_edge", "per_edge_non_fma", "channel_pass", "error")})
P
