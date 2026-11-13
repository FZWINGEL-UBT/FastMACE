set -u
B=/mnt/homes/zwingel/t215
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -); [ -n "$gpu" ] && { echo "GPU busy $gpu"; exit 3; }
R=$B/runs/leeloo; S=$R/src; O=$R/out-nvrtc; mkdir -p $O/tmp
LIB=/mnt/homes/zwingel/environments/mace_cueq_oeq/lib/python3.10/site-packages/nvidia/cu13/lib/libnvrtc.so.13
export ENVROOT=/mnt/homes/zwingel/environments PYTHONPATH=$S FM_SRC=$S FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
 FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TMPDIR=$O/tmp FASTMACE_CACHE=$R/cache/b TORCHINDUCTOR_CACHE_DIR=$R/cache/b/inductor \
 TRITON_CACHE_DIR=$R/cache/b/triton MACE_COMPILE_CACHE=$R/cache/b/compile XDG_CACHE_HOME=$R/cache/b/xdg PYTHONUNBUFFERED=1 \
 PROFILE_OUT=$O/profile PROFILE_STEPS=20 REFUSE_FALLBACK=1 MEMORY_FRACTION=0.9 CUDA_CACHE_DISABLE=1
cd $S
for t in b1 n1 b2 n2; do
  extra=(); unset EXTRA_LD_LIBRARY_PATH; [ ${t:0:1} = n ] && { extra=(--constant "fastmace.kernels.oeq_streaming.runtime:NVRTC='$LIB'"); export EXTRA_LD_LIBRARY_PATH=$(dirname $LIB); }
  PROFILE_TAG=$t timeout 900 bash $B/jobs/py-b.sh $B/jobs/profile_point.py point --structure SrTiO3 --repeat 10x10x10 --model omat0-medium \
    --precision float32 --output $O/points/$t --warmup 200 --steps 1000 --timestep-fs 1 "${extra[@]}" > $O/$t.log 2>&1
  echo "$t $(grep -h '^point' $O/$t.log | cut -c1-90)"
done
for t in b1 n1 b2 n2; do python3 -c "
import json,sys;d=json.load(open('$O/profile/kernels-$t.json'));k={r['name']:r for r in d['kernels']}
print('$t', {n:round(k[n]['us_per_step'],1) for n in k if n in ('oeq_projected_backward_wprime','spline_forward0','sc_generated_backward','backward')}, 'total', round(d['info']['us_per_step_total']))"; done
grep -h -i "nvrtc\|constant" $O/n1.log | head -5 | cut -c1-200
