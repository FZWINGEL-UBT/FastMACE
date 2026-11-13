# Does cuBLAS 13.4 alone (no torch upgrade) give the Ada its GEMM gain? a = production 2.14.1 env, d = a clone of it with
# nvidia-cublas 13.4.1.3 (the 2.15 cu132 wheel's cuBLAS). Processes a d a d d a d a, SrTiO3 10^3 and NbCl5, then profiles.
set -u
B=/mnt/homes/zwingel/t215; export UV_CACHE_DIR=$B/uv-cache TMPDIR=$B/tmp
E=/mnt/homes/zwingel/environments/mace_cueq_oeq_cublas134
if [ ! -d $E ]; then
  $B/uvpkg/bin/uv venv -q --seed --python /mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python3.10 $E 2>/dev/null || \
    $B/uvpkg/bin/uv venv -q --seed --python 3.10 $E
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python -r /mnt/homes/zwingel/environments/mace_cueq_oeq.requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python nvidia-cublas==13.4.1.3
  $B/uvpkg/bin/uv pip freeze --python $E/bin/python > $E.requirements.txt
  rm -f /home/zwingel/.local/bin/python3.10
fi
$E/bin/python -c "import sys,torch;print(sys.version.split()[0],torch.__version__)"; grep -i "nvidia-cublas\|^torch=" $E.requirements.txt
sed 's|mace_cueq_oeq$|mace_cueq_oeq_cublas134|' $B/jobs/py-a.sh > $B/jobs/py-d.sh
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -); [ -n "$gpu" ] && { echo "GPU busy $gpu"; exit 3; }
R=$B/runs/leeloo; S=$R/src; O=$R/out-cublas; mkdir -p $O/tmp
export ENVROOT=/mnt/homes/zwingel/environments PYTHONPATH=$S FM_SRC=$S FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
 FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TMPDIR=$O/tmp PYTHONUNBUFFERED=1 \
 PROFILE_OUT=$O/profile REFUSE_FALLBACK=1 MEMORY_FRACTION=0.9 CUDA_CACHE_DISABLE=1
cd $S
run() { local e=$1 tag=$2 st=$3 rep=$4 steps=$5; local c=$R/cache/$e
  FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile XDG_CACHE_HOME=$c/xdg \
  PROFILE_TAG=$tag PROFILE_STEPS=$steps timeout 900 bash $B/jobs/py-$e.sh $B/jobs/profile_point.py point --structure $st --repeat $rep \
    --model omat0-medium --precision float32 --output $O/points/$tag --warmup 200 --steps 1000 --timestep-fs 1 > $O/$tag.log 2>&1
  echo "$tag $(grep -a -h '^point' $O/$tag.log | cut -c1-80)"; }
run d warm-d SrTiO3 10x10x10 0
for st in SrTiO3:10x10x10 NbCl5:3x3x3; do s=${st%%:*} r=${st#*:} i=0
  for e in a d a d d a d a; do i=$((i+1)); run $e ab-$s-$i$e $s $r 0; done; done
run a prof-a SrTiO3 10x10x10 20; run d prof-d SrTiO3 10x10x10 20
for t in a d; do python3 -c "
import json;d=json.load(open('$O/profile/kernels-prof-$t.json'));print('$t',{k:round(v) for k,v in d['info']['us_per_step_by_category'].items()})"; done
echo CUBLAS DONE
