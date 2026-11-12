set -u
B=/mnt/homes/zwingel/t215; export UV_CACHE_DIR=$B/uv-cache TMPDIR=$B/tmp
E=/mnt/homes/zwingel/environments/mace_cueq_oeq_torch215_prof
if [ ! -d $E ]; then
  $B/uvpkg/bin/uv venv -q --seed --python 3.11.16 $E
  grep -v nfs-soft-locks /mnt/homes/zwingel/environments/mace_cueq_oeq_torch215.requirements.txt > $B/freeze215.txt
  grep nfs-soft-locks /mnt/homes/zwingel/environments/mace_cueq_oeq.requirements.txt >> $B/freeze215.txt
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python -r $B/freeze215.txt --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python cupti-python
  rm -f /home/zwingel/.local/bin/python3.11
fi
$B/uvpkg/bin/uv pip freeze --python $E/bin/python | grep -i -E "cupti|^torch="
nvidia-smi | sed -n 3p
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -); [ -n "$gpu" ] && { echo "GPU busy $gpu"; exit 3; }
R=$B/runs/leeloo; S=$R/src; O=$R/out; mkdir -p $O/tmp
export ENVROOT=/mnt/homes/zwingel/environments PYTHONPATH=$S FM_SRC=$S FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
 FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TMPDIR=$O/tmp FASTMACE_CACHE=$R/cache/c TORCHINDUCTOR_CACHE_DIR=$R/cache/c/inductor \
 TRITON_CACHE_DIR=$R/cache/c/triton MACE_COMPILE_CACHE=$R/cache/c/compile XDG_CACHE_HOME=$R/cache/c/xdg PYTHONUNBUFFERED=1 \
 PROFILE_OUT=$O/profile PROFILE_TAG=prof-c PROFILE_STEPS=20 PYSTACKS=1 REFUSE_FALLBACK=1 MEMORY_FRACTION=0.9
cd $S
timeout 1200 bash $B/jobs/py-c.sh $B/jobs/profile_point.py point --structure SrTiO3 --repeat 10x10x10 --model omat0-medium \
  --precision float32 --output $O/points/prof-c --warmup 200 --steps 1000 --timestep-fs 1 > $O/prof-c.log 2>&1
grep -v -E "mplstyle|pynvml|warnings.warn|jit.script|Pyparsing" $O/prof-c.log | grep -i -E "^profile|cuspy|cupti|annot|stack|error" | head -20 | cut -c1-400
