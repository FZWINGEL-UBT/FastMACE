# Environment of the 1b-T03b runs on murphy (source it from the checkout root).
E=/mnt/homes/zwingel/environments/mace_cueq_latest
SP=$E/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib
export PYTHONPATH=/mnt/homes/zwingel/fastmace-1b-T03b PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/mnt/homes/zwingel/fastmace-cache/tmp
export XDG_CACHE_HOME=/mnt/homes/zwingel/fastmace-cache/xdg
export FASTMACE_CACHE=/mnt/homes/zwingel/fastmace-cache/fastmace
export TORCHINDUCTOR_COMPILE_THREADS=4
export FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/1a-T18
export FASTMACE_HYBRID_PYTHON=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
PY=$E/bin/python
