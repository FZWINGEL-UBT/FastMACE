set -u
B=/mnt/homes/zwingel/t215; export UV_CACHE_DIR=$B/uv-cache TMPDIR=$B/tmp
E=/mnt/homes/zwingel/environments/mace_cueq_oeq_cublas138
if [ ! -d $E ]; then
  $B/uvpkg/bin/uv venv -q --seed --python /mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python3.10 $E
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python -r /mnt/homes/zwingel/environments/mace_cueq_oeq.requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python nvidia-cublas==13.8.2.12
  $B/uvpkg/bin/uv pip freeze --python $E/bin/python > $E.requirements.txt
fi
diff <(sort /mnt/homes/zwingel/environments/mace_cueq_oeq.requirements.txt) <(sort $E.requirements.txt)
cat $E/pyvenv.cfg | head -3
