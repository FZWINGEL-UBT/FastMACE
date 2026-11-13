set -e
B=/mnt/homes/zwingel/t215
export UV_CACHE_DIR=$B/uv-cache UV_PYTHON_INSTALL_DIR=$B/uv-python TMPDIR=$B/tmp
UV=$B/uvpkg/bin/uv
for pair in mace_cueq_oeq:mace_cueq_oeq_torch215; do
  src=${pair%%:*} E=/mnt/homes/zwingel/environments/${pair#*:}
  [ -d $E ] || $UV venv -q --seed --python 3.11.16 $E
  grep -v -i -E '^(nvidia|cuda|torch==|triton==|exceptiongroup|tomli)' /mnt/homes/zwingel/environments/$src.requirements.txt > $B/req-$src.txt
  $UV pip install -q --python $E/bin/python -r $B/req-$src.txt "torch==2.15.0+cu132" --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match 2>&1 | tail -5
  $UV pip freeze --python $E/bin/python > $E.requirements.txt
  diff <(sed 's/ @.*//' /mnt/homes/zwingel/environments/$src.requirements.txt | tr A-Z a-z | tr _ - | sort) <(sed 's/ @.*//' $E.requirements.txt | tr A-Z a-z | tr _ - | sort) | grep -v -E 'nvidia|cuda-' | grep '^[<>]'
done
rm -f /home/zwingel/.local/bin/python3.11
/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python -c "import sys;print(sys.version)"
