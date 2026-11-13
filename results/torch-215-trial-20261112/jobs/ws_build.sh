set -e
B=/mnt/homes/zwingel/t215; mkdir -p $B
export UV_CACHE_DIR=$B/uv-cache UV_PYTHON_INSTALL_DIR=$B/uv-python UV_CONCURRENT_INSTALLS=2 TMPDIR=$B/tmp
mkdir -p $TMPDIR
P=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
[ -x $B/uvpkg/bin/uv ] || nice -n 19 $P -m pip install -q --target $B/uvpkg uv
UV=$B/uvpkg/bin/uv; $UV --version
E=/mnt/homes/zwingel/environments/mace_cueq_latest_torch215
nice -n 19 $UV python install 3.11.16 2>&1 | tail -1
[ -d $E ] || nice -n 19 $UV venv -q --seed --python 3.11.16 $E
R=/mnt/homes/zwingel/environments/mace_cueq_latest.requirements.txt
grep -v -i -E '^(nvidia|cuda|torch==|triton==|exceptiongroup|tomli)' $R > $B/req.txt
nice -n 19 $UV pip install -q --python $E/bin/python -r $B/req.txt "torch==2.15.0+cu132" --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match 2>&1 | tail -15
$UV pip freeze --python $E/bin/python > /mnt/homes/zwingel/environments/mace_cueq_latest_torch215.requirements.txt
diff <(sort $R) <(sort /mnt/homes/zwingel/environments/mace_cueq_latest_torch215.requirements.txt) | grep -v -i -E 'nvidia|cuda-' | head -30
CUDA_VISIBLE_DEVICES= $E/bin/python -W ignore -c "import sys,torch,triton,pytest;print(sys.version,torch.__version__,triton.__version__,pytest.__version__)"
