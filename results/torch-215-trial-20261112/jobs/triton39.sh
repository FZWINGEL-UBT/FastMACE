# Environment t9 (12 November 2026): production 2.14.1+cu130 with Triton 3.9.0 (torch 2.15's) instead of 3.8.0, to
# separate the Triton compiler's gain (tf32x3 gemm: 3.9 splits the operands in registers, 3.8 through shared memory)
# from torch 2.15. Triton 3.9.0 has wheels for cp311+ only, so t9 runs Python 3.11.16 (the 2.15 env's); torch 2.14.1
# pins triton==3.8.0, so 3.9.0 goes in with --no-deps. Then a CPU-side Inductor smoke.
set -u
B=/mnt/homes/zwingel/t215; export UV_CACHE_DIR=$B/uv-cache TMPDIR=$B/tmp
E=/mnt/homes/zwingel/environments/mace_cueq_oeq_triton39
if [ ! -x $E/bin/python ] || ! $E/bin/python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
  rm -rf $E; PY311=$(readlink -f /mnt/homes/zwingel/environments/mace_cueq_oeq_torch215/bin/python)
  $B/uvpkg/bin/uv venv -q --seed --python $PY311 $E
  grep -v -i -E '^(exceptiongroup|tomli)==' /mnt/homes/zwingel/environments/mace_cueq_oeq.requirements.txt > $B/tmp/req-t9.txt
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python -r $B/tmp/req-t9.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python --no-deps triton==3.9.0
  $B/uvpkg/bin/uv pip freeze --python $E/bin/python > $E.requirements.txt
fi
diff <(sort /mnt/homes/zwingel/environments/mace_cueq_oeq.requirements.txt) <(sort $E.requirements.txt)
ENVROOT=/mnt/homes/zwingel/environments CUDA_VISIBLE_DEVICES= bash $B/jobs/py-t9.sh -c "import torch, triton; print(torch.__version__, triton.__version__); import torch._inductor.compile_fx; print('inductor import ok')"
