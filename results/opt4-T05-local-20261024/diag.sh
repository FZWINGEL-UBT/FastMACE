set -u
R=$HOME/fastmace-opt4-t05/run SRC=$HOME/fastmace-opt4-t05/src-13a1983
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_CACHE=$R/cache-warm XDG_CACHE_HOME=$R/cache-warm/xdg TRITON_HOME=$R/cache-warm/triton-home TMPDIR=$R/tmp
TORCH_LOGS="recompiles,graph_breaks,dynamic" timeout 600 $HOME/environments/mace_cueq_oeq/bin/python $SRC/results/opt4-T05-probes/setup_md.py --output $R/diag-1-mh1-NaCl --structure NaCl --label diag > $R/diag-1-mh1-NaCl.log 2>&1
echo exit $?
