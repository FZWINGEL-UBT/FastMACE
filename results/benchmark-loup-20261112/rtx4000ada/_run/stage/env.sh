# Sourced by every job of this run: keeps all files under BASE (no /tmp, no /home writes).
export BASE=/mnt/homes/zwingel/benchmark-loup-20261112
export PREFIX=$BASE/prefix
export HOME=$BASE/home
export TMPDIR=$BASE/tmp/$(hostname); mkdir -p "$TMPDIR" "$HOME"
export XDG_CACHE_HOME=$BASE/cache/$(hostname)
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor
export CUDA_CACHE_PATH=$XDG_CACHE_HOME/nv
export UV_PYTHON_INSTALL_DIR=$PREFIX/uv-python
# Symmetrix-XL JIT cache only: private local-disk dir (conductor decision (a); /mnt/homes ancestors are 0777 non-sticky)
export JITROOT=/home/zwingel/benchmark-loup-20261112-jit
( umask 077; mkdir -p "$JITROOT" ); chmod 700 "$JITROOT"
export SYMMETRIX_JIT_CACHE=$JITROOT/cache
export JOBS=${JOBS:-32}
