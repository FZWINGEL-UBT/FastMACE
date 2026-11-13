#!/bin/bash
# Environment p11 (12 November 2026): production's pins (torch 2.14.1+cu130, Triton 3.8.0) on Python 3.11.16 instead of
# 3.10.12, to separate the interpreter from Triton 3.9 (t9). Works on a workstation (NAS envs, uv in t215/uvpkg) and in
# WSL (~/environments, ~/.local/bin/uv). No GPU use.
set -u
if [ -d /mnt/homes/zwingel/environments ] && [ "$(hostname)" != "$(cat /etc/hostname 2>/dev/null)x" ] && [ -x /mnt/homes/zwingel/t215/uvpkg/bin/uv ]; then
  ENVS=/mnt/homes/zwingel/environments UV=/mnt/homes/zwingel/t215/uvpkg/bin/uv T=/mnt/homes/zwingel/t215/tmp
  export UV_CACHE_DIR=/mnt/homes/zwingel/t215/uv-cache TMPDIR=$T
else
  ENVS=$HOME/environments UV=$HOME/.local/bin/uv T=$HOME/fm-t215
fi
E=$ENVS/mace_cueq_oeq_py311
if ! $E/bin/python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
  rm -rf $E; $UV venv -q --seed --python $(readlink -f $ENVS/mace_cueq_oeq_torch215/bin/python) $E
  grep -v -i -E '^(exceptiongroup|tomli)==' $ENVS/mace_cueq_oeq.requirements.txt | grep -v '@ file' > $T/req-p11.txt
  $UV pip install -q --python $E/bin/python -r $T/req-p11.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  $UV pip freeze --python $E/bin/python > $E.requirements.txt
fi
diff <(sort $ENVS/mace_cueq_oeq.requirements.txt) <(sort $E.requirements.txt)
ENVROOT=$ENVS CUDA_VISIBLE_DEVICES= bash $(dirname $0)/py-p11.sh -c "import sys, torch, triton; print(sys.version.split()[0], torch.__version__, triton.__version__)" 2>&1 | tail -1
