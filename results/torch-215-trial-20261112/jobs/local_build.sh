#!/bin/bash
# Local WSL pieces of the CUDA library matrix (12 November 2026): NVRTC wheel dirs, cuBLAS overlays (py-c4o/c8o.sh) and
# the t9 env (torch 2.14.1 + Triton 3.9.0 on Python 3.11.16). No GPU use; checks which libcublas each variant loads.
set -u
J=$HOME/fm-t215/jobs; UV=$HOME/.local/bin/uv; command -v uv >/dev/null && UV=$(command -v uv)
[ -x "$UV" ] || UV=$(ls $HOME/.local/share/uv/../../bin/uv 2>/dev/null || ls /usr/local/bin/uv 2>/dev/null)
echo "uv $UV"
P=$HOME/environments/mace_cueq_oeq/bin/python
for v in 13.0.88 13.2.86 13.4.92 13.5.37; do d=$HOME/fm-t215/census/nvrtc-$v
  [ -e $d/nvidia/cu13/lib/libnvrtc.so.13 ] || $UV pip install -q --python $P --target $d --no-deps nvidia-cuda-nvrtc==$v; done
for v in 13.4.1.3 13.8.2.12; do d=$HOME/fm-t215/overlay/cublas-$v
  [ -e $d/nvidia/cu13/lib/libcublas.so.13 ] || $UV pip install -q --python $P --target $d --no-deps nvidia-cublas==$v; done
E=$HOME/environments/mace_cueq_oeq_triton39
if ! $E/bin/python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
  rm -rf $E; $UV venv -q --seed --python $HOME/environments/mace_cueq_oeq_torch215/bin/python $E
  grep -v -i -E '^(exceptiongroup|tomli)==' $HOME/environments/mace_cueq_oeq.requirements.txt | grep -v '@ file' > $HOME/fm-t215/req-t9.txt
  $UV pip install -q --python $E/bin/python -r $HOME/fm-t215/req-t9.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  $UV pip install -q --python $E/bin/python --no-deps triton==3.9.0
  $UV pip freeze --python $E/bin/python > $E.requirements.txt
fi
diff <(sort $HOME/environments/mace_cueq_oeq.requirements.txt) <(sort $E.requirements.txt)
for e in a c4o c8o t9 b; do CUDA_VISIBLE_DEVICES= bash $J/py-$e.sh -c "
import torch, triton, ctypes, sys
m = sorted(set(l.split()[-1] for l in open('/proc/self/maps') if 'libcublas.so' in l)); L = ctypes.CDLL(m[0]); r = []
for i in range(3):
    v = ctypes.c_int(); L.cublasGetProperty(i, ctypes.byref(v)); r.append(v.value)
print('$e', sys.version.split()[0], torch.__version__, 'triton', triton.__version__, 'cublas', r, m[0][-60:])" 2>&1 | tail -1; done
df -h ~ | tail -1
