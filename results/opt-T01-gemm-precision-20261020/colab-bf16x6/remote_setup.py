import os
import subprocess

W = "/content/bf16x6"
os.makedirs(W, exist_ok=True)


def sh(cmd, timeout=1500):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
    return r.returncode, (r.stdout + r.stderr).strip()


code, out = sh(
    f"set -e; cd {W}; (command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh) > uv_install.log 2>&1;"
    f" export PATH=$HOME/.local/bin:$PATH; uv venv -p 3.13 venv > venv.log 2>&1;"
    f" uv pip install -p venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 > pip.log 2>&1;"
    f" venv/bin/python -c 'import torch, triton; print(torch.__version__, triton.__version__, torch.version.cuda)'")
print("setup exit", code)
print(out[-3000:])
