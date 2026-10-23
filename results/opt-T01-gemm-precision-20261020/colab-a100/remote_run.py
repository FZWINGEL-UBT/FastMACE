"""On the Colab A100: install torch 2.14.1+cu130 in a venv, unpack FastMACE, run tools/gemm_precision_bench.py."""
import os
import subprocess

V = "/content/tfv"
PY = f"{V}/bin/python"
W = "/content/tf32x3"


def sh(cmd, timeout=3000):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


os.makedirs(f"{W}/fm", exist_ok=True)
print(sh(f"tar -xf {W}/fastmace.tar -C {W}/fm && echo unpacked")[1])
if not os.path.exists(PY):
    code, out = sh(f"uv venv -q -p 3.10 {V} && uv pip install -q -p {PY} 'torch==2.14.1' numpy==2.2.6 ase==3.27.0 "
                   "--index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple")
    print("install", code, out[-800:])
code, out = sh(f"cd {W}/fm && PYTHONPATH={W}/fm {PY} tools/gemm_precision_bench.py --output {W}/out "
               f"--gemms {W}/g/NaCl/gemms.csv {W}/g/NbCl5/gemms.csv > {W}/bench.log 2>&1; echo exit=$?; "
               f"tail -15 {W}/bench.log; cd {W} && tar -czf result.tgz out bench.log")
print(out)
