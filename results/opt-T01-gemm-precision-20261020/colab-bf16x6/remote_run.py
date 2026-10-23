import subprocess

W = "/content/bf16x6"


def sh(cmd, timeout=3000):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
    return r.returncode, (r.stdout + r.stderr).strip()


print(sh(f"nvidia-smi > {W}/nvidia-smi.txt 2>&1; nvidia-smi --query-compute-apps=pid,name --format=csv >> {W}/nvidia-smi.txt;"
         f" head -12 {W}/nvidia-smi.txt")[1])
print(sh(f"cd {W} && venv/bin/python -c 'import sys, torch, triton; print(sys.version.split()[0], torch.__version__,"
         f" triton.__version__, torch.version.cuda)' > versions.txt 2>&1; venv/bin/pip list 2>/dev/null >> versions.txt;"
         f" cat versions.txt | head -3")[1])
code, out = sh(f"cd {W} && env -u NVIDIA_TF32_OVERRIDE venv/bin/python probe_bf16x6.py summary.json > probe.log 2>&1;"
               f" echo exit=$?; tail -60 probe.log")
print(out)
print(sh(f"cd {W} && tar -czf result.tgz summary.json probe.log nvidia-smi.txt versions.txt probe_bf16x6.py")[1])
