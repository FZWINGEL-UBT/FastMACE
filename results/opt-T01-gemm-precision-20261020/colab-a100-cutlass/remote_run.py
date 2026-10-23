import subprocess
W = "/content/cutlass_bench"


def sh(cmd, timeout=3300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


print(sh(f"nvidia-smi > {W}/nvidia-smi.txt; nvidia-smi --query-compute-apps=pid,name --format=csv >> {W}/nvidia-smi.txt;"
         f" nvcc --version > {W}/nvcc.txt; cat {W}/nvidia-smi.txt | head -15")[1])
code, out = sh(f"cd {W} && env -u NVIDIA_TF32_OVERRIDE ./bench src/shapes_in.tsv long.csv capture.json 50 4096 > bench.log 2>&1;"
               f" echo exit=$?; head -3 bench.log; tail -8 bench.log")
print(out)
print(sh(f"cd {W} && tar -czf result.tgz long.csv capture.json bench.log nvidia-smi.txt nvcc.txt cutlass_version.txt "
         f"build_command.txt obj/*.log")[1])
