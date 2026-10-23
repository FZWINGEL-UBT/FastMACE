import subprocess
W = "/content/cutlass_bench"
subprocess.run(f"mkdir -p {W}/src && tar -xf {W}/src.tar -C {W}/src", shell=True, check=True)
r = subprocess.run(f"bash {W}/src/build.sh", shell=True, capture_output=True, text=True, timeout=3300)
print("exit", r.returncode)
print((r.stdout + r.stderr)[-6000:])
