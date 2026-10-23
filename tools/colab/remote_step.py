# Run one remote.sh stage detached on the Colab VM and wait for it (calling again reattaches).
# Sent by tools/colab/fastmace_colab.py as `colab exec -s NAME -f remote_step.py --env STEP=<id> --env STAGE=<stage>
# [--env ARG=<n>] --env WAIT_S=<s> [--env RESET=1] [--env KILL=1] --timeout <s+60>`.
# Prints one line "@@STEP {json}" at the end: {"step", "state": "done"|"running"|"killed", "exit", "seconds"}.
import json
import os
import signal
import subprocess
import time

FM = "/content/fm"
STEPS = f"{FM}/steps"
step, stage = os.environ["STEP"], os.environ["STAGE"]
arg = os.environ.get("ARG", "")
wait_s = float(os.environ.get("WAIT_S", "1200"))
if os.environ.get("RESET") == "1":
    # A new batch (also on a kept session): the previous batch's steps and records go.
    subprocess.run(f"rm -rf {STEPS} {FM}/out", shell=True)
os.makedirs(STEPS, exist_ok=True)
chunked = {}
for name in sorted(os.listdir("/content")):
    if not name.startswith("fm_"):  # the driver's uploads (fastmace_colab.STAGING)
        continue
    base, _, suffix = name.rpartition(".chunk")
    if base and suffix.isdigit():  # a large upload in pieces (fastmace_colab.split_file), joined in order
        chunked.setdefault(base, []).append(name)
    else:
        os.replace(f"/content/{name}", f"{FM}/{name[3:]}")
for base, names in chunked.items():
    with open(f"{FM}/{base[3:]}", "wb") as out:
        for name in names:
            with open(f"/content/{name}", "rb") as f:
                while block := f.read(1 << 24):
                    out.write(block)
            os.remove(f"/content/{name}")
log, rc, pidf = (f"{STEPS}/{step}.{suffix}" for suffix in ("log", "rc", "pid"))


def pid():
    try:
        return int(open(pidf).read())
    except (OSError, ValueError):
        return None


def alive():
    p = pid()
    if p is None:
        return False
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False


def tail(n):
    try:
        return open(log, errors="replace").read()[-n:]
    except OSError:
        return ""


if os.environ.get("KILL") == "1":
    if alive():
        os.killpg(pid(), signal.SIGTERM)
        time.sleep(5)
        if alive():
            os.killpg(pid(), signal.SIGKILL)
        with open(rc, "w") as f:
            f.write("143 0\n")
    print(tail(3000))
    print("@@STEP " + json.dumps(dict(step=step, state="killed", exit=143, seconds=None)), flush=True)
    raise SystemExit(0)

if not os.path.exists(rc) and not alive():
    # Clean environment: the Colab kernel exports MPLBACKEND=module://matplotlib_inline...,
    # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD and others that break or change the tests. Keep only the driver path
    # (libcuda lives in /usr/lib64-nvidia); /usr/local/cuda/bin is left off PATH, as in the Modal image (no nvcc).
    clean = dict(HOME="/root", LANG="C.UTF-8", LD_LIBRARY_PATH="/usr/lib64-nvidia",
                 PATH="/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    p = subprocess.Popen(["bash", "-c", f"start=$(date +%s); bash {FM}/remote.sh {stage} {arg} > {log} 2>&1; "
                          f"echo \"$? $(( $(date +%s) - start ))\" > {rc}.tmp && mv {rc}.tmp {rc}"],
                         start_new_session=True, env=clean, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    with open(pidf, "w") as f:
        f.write(str(p.pid))
start = time.time()
while not os.path.exists(rc) and time.time() - start < wait_s:
    time.sleep(2)
if os.path.exists(rc):
    code, seconds = open(rc).read().split()
    print(tail(6000))
    print("@@STEP " + json.dumps(dict(step=step, state="done", exit=int(code), seconds=int(seconds))), flush=True)
else:
    print(tail(1500))
    print("@@STEP " + json.dumps(dict(step=step, state="running", exit=None, seconds=None)), flush=True)
