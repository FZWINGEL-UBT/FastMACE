"""FastMACE GPU jobs on Modal (MODAL-T01): one pinned source tarball, one command, one GPU, results back.

Run from WSL with the Modal client venv (``~/environments/modal_client``). The owner authenticates once
with ``modal token new``; this file never reads or prints the token.

Image: recipe C / hybrid (cuEquivariance + OpenEquivariance), Python 3.10, every package pinned from
``~/environments/mace_cueq_oeq.requirements.txt`` (``requirements-hybrid.txt`` here), with ``g++`` for
OpenEquivariance's host extension (FastMACE builds it in-process, ``fastmace/kernels/_oeq_runtime.py``; no
``nvcc``). Volume ``fastmace-data``: ``/data/models`` (checkpoints), ``/data/baseline/1a-T18`` (the
research-engine export with the benchmark structures), ``/data/cache/<gpu>`` (FastMACE, Inductor, Triton and
OEq caches, per GPU type, kept between runs).

Entry points (``modal run tools/modal/fastmace_modal.py::<name> ...``):

- ``upload``: models and baseline into the volume (``--models`` file names under ``~/models``).
- ``smoke``: ``nvidia-smi``, the image's ``pip freeze``, torch's CUDA view, ``tests/gpu/test_smoke.py``.
- ``job``: ``--tarball`` (a ``git archive`` of a commit, with ``.source_commit``) or ``--commit`` (archived
  from ``--repo``), ``--command`` (a shell command run from the source root; ``$OUT`` is the results
  directory), ``--gpu L4|A100|A100-80GB|H100``, ``--timeout-min``, ``--out`` (a local directory for the
  returned results).

Every job writes ``modal.json`` into its results: the GPU, driver, CUDA, image pip freeze hash, the source
commit, the wall time and the estimated cost at Modal's list prices (``PRICE_PER_HOUR``, check them on
modal.com/pricing). Timeouts are set on every function; containers stop when the call returns.

Budget guard (MODAL-T02, ``budget.py``): before every launch, this month's metered cost
(``modal.Workspace.billing.summary``) plus the worst case of the jobs still running from any checkout plus this
job's worst case (function timeout plus a start allowance, at the GPU, CPU-limit and memory-limit rates of
``modal.Workspace.billing.rates``) must stay within $27 (the $30 monthly credit minus a $3 reserve). Otherwise,
or if the billing cannot be read, the job is refused. ``--limit-usd`` (or ``FASTMACE_MODAL_LIMIT_USD``) sets a
lower limit; a higher one needs ``--owner-override "who: why"``, which the ledger records. After every job the
month ledger ``~/.fastmace-modal/modal-ledger.csv`` gets a row with the metered cost before and after (Modal's
metering lags by minutes, so the after value may not contain the whole job yet). The ledger and the list of
jobs in flight live in one directory for every checkout and worktree on the machine (``FASTMACE_MODAL_STATE``,
default ``~/.fastmace-modal``), so parallel jobs from different worktrees count against each other; the
conductor commits copies of the ledger to ``results/modal-ledger.csv``. The functions have hard CPU and memory
limits, no retries and a 2 s idle window.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# The budget guard runs on the host only. The container gets this file alone, so importing budget there
# failed every job at container start (22 October 2026, 3-T05 and 3-T16).
if modal.is_local():
    sys.path.insert(0, str(HERE))
    import budget  # noqa: E402  (tools/modal/budget.py: pure Python, tested on CPU)

APP_NAME = "fastmace"
VOLUME_NAME = "fastmace-data"
GPUS = ("L4", "A100", "A100-80GB", "H100")
# USD per GPU-hour at Modal's list prices of October 2026 (GPU only; CPU and memory are billed on top, a few %).
PRICE_PER_HOUR = {"L4": 0.80, "A100": 2.10, "A100-80GB": 2.50, "H100": 3.95}
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
# Hard container limits (cores, MiB): the budget's worst case is computed at these limits.
JOB_CPU, JOB_MEMORY_MIB = (4.0, 8.0), (16384, 32768)
LISTING_CPU, LISTING_MEMORY_MIB = (1.0, 2.0), (1024, 2048)
LISTING_TIMEOUT_S = 600
GRACE_S = 600  # the job function's timeout above the command's own
# Shared by every checkout and worktree (22 October 2026: per-checkout files let parallel jobs from different
# worktrees each pass the guard alone).
STATE = Path(os.path.expanduser(os.environ.get("FASTMACE_MODAL_STATE", "~/.fastmace-modal")))
LEDGER = STATE / "modal-ledger.csv"
IN_FLIGHT = STATE / "modal-in-flight.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("g++", "git", "procps")
    .pip_install_from_requirements(str(HERE / "requirements-hybrid.txt"), extra_index_url=TORCH_INDEX)
    .env({"PYTHONDONTWRITEBYTECODE": "1", "TORCHINDUCTOR_COMPILE_THREADS": "4"})
)


def _environment(gpu, src, out):
    cache = f"/data/cache/{gpu}"
    env = dict(os.environ)
    env.update(PYTHONPATH=str(src), FASTMACE_MODELS="/data/models", FASTMACE_STRUCTURES="/data/baseline/1a-T18",
               FASTMACE_CACHE=cache, TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor", TRITON_CACHE_DIR=f"{cache}/triton",
               MACE_COMPILE_CACHE=f"{cache}/compile", XDG_CACHE_HOME=f"{cache}/xdg", OUT=str(out),
               TMPDIR="/tmp/fastmace-tmp")
    os.makedirs("/tmp/fastmace-tmp", exist_ok=True)
    return env


def _sh(command, env=None, cwd=None, timeout=None):
    done = subprocess.run(command, shell=True, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return done.returncode, done.stdout + done.stderr


def _gpu_record():
    _, smi = _sh("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader")
    _, cuda = _sh("nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9.]+'")
    _, freeze = _sh("pip freeze")
    import hashlib

    digest = hashlib.sha256(freeze.encode()).hexdigest()
    return dict(nvidia_smi=smi.strip(), cuda=cuda.strip(), pip_freeze_sha256=digest, pip_freeze=freeze)


def _run(gpu, tarball: bytes, command: str, timeout_s: int) -> bytes:
    src, out = Path("/tmp/fastmace-src"), Path("/tmp/fastmace-out")
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tarball)) as archive:
        archive.extractall(src)
    commit = (src / ".source_commit").read_text().strip() if (src / ".source_commit").exists() else None
    env = _environment(gpu, src, out)
    record = dict(gpu_requested=gpu, source_commit=commit, command=command, **_gpu_record())
    (out / "pip-freeze.txt").write_text(record.pop("pip_freeze"))
    start = time.time()
    try:
        code, log = _sh(command, env=env, cwd=src, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        code, log = 124, f"timed out after {timeout_s} s\n{error.stdout or ''}{error.stderr or ''}"
    record.update(exit=code, wall_s=time.time() - start)
    record["estimated_gpu_cost_usd"] = record["wall_s"] / 3600 * PRICE_PER_HOUR.get(gpu, float("nan"))
    (out / "command.log").write_text(log)
    (out / "modal.json").write_text(json.dumps(record, indent=2) + "\n")
    volume.commit()  # keep the compile caches for the next run
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(out, arcname="out")
    return buffer.getvalue()


@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=3 * 3600, retries=0, scaledown_window=2,
              cpu=JOB_CPU, memory=JOB_MEMORY_MIB)
def run_job(gpu: str, tarball: bytes, command: str, timeout_s: int) -> bytes:
    """The command on the GPU this function was called with (``run_job.with_options(gpu=...)``)."""
    return _run(gpu, tarball, command, timeout_s)


@app.function(image=image, volumes={"/data": volume}, timeout=LISTING_TIMEOUT_S, retries=0, scaledown_window=2,
              cpu=LISTING_CPU, memory=LISTING_MEMORY_MIB)
def volume_listing() -> str:
    return _sh("cd /data && find . -maxdepth 3 | sort | head -200; du -sh /data/* 2>/dev/null")[1]


def _archive(commit, repo):
    """A git archive of ``commit`` with ``.source_commit``, as the pinned copies of AGENTS.md."""
    sha = subprocess.run(["git", "-C", repo, "rev-parse", commit], capture_output=True, text=True,
                         check=True).stdout.strip()
    data = subprocess.run(["git", "-C", repo, "-c", "core.autocrlf=false", "archive", "--format=tar", sha],
                          capture_output=True, check=True).stdout
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(data)) as source, tarfile.open(fileobj=out, mode="w") as dest:
        for member in source.getmembers():
            dest.addfile(member, source.extractfile(member) if member.isfile() else None)
        info = tarfile.TarInfo(".source_commit")
        payload = (sha + "\n").encode()
        info.size = len(payload)
        dest.addfile(info, io.BytesIO(payload))
    return sha, out.getvalue()


def _read_metered():
    return modal.Workspace.from_context().billing.summary().metered_cost


def _read_rates():
    return modal.Workspace.from_context().billing.rates()


def _guarded(launch, *, commit, gpu, function_timeout_s, cpu, memory_mib, limit_usd, owner_override):
    """``launch()`` only if the month's budget allows it; the ledger gets a row afterwards, whatever happens."""
    try:
        decision = budget.check(_read_metered, _read_rates, gpu, function_timeout_s, cpu[1], memory_mib[1] / 1024,
                                limit=limit_usd, override=owner_override, in_flight=IN_FLIGHT)
    except budget.BudgetRefused as refused:
        raise SystemExit(str(refused)) from None
    print(decision.message)
    started, code = time.time(), "error"
    with budget.holding(IN_FLIGHT, decision):
        try:
            result = launch()
            code = result.get("exit", "?") if isinstance(result, dict) else 0
            return result
        finally:
            wall = time.time() - started
            try:
                estimate = budget.cost(_read_rates(), gpu, wall, cpu[1], memory_mib[1] / 1024)
                after = _read_metered()
            except Exception:  # noqa: BLE001  (the row is still written; the next check reads billing again)
                estimate, after = float("nan"), None
            row = budget.record(LEDGER, decision, commit=commit, gpu=gpu, timeout_s=function_timeout_s, wall_s=wall,
                                estimated_cost=estimate, metered_after=after, exit_code=code)
            print("ledger:", LEDGER, json.dumps(row))


def _receive(result: bytes, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(result)) as archive:
        archive.extractall(dest)
    record = json.loads((dest / "out" / "modal.json").read_text())
    print(json.dumps({k: record[k] for k in ("gpu_requested", "nvidia_smi", "cuda", "source_commit", "exit",
                                             "wall_s", "estimated_gpu_cost_usd")}, indent=2))
    print("results:", dest / "out")
    return record


@app.local_entrypoint()
def upload(models: str = "mace-mh-1.model", baseline: str = "~/fastmace-baseline/1a-T18", limit_usd: str = "",
           owner_override: str = ""):
    """Copy checkpoints (comma-separated names under ~/models) and the baseline export into the volume."""
    home = Path.home()
    base = Path(os.path.expanduser(baseline))

    def launch():
        with volume.batch_upload(force=True) as batch:
            for name in [m for m in models.split(",") if m]:
                batch.put_file(str(home / "models" / name), f"/models/{name}")
            batch.put_directory(str(base), "/baseline/1a-T18")
        print(volume_listing.remote())

    _guarded(launch, commit="upload", gpu=None, function_timeout_s=LISTING_TIMEOUT_S, cpu=LISTING_CPU,
             memory_mib=LISTING_MEMORY_MIB, limit_usd=limit_usd, owner_override=owner_override)


@app.local_entrypoint()
def listing(limit_usd: str = "", owner_override: str = ""):
    """The volume's top three levels, from a one-core CPU container (seconds; also checks that jobs start)."""
    _guarded(lambda: print(volume_listing.remote()), commit="listing", gpu=None, function_timeout_s=LISTING_TIMEOUT_S,
             cpu=LISTING_CPU, memory_mib=LISTING_MEMORY_MIB, limit_usd=limit_usd, owner_override=owner_override)


@app.local_entrypoint()
def smoke(commit: str = "HEAD", repo: str = "/mnt/d/Margraf/FastMACE", gpu: str = "L4", timeout_min: int = 20,
          out: str = "~/fastmace-acceptance/modal-smoke", limit_usd: str = "", owner_override: str = ""):
    """nvidia-smi, torch's CUDA view and tests/gpu/test_smoke.py on the smallest GPU."""
    command = ("nvidia-smi > $OUT/nvidia-smi.txt 2>&1; "
               "python -c 'import torch, json; print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, "
               "available=torch.cuda.is_available(), device=torch.cuda.get_device_name(), "
               "capability=torch.cuda.get_device_capability())))' > $OUT/torch.json 2>&1; "
               "python -m pytest -p no:cacheprovider -q -rs tests/gpu/test_smoke.py > $OUT/test_smoke.log 2>&1; "
               "tail -3 $OUT/test_smoke.log")
    _job(command, commit, repo, "", gpu, timeout_min, out, limit_usd, owner_override)


@app.local_entrypoint()
def job(command: str, commit: str = "HEAD", repo: str = "/mnt/d/Margraf/FastMACE", tarball: str = "", gpu: str = "L4",
        timeout_min: int = 60, out: str = "~/fastmace-acceptance/modal-job", limit_usd: str = "",
        owner_override: str = ""):
    """Run ``command`` from the root of a pinned source tarball on ``gpu``; results land in ``out``."""
    _job(command, commit, repo, tarball, gpu, timeout_min, out, limit_usd, owner_override)


def _job(command, commit, repo, tarball, gpu, timeout_min, out, limit_usd="", owner_override=""):
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {GPUS}")
    timeout_s = int(timeout_min * 60)
    if timeout_s <= 0:
        raise SystemExit("--timeout-min must be positive")
    if tarball:
        data = Path(os.path.expanduser(tarball)).read_bytes()
        sha = "tarball"
    else:
        sha, data = _archive(commit, os.path.expanduser(repo))
    print(f"source {sha}, {len(data) / 1e6:.1f} MB; gpu {gpu}; timeout {timeout_min} min")
    function = run_job.with_options(gpu=gpu, timeout=timeout_s + GRACE_S, retries=0)

    def launch():
        return _receive(function.remote(gpu, data, command, timeout_s), Path(os.path.expanduser(out)))

    return _guarded(launch, commit=sha, gpu=gpu, function_timeout_s=timeout_s + GRACE_S, cpu=JOB_CPU,
                    memory_mib=JOB_MEMORY_MIB, limit_usd=limit_usd, owner_override=owner_override)
