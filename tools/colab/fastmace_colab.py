"""FastMACE jobs on Google Colab GPUs through the ``colab`` CLI (plan task ops-colab, 23 October 2026).

Run from WSL with the system Python (standard library only) and the ``colab`` CLI, signed in as the owner:

    python3 tools/colab/fastmace_colab.py --gpu G4 --commit <sha> --out ~/fastmace-acceptance/colab-<run> \\
        --command 'python tools/acceptance/compat.py --provider colab --label "Colab G4" --output $OUT'

One VM per call (``--session`` names it, ``--keep`` leaves it running for the next call with the same name).
The driver:

1. reads ``colab usage`` (compute units before);
2. allocates the VM (``colab new``) unless ``--session`` names a live one;
3. uploads one payload: a ``git archive`` of ``--commit`` with ``.source_commit`` (the repository is private,
   so no clone), the baseline export with the structures, the requirements, the model list and the commands;
4. on the VM (``remote.sh``): probes the GPU, installs the pinned hybrid stack into a uv-managed Python 3.10
   venv (about a minute; the venv is kept on a kept session), fetches the released checkpoints from their
   official URLs with the label table's SHA-256, restores the compile caches for this GPU capability and
   stack, runs each ``--command`` from the source root with ``$OUT`` as its results directory, and packs the
   new or changed cache files (the store's next part) and the records;
5. downloads the records to ``<out>/out/`` and the changed caches to the store;
6. stops the VM, also on an error and on Ctrl-C (``--keep`` excepted);
7. writes ``<out>/colab.json`` and a row in the ledger: GPU, capability, driver, versions, commit, setup and
   wall time, compute units before and after.

Why no Google Drive: ``colab drivemount`` asks for the owner's browser consent on every new VM (checked
23 October 2026: a second VM got HTTP 400 without a fresh consent), so it cannot run unattended. The store is
local instead (``FASTMACE_COLAB_STATE``, default ``~/.fastmace-colab``): ``caches/<key>/<nnn>.tar.gz`` with
``key = sm<capability>-py310-<sha256 of the requirements>``, ``manifests/<stack>.json`` (the environment's
``pip freeze`` and its sha256, from ``--build-env``) and ``colab-ledger.csv``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GPUS = ("G4", "A100", "L4", "H100", "T4")
REMOTE = "/content/fm"
# Uploads land in /content (a new VM has no /content/fm yet); remote_step.py moves "fm_<name>" to REMOTE/<name>.
STAGING = "/content/fm_"
PYTHON = "3.10"
REQUIREMENTS = ROOT / "tools" / "modal" / "requirements-hybrid.txt"
# certifi raised to the 22 October 2026 floor (eMudhra/emSign incident), as ~/environments/mace_cueq_oeq.
OVERRIDES = {"certifi": "2026.10.19"}
# simple-dftd3's Python API, the second D3 reference of tests/gpu/test_d3_kernel_gpu.py (16 tests skip without it).
EXTRAS = ("dftd3==1.6.0",)
# The compatibility tier's checkpoints (tests/_support/tiers.py: MH-1, MP-0 medium, POLAR-1-S).
DEFAULT_MODELS = "mh-1,medium,polar-1-s"
DEFAULT_REPO = "/mnt/d/Margraf/FastMACE"  # the main checkout sees every pushed branch
STATE = Path(os.path.expanduser(os.environ.get("FASTMACE_COLAB_STATE", "~/.fastmace-colab")))
WAIT_S = 1200  # one `colab exec` waits at most this long; the driver calls again while a step runs
# Largest single upload: 64 MiB went up and 80 MiB failed with HTTP 400 (CPU VM, 23 October 2026; base64 adds a
# third on the wire).
CHUNK_BYTES = 48 * 2**20
LEDGER_FIELDS = ("date_utc", "session", "gpu", "gpu_name", "capability", "commit", "commands", "exit", "wall_s",
                 "setup_s", "cache_key", "cache_hit", "cu_before", "cu_after", "cu_used", "out")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", choices=GPUS, required=True, help="Colab accelerator (G4 = RTX PRO 6000 Blackwell)")
    p.add_argument("--commit", default="HEAD", help="commit or branch to run, archived from --repo")
    p.add_argument("--command", action="append", default=[],
                   help="shell command, run from the source root with $OUT; repeat to batch several")
    p.add_argument("--build-env", action="store_true",
                   help="set up the environment and record its manifest in the store (no command needed)")
    p.add_argument("--out", required=True, type=Path, help="local directory for colab.json and the records")
    p.add_argument("--session", default=None, help="session name (default fm-<gpu>-<UTC time>); reused if live")
    p.add_argument("--keep", action="store_true", help="leave the VM running afterwards (stop it yourself)")
    p.add_argument("--models", default=DEFAULT_MODELS, help="checkpoint names of fastmace.model.labels")
    p.add_argument("--baseline", default="~/fastmace-baseline/1a-T18", type=Path,
                   help="the research-engine export with the structures (FASTMACE_STRUCTURES on the VM)")
    p.add_argument("--repo", default=DEFAULT_REPO if Path(DEFAULT_REPO).is_dir() else str(ROOT), type=Path)
    p.add_argument("--timeout-min", type=float, default=180, help="per command; the step is killed after it")
    p.add_argument("--store", default=STATE, type=Path, help="caches, manifests and the default ledger")
    p.add_argument("--ledger", default=None, type=Path, help="default <store>/colab-ledger.csv")
    p.add_argument("--dry-run", action="store_true", help="print the colab commands, run nothing")
    args = p.parse_args(argv)
    if not args.command and not args.build_env:
        p.error("give --command (repeatable) or --build-env")
    return args


def default_session(gpu, now=None):
    now = now or datetime.now(timezone.utc)
    return f"fm-{gpu.lower()}-{now:%Y%m%d-%H%M%S}"


def new_args(gpu, session):
    return ["colab", "new", "--gpu", gpu, "-s", session]


def stop_args(session):
    return ["colab", "stop", "-s", session]


def upload_args(session, local, remote):
    return ["colab", "upload", "-s", session, str(local), remote]


def download_args(session, remote, local):
    return ["colab", "download", "-s", session, remote, str(local)]


def exec_args(session, script, timeout_s, env):
    line = ["colab", "exec", "-s", session, "-f", str(script), "--timeout", str(int(timeout_s))]
    for key, value in env.items():
        line += ["--env", f"{key}={value}"]
    return line


def step_env(step, stage, arg="", wait_s=WAIT_S, reset=False, kill=False):
    """Every variable remote_step.py reads, each call: ``--env`` values persist in the remote kernel."""
    return {"STEP": step, "STAGE": stage, "ARG": str(arg), "WAIT_S": str(int(wait_s)), "RESET": "1" if reset else "0",
            "KILL": "1" if kill else "0"}


def parse_usage(text):
    balance = re.search(r"balance:\s*([0-9.]+)", text)
    rate = re.search(r"rate:\s*([0-9.]+)\s*/\s*hr", text)
    return {"balance": float(balance.group(1)) if balance else None,
            "rate_per_h": float(rate.group(1)) if rate else None}


def parse_step(output):
    lines = [line for line in output.splitlines() if line.startswith("@@STEP ")]
    return json.loads(lines[-1][len("@@STEP "):]) if lines else None


def stack_key(requirements):
    return hashlib.sha256(requirements.encode()).hexdigest()[:12]


def cache_key(capability, stack):
    return f"sm{capability.replace('.', '')}-py{PYTHON.replace('.', '')}-{stack}"


def split_file(path, directory, size):
    """``[path]`` if it fits in one upload, else its pieces ``<directory>/<name>.chunk<nnn>``."""
    path = Path(path)
    if path.stat().st_size <= size:
        return [path]
    Path(directory).mkdir(parents=True, exist_ok=True)
    chunks = []
    with open(path, "rb") as f:
        for n, block in enumerate(iter(lambda: f.read(size), b"")):
            chunk = Path(directory) / f"{path.name}.chunk{n:03d}"
            chunk.write_bytes(block)
            chunks.append(chunk)
    return chunks


def cache_parts(directory):
    """The store's parts for one key, in order: the first run's caches, then each later run's new files."""
    return sorted(Path(directory).glob("[0-9][0-9][0-9].tar.gz")) if Path(directory).is_dir() else []


def next_part_name(parts):
    return f"{int(parts[-1].name[:3]) + 1:03d}.tar.gz" if parts else "000.tar.gz"


def requirements_text(text, overrides=OVERRIDES, extras=EXTRAS):
    out = []
    for line in text.splitlines():
        name = re.split(r"[=<>~ @]", line.strip(), maxsplit=1)[0].lower()
        out.append(f"{name}=={overrides[name]}" if name in overrides and not line.startswith("#") else line)
    if extras:
        out += ["# Colab only (tools/colab/fastmace_colab.EXTRAS)", *extras]
    return "\n".join(out) + "\n"


def ledger_row(record):
    row = {field: record.get(field) for field in LEDGER_FIELDS}
    row["commands"] = " ; ".join(record.get("commands_text") or record.get("commands") or [])
    before, after = record.get("cu_before"), record.get("cu_after")
    row["cu_used"] = round(before - after, 2) if before is not None and after is not None else None
    return {k: "" if v is None else v for k, v in row.items()}


def append_ledger(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                csv.writer(f).writerow(LEDGER_FIELDS)
            csv.DictWriter(f, fieldnames=LEDGER_FIELDS).writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
    tar.addfile(info, io.BytesIO(data))


def source_archive(repo, commit):
    """``git archive`` of ``commit`` (LF endings) with ``.source_commit``, as the pinned copies of AGENTS.md."""
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{commit}^{{commit}}"], capture_output=True,
                         text=True, check=True).stdout.strip()
    data = subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", "archive", "--format=tar", sha],
                          capture_output=True, check=True).stdout
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(data)) as source, tarfile.open(fileobj=out, mode="w") as dest:
        for member in source.getmembers():
            dest.addfile(member, source.extractfile(member) if member.isfile() else None)
        _add_bytes(dest, ".source_commit", (sha + "\n").encode())
    return sha, out.getvalue()


def build_payload(work, repo, commit, commands, models, baseline, requirements):
    """One tar for one upload: source.tar.gz, baseline.tar.gz, requirements.txt, models.txt, command<n>.sh.
    The source is gzipped: its 66 MB are mostly JSON records under results/."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sha, source = source_archive(repo, commit)
    base = io.BytesIO()
    with tarfile.open(fileobj=base, mode="w:gz") as tar:
        if Path(baseline).is_dir():
            tar.add(str(baseline), arcname=Path(baseline).name)
    payload = work / "payload.tar"
    with tarfile.open(payload, "w") as tar:
        _add_bytes(tar, "source.tar.gz", gzip.compress(source, compresslevel=6, mtime=0))
        _add_bytes(tar, "baseline.tar.gz", base.getvalue())
        _add_bytes(tar, "requirements.txt", requirements.encode())
        _add_bytes(tar, "models.txt", "".join(f"{m}\n" for m in models).encode())
        for n, command in enumerate(commands, 1):
            _add_bytes(tar, f"command{n}.sh", (command + "\n").encode())
    return sha, payload


def run_colab(argv, timeout=None):
    """The real CLI: stdout on success, CalledProcessError (with the output) otherwise."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        text = re.sub(r"(token=)[^&\s]+", r"\1REDACTED", f"{done.stdout[-3000:]}{done.stderr[-3000:]}")
        sys.stderr.write(f"{shlex.join(argv)} failed ({done.returncode}):\n{text}\n")
        raise subprocess.CalledProcessError(done.returncode, argv, done.stdout, done.stderr)
    return done.stdout


class Job:
    """One driver call against ``colab`` (a callable ``colab(argv, timeout=None) -> stdout``)."""

    def __init__(self, args, colab):
        self.args, self.colab = args, colab
        self.session = args.session or default_session(args.gpu)
        self.out = Path(args.out).expanduser()
        self.store = Path(args.store).expanduser()
        self.work = self.out / ".colab-work"

    def call(self, argv, timeout=None):
        if self.args.dry_run:
            print(shlex.join(argv))
            return ""
        return self.colab(argv, timeout=timeout)

    def usage(self):
        try:
            return parse_usage(self.call(["colab", "usage"], timeout=120))
        except (subprocess.SubprocessError, OSError):
            return parse_usage("")

    def transfer(self, argv, attempts=4):
        """Uploads and downloads, retried: the VM's file API can answer 500 just after `colab new`."""
        for attempt in range(1, attempts + 1):
            try:
                return self.call(argv, timeout=1800)
            except subprocess.CalledProcessError:
                if attempt == attempts:
                    raise
                _log(f"{argv[1]} failed (attempt {attempt}), retrying in {10 * attempt} s")
                time.sleep(10 * attempt)

    def upload(self, local, remote):
        """`colab upload` sends a file as one base64 JSON request, which fails above the Jupyter server's
        request limit (a 513 MB cache part did, 23 October 2026): larger files go up in chunks that
        remote_step.py joins."""
        chunks = split_file(local, self.work / "chunks", CHUNK_BYTES) if not self.args.dry_run else [local]
        if len(chunks) == 1:
            self.transfer(upload_args(self.session, local, remote))
            return
        for chunk in chunks:
            self.transfer(upload_args(self.session, chunk, f"{remote}.{chunk.name.rsplit('.', 1)[1]}"))
            chunk.unlink()

    def download(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.transfer(download_args(self.session, remote, local))

    def step(self, step, stage, arg="", timeout_s=None, reset=False):
        """Run one remote.sh stage to the end; returns (exit, seconds, log tail)."""
        deadline = time.time() + (timeout_s or 3600)
        first = True
        while True:
            wait = max(10, min(WAIT_S, deadline - time.time()))
            env = step_env(step, stage, arg, wait_s=wait, reset=reset and first)
            out = self.call(exec_args(self.session, HERE / "remote_step.py", wait + 120, env), timeout=wait + 300)
            first = False
            if self.args.dry_run:
                return 0, 0, ""
            state = parse_step(out) or {}
            if state.get("state") == "done":
                return state["exit"], state["seconds"], out
            if time.time() >= deadline:
                out = self.call(exec_args(self.session, HERE / "remote_step.py", 180,
                                          step_env(step, stage, arg, wait_s=0, kill=True)), timeout=300)
                return 124, None, out
            print(out.rsplit("@@STEP", 1)[0][-800:], flush=True)

    def live_sessions(self):
        try:
            text = self.call(["colab", "sessions"], timeout=120)
        except (subprocess.SubprocessError, OSError):
            return set()
        return set(re.findall(r"^\[([^\]]+)\]", text, flags=re.M))


def _log(message):
    print(f"[fastmace-colab {datetime.now(timezone.utc):%H:%M:%S}] {message}", flush=True)


def run_job(args, colab=run_colab):
    """The whole call; the VM is stopped in ``finally`` unless ``--keep``."""
    job = Job(args, colab)
    started = time.time()
    raw = REQUIREMENTS.read_text()
    requirements = requirements_text(raw)
    stack = stack_key(requirements)
    models = [m for m in args.models.split(",") if m]
    record = dict(provider="colab", date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  session=job.session, gpu=args.gpu, commit=None, requested_commit=args.commit,
                  commands_text=list(args.command), commands=[], exit="error", stack=stack, models=models,
                  out=str(job.out), timings={})
    timings = record["timings"]
    job.out.mkdir(parents=True, exist_ok=True)
    stopped = False
    try:
        sha, payload = build_payload(job.work, Path(args.repo).expanduser(), args.commit, args.command, models,
                                     Path(args.baseline).expanduser(), requirements)
        record["commit"] = sha
        before = job.usage()
        record["cu_before"], record["rate_per_h"] = before["balance"], before["rate_per_h"]
        t = time.time()
        if job.session in job.live_sessions():
            _log(f"reusing live session {job.session}")
            record["reused_session"] = True
        else:
            _log(f"allocating {args.gpu} as {job.session}")
            job.call(new_args(args.gpu, job.session), timeout=900)
        timings["allocate_s"] = round(time.time() - t, 1)
        t = time.time()
        job.upload(HERE / "remote.sh", f"{STAGING}remote.sh")
        job.upload(payload, f"{STAGING}payload.tar")
        timings["upload_s"] = round(time.time() - t, 1)
        code, _, log = job.step("probe", "probe", timeout_s=300, reset=True)
        probe = _last_json(log) if not args.dry_run else {"capability": "0.0"}
        record.update({k: probe.get(k) for k in ("gpu_name", "driver", "gpu_memory", "capability", "cuda_driver",
                                                 "cpus", "ram_gib")})
        key = cache_key(record["capability"] or "0.0", stack)
        record["cache_key"] = key
        _log(f"{record.get('gpu_name')} (sm {record['capability']}, driver {record.get('driver')}); cache {key}")
        code, seconds, log = job.step("setup", "setup", timeout_s=3600)
        timings["setup_step_s"] = seconds
        if code != 0:
            raise RuntimeError(f"setup failed (exit {code}):\n{log[-3000:]}")
        cache = job.store / "caches" / key
        parts = cache_parts(cache)
        record["cache_hit"] = bool(parts)
        t = time.time()
        for part in parts:
            job.upload(part, f"{STAGING}cache-in-{part.name}")
        record["cache_bytes_in"] = sum(part.stat().st_size for part in parts)
        code, seconds, log = job.step("restore", "restore", timeout_s=900)
        timings["cache_restore_s"] = round(time.time() - t, 1)
        record["setup_s"] = round(time.time() - started, 1)
        _log(f"ready after {record['setup_s']} s; running {len(args.command)} command(s)")
        worst = 0
        for n, command in enumerate(args.command, 1):
            code, seconds, log = job.step(f"run{n}", "run", arg=n, timeout_s=args.timeout_min * 60)
            record["commands"].append(dict(command=command, exit=code, wall_s=seconds))
            _log(f"command {n} exit {code} after {seconds} s")
            print(log.rsplit("@@STEP", 1)[0][-2000:], flush=True)
            worst = worst or code
        t = time.time()
        code, _, log = job.step("pack", "pack", timeout_s=1800)
        changed = "caches changed" in log
        job.download(f"{REMOTE}/out.tar.gz", job.work / "out.tar.gz")
        if changed and not args.dry_run:
            cache.mkdir(parents=True, exist_ok=True)
            part = cache / next_part_name(parts)
            partial = part.with_suffix(".part")
            job.download(f"{REMOTE}/cache-out.tar.gz", partial)
            partial.replace(part)
            record["cache_bytes_out"] = part.stat().st_size
        record["cache_saved"] = changed
        timings["pack_download_s"] = round(time.time() - t, 1)
        if not args.dry_run:
            _receive(job.work / "out.tar.gz", job.out, record)
            _manifest(job, record, stack)
        record["exit"] = worst
    finally:
        if not args.keep:
            stopped = _stop(job)
        record["stopped"] = stopped
        after = job.usage() if not args.dry_run else {}
        record["cu_after"], record["rate_per_h"] = after.get("balance"), after.get("rate_per_h") or record.get(
            "rate_per_h")
        record["wall_s"] = round(time.time() - started, 1)
        if record.get("cu_before") is not None and record.get("cu_after") is not None:
            record["cu_used"] = round(record["cu_before"] - record["cu_after"], 2)
        if not args.dry_run:
            (job.out / "colab.json").write_text(json.dumps(record, indent=2) + "\n")
            ledger = args.ledger or job.store / "colab-ledger.csv"
            append_ledger(ledger, ledger_row(record))
            _log(f"exit {record['exit']}, wall {record['wall_s']} s, CU {record.get('cu_before')} -> "
                 f"{record.get('cu_after')}; {job.out / 'colab.json'}; ledger {ledger}")
    return record


def _stop(job):
    """``colab stop``, retried; a second Ctrl-C must not abort it."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for attempt in range(3):
            try:
                job.call(stop_args(job.session), timeout=300)
                _log(f"stopped {job.session}")
                return True
            except (subprocess.SubprocessError, OSError) as error:
                _log(f"stop attempt {attempt + 1} failed: {error}")
                time.sleep(5)
        _log(f"WARNING: {job.session} may still be running; run: colab stop -s {job.session}")
        return False
    finally:
        signal.signal(signal.SIGINT, previous)


def _last_json(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(line)
    return {}


def _receive(archive, dest, record):
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data") if sys.version_info >= (3, 12) else tar.extractall(dest)
    setup = dest / "out" / "setup.json"
    if setup.exists():
        data = json.loads(setup.read_text())
        record["environment"] = data
        record["timings"].update({k: data.get(k) for k in ("unpack_s", "install_s", "models_s")})


def _manifest(job, record, stack):
    """The environment's package list: written by --build-env (or the first run), compared on every run."""
    freeze = job.out / "out" / "pip-freeze.txt"
    digest = (record.get("environment") or {}).get("pip_freeze_sha256")
    if not freeze.exists() or not digest:
        return
    path = job.store / "manifests" / f"{stack}.json"
    if job.args.build_env or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(stack=stack, python=record["environment"].get("python"),
                                        pip_freeze_sha256=digest, built=record["date_utc"], session=job.session,
                                        gpu=job.args.gpu, pip_freeze=freeze.read_text().splitlines()), indent=2) + "\n")
    record["env_matches_manifest"] = json.loads(path.read_text())["pip_freeze_sha256"] == digest


def main(argv=None):
    args = parse_args(argv)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    record = run_job(args)
    code = record["exit"]
    return code if isinstance(code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
