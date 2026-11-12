"""Manifest-driven launcher for packed Modal jobs (AGENTS.md "Using resources"; resource framework §3.4-3.5).

A batch manifest is a CSV with the columns ``idea,switch,gpu,profile,timeout`` and the optional ``kind``
(``ab`` default, ``gate``, ``profile``) and ``name`` (the switch's short name; default: the idea id). ``switch`` is
side b's ``MODULE:NAME=VALUE`` words (``ab`` also takes ab_rewrites words starting with ``--``); ``timeout`` is in
minutes and may be empty (then twice the template's p90, ``p90.json``). Lines starting with ``#`` are comments.

Rows with the same kind, GPU and profile go into one container: at most ``MAX_PACK`` switches per ``ab_pack`` or
``gate_pack`` job, one row per ``profile`` job. A pack with a single switch is refused unless ``--allow-single``
(the switch blocks a decision).

Commands (run them in WSL, where the Modal client lives):

    python3 tools/jobs/launch.py plan  MANIFEST --batch ID              # validate and show the packs
    python3 tools/jobs/launch.py start MANIFEST --batch ID --commit SHA  # launch as a systemd system unit
    python3 tools/jobs/launch.py status ~/fm-runs/ID

``start`` makes the batch folder ``~/fm-runs/ID`` (refused if it exists), copies the templates, this launcher, the
Modal app and the manifest into it, builds the slim tarball of ``SHA`` there, drops stale in-flight budget entries,
starts the Windows keep-alive and runs ``launch.py run`` from the batch folder's copy as a ``systemd-run`` system unit,
so later edits never touch the running batch. The run starts at most ``MAX_CLIENTS`` Modal clients at once (counting
every ``fastmace_modal.py`` client on the machine), ``LAUNCH_GAP_S`` apart, retries a failed job once and leaves a
second failure for a person. A budget refusal or Modal's rate limit waits and relaunches without using the retry.
Each client gets ``MODAL_PROFILE``, ``FASTMACE_BATCH`` and ``FASTMACE_IDEA``, which the ledger records.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 1
MAX_CLIENTS = 6
LAUNCH_GAP_S = 25
MAX_PACK = 8
MAX_ATTEMPTS = 2  # one retry
MAX_WAITS = 40  # budget refusals and rate limits, WAIT_S apart
WAIT_S = 90
GPUS = ("L4", "A100", "A100-80GB", "H100", "H200")
GPU_FAMILY = {"L4": "L4", "A100": "A100", "A100-80GB": "A100", "H100": "H100", "H200": "H200"}
# The owner (6 November 2026): any A100 80GB landing (SXM4 or PCIe) is fine.
EXPECT_GPU = {"L4": "L4", "A100": "A100", "A100-80GB": "A100", "H100": "H100", "H200": "H200"}
KINDS = {"ab": "ab_pack.sh", "gate": "gate_pack.sh", "profile": "profile.sh"}
REQUIRED = ("idea", "switch", "gpu", "profile", "timeout")
OPTIONAL = ("kind", "name")
TEMPLATE_FILES = ("ab_pack.sh", "gate_pack.sh", "profile.sh", "launch.py", "slim_tar.py", "p90.json")
MODAL_FILES = ("fastmace_modal.py", "budget.py", "requirements-hybrid.txt")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
CONSTANT_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*=\S+$")
ENV_RE = re.compile(r"^(MODEL|CHECKPOINT|STRUCTURE|STRUCTURES|CELL|BLOCKS|ALIAS|BACKEND|STEPS)=(\S*)$")
CLIENT_MARK = "fastmace_modal.py::"
REFUSAL_RE = re.compile(r"refused: Modal budget|rate limit exceeded", re.IGNORECASE)


class ManifestError(ValueError):
    """A manifest the launcher will not run."""


@dataclass(frozen=True)
class Row:
    line: int
    idea: str
    switch: str
    gpu: str
    profile: str
    timeout_min: int | None
    kind: str = "ab"
    name: str = ""


@dataclass
class Job:
    id: str
    kind: str
    gpu: str
    profile: str
    timeout_min: int
    entries: list  # [name, switch] pairs
    ideas: list
    env: dict = field(default_factory=dict)
    attempts: int = 0
    waits: int = 0
    state: str = "queued"  # queued, running, done, failed
    history: list = field(default_factory=list)

    def pack(self) -> str:
        return ";".join(f"{name}|{switch}" for name, switch in self.entries)

    def command(self) -> str:
        """The container command: the template with its environment, every value shell-quoted."""
        env = dict(self.env, EXPECT_GPU=EXPECT_GPU[self.gpu], PACK=self.pack())
        words = [f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())]
        return " ".join(words + ["bash", f"tools/jobs/{KINDS[self.kind]}"])


# --- manifest -----------------------------------------------------------------------------------------------------

def _check_switch(switch, kind, line):
    if not switch.strip():
        if kind == "profile":
            return ""
        raise ManifestError(f"line {line}: empty switch")
    if any(c in switch for c in ";|\n\r'\"`$\\"):
        raise ManifestError(f"line {line}: switch {switch!r} has one of ; | ' \" ` $ \\ (separators or shell)")
    words = switch.split()
    if kind in ("ab", "gate") and words[0].startswith("--"):
        return " ".join(words)
    bad = [w for w in words if not CONSTANT_RE.match(w)]
    if bad:
        raise ManifestError(f"line {line}: {bad} is not MODULE:NAME=VALUE")
    return " ".join(words)


def parse_manifest(text: str) -> list[Row]:
    """The rows of a manifest; ``ManifestError`` names the first bad line."""
    lines = [(n, raw) for n, raw in enumerate(text.splitlines(), 1) if raw.strip() and not raw.lstrip().startswith("#")]
    if not lines:
        raise ManifestError("the manifest is empty")
    reader = csv.reader(io.StringIO("\n".join(raw for _, raw in lines)))
    header = [h.strip().lower() for h in next(reader)]
    missing = [c for c in REQUIRED if c not in header]
    unknown = [c for c in header if c not in REQUIRED + OPTIONAL]
    if missing or unknown or len(set(header)) != len(header):
        raise ManifestError(f"header {header}: needs {list(REQUIRED)}, may add {list(OPTIONAL)} "
                            f"(missing {missing}, unknown {unknown})")
    rows, seen = [], set()
    for (line, _), values in zip(lines[1:], reader):
        if len(values) != len(header):
            raise ManifestError(f"line {line}: {len(values)} fields, the header has {len(header)}")
        rec = {k: v.strip() for k, v in zip(header, values)}
        kind = rec.get("kind") or "ab"
        if kind not in KINDS:
            raise ManifestError(f"line {line}: kind {kind!r} is not one of {sorted(KINDS)}")
        if not ID_RE.match(rec["idea"]):
            raise ManifestError(f"line {line}: idea {rec['idea']!r} must match {ID_RE.pattern}")
        name = rec.get("name") or rec["idea"]
        if not ID_RE.match(name):
            raise ManifestError(f"line {line}: name {name!r} must match {ID_RE.pattern}")
        if rec["gpu"] not in GPUS:
            raise ManifestError(f"line {line}: gpu {rec['gpu']!r} is not one of {GPUS}")
        if not PROFILE_RE.match(rec["profile"]):
            raise ManifestError(f"line {line}: profile {rec['profile']!r} must match {PROFILE_RE.pattern}")
        timeout = None
        if rec["timeout"]:
            if not rec["timeout"].isdigit() or int(rec["timeout"]) <= 0:
                raise ManifestError(f"line {line}: timeout {rec['timeout']!r} must be whole minutes > 0 or empty")
            timeout = int(rec["timeout"])
        switch = _check_switch(rec["switch"], kind, line)
        key = (kind, rec["gpu"], rec["profile"], switch)
        if key in seen:
            raise ManifestError(f"line {line}: the same {kind} switch on {rec['gpu']}/{rec['profile']} twice")
        seen.add(key)
        rows.append(Row(line, rec["idea"], switch, rec["gpu"], rec["profile"], timeout, kind, name))
    if not rows:
        raise ManifestError("the manifest has a header but no rows")
    return rows


# --- packing and timeouts -----------------------------------------------------------------------------------------

def load_p90(path=HERE / "p90.json"):
    return json.loads(Path(path).read_text())


def p90_s(p90, kind, gpu, switches):
    fit = p90[kind][GPU_FAMILY[gpu]]
    units = 1 if kind == "profile" else switches + (1 if kind == "gate" else 0)
    return fit["base_s"] + fit["per_unit_s"] * units


def timeout_min(p90, kind, gpu, switches, explicit=()):
    """Twice the template's p90 wall time, in whole minutes; an explicit manifest timeout wins (the largest)."""
    explicit = [t for t in explicit if t]
    if explicit:
        return max(explicit)
    return math.ceil(2 * p90_s(p90, kind, gpu, switches) / 60)


def plan(rows, batch, p90, env=None, allow_single=False, max_pack=MAX_PACK):
    """Pack ``rows`` into jobs: same kind, GPU and profile share a container, ``max_pack`` switches at most."""
    if not ID_RE.match(batch):
        raise ManifestError(f"batch {batch!r} must match {ID_RE.pattern}")
    groups: dict = {}
    for row in rows:
        groups.setdefault((row.kind, row.gpu, row.profile), []).append(row)
    jobs = []
    for (kind, gpu, profile), members in groups.items():
        size = 1 if kind == "profile" else max_pack
        chunks = [members[i:i + size] for i in range(0, len(members), size)]
        for n, chunk in enumerate(chunks, 1):
            if kind != "profile" and len(chunk) == 1 and not allow_single:
                raise ManifestError(f"line {chunk[0].line}: a {kind} pack with one switch ({chunk[0].idea} on "
                                    f"{gpu}/{profile}); pack 4-8 switches, or --allow-single if it blocks a decision")
            names, entries = set(), []
            for row in chunk:
                name, k = row.name, 2
                while name in names:
                    name, k = f"{row.name}-{k}", k + 1
                names.add(name)
                entries.append([name, row.switch])
            job_id = f"{batch}-{kind}-{GPU_FAMILY[gpu].lower()}-{profile}-{n}"
            jobs.append(Job(job_id, kind, gpu, profile,
                            timeout_min(p90, kind, gpu, len(chunk), [r.timeout_min for r in chunk]),
                            entries, sorted({r.idea for r in chunk}), dict(env or {})))
    return jobs


def parse_env(pairs):
    env = {}
    for pair in pairs or ():
        match = ENV_RE.match(pair)
        if not match:
            raise ManifestError(f"--env {pair!r}: only {ENV_RE.pattern}")
        env[match.group(1)] = match.group(2)
    return env


# --- running ------------------------------------------------------------------------------------------------------

def count_clients(proc=Path("/proc")):
    """Modal clients of fastmace_modal.py running on this machine (any batch, any session)."""
    count = 0
    for entry in proc.glob("[0-9]*"):
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if entry.name != str(os.getpid()) and any(CLIENT_MARK.encode() in a for a in args):
            count += 1
    return count


class Runner:
    """Runs the jobs of a batch: ``max_clients`` Modal clients at most, ``gap_s`` between launches, one retry.

    ``launch(job, attempt) -> handle``, ``poll(handle) -> None | outcome`` with outcome one of "ok", "failed",
    "refused"; ``clients()`` counts the Modal clients on the machine (this batch's included); ``clock``/``sleep``
    are the time functions. All are injectable for the CPU tests."""

    def __init__(self, jobs, *, launch, poll, clients, clock=time.time, sleep=time.sleep, log=print,
                 max_clients=MAX_CLIENTS, gap_s=LAUNCH_GAP_S, wait_s=WAIT_S, max_waits=MAX_WAITS,
                 max_attempts=MAX_ATTEMPTS, tick_s=5.0, on_change=None):
        self.jobs, self.launch, self.poll, self.clients = jobs, launch, poll, clients
        self.clock, self.sleep, self.log = clock, sleep, log
        self.max_clients, self.gap_s, self.wait_s = max_clients, gap_s, wait_s
        self.max_waits, self.max_attempts, self.tick_s = max_waits, max_attempts, tick_s
        self.on_change = on_change or (lambda: None)
        self.running: dict = {}
        self.not_before = {job.id: 0.0 for job in jobs}
        self.last_launch = -math.inf

    def _event(self, job, what):
        job.history.append([round(self.clock(), 1), what])
        self.log(f"{job.id}: {what}")
        self.on_change()

    def step(self):
        """One scheduling pass; returns True while work remains."""
        for job_id, (job, handle) in list(self.running.items()):
            outcome = self.poll(handle)
            if outcome is None:
                continue
            del self.running[job_id]
            if outcome == "ok":
                job.state = "done"
                self._event(job, f"done (attempt {job.attempts})")
            elif outcome == "refused" and job.waits < self.max_waits:
                job.attempts -= 1  # a refusal or rate limit does not use the retry
                job.waits += 1
                job.state = "queued"
                self.not_before[job.id] = self.clock() + self.wait_s
                self._event(job, f"refused or rate-limited, wait {self.wait_s} s ({job.waits}/{self.max_waits})")
            elif job.attempts < self.max_attempts:
                job.state = "queued"
                self._event(job, f"{outcome} (attempt {job.attempts}), retry once")
            else:
                job.state = "failed"
                self._event(job, f"{outcome} (attempt {job.attempts}): stopped, a person looks at it")
        now = self.clock()
        for job in self.jobs:
            if job.state != "queued" or now < self.not_before[job.id]:
                continue
            if now - self.last_launch < self.gap_s or self.clients() >= self.max_clients:
                break
            job.attempts += 1
            job.state = "running"
            self.running[job.id] = (job, self.launch(job, job.attempts))
            self.last_launch = now
            self._event(job, f"launched attempt {job.attempts}")
            now = self.clock()
        return any(job.state in ("queued", "running") for job in self.jobs)

    def run(self):
        while self.step():
            self.sleep(self.tick_s)
        return [job for job in self.jobs if job.state == "failed"]


# --- the batch folder ---------------------------------------------------------------------------------------------

def save_plan(batch_dir, jobs, meta):
    path = Path(batch_dir) / "plan.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(meta, jobs=[asdict(j) for j in jobs]), indent=1) + "\n")
    os.replace(tmp, path)


def load_plan(path):
    data = json.loads(Path(path).read_text())
    jobs = [Job(**j) for j in data.pop("jobs")]
    return data, jobs


def copy_templates(batch_dir, repo=REPO, commit=None):
    """Copy the job templates, this launcher and the Modal app into the batch folder (never edited afterwards):
    from ``commit`` in the git repository ``repo`` (the tarball's own versions), or from ``repo``'s files."""
    batch_dir = Path(batch_dir)
    for sub, names in (("tools/jobs", TEMPLATE_FILES), ("tools/modal", MODAL_FILES)):
        dest = batch_dir / sub
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            if commit is None:
                shutil.copy2(Path(repo) / sub / name, dest / name)
                continue
            data = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{sub}/{name}"], capture_output=True,
                                  check=True).stdout
            (dest / name).write_bytes(data)
            if name.endswith((".sh", ".py")):
                (dest / name).chmod(0o755)
    return batch_dir / "tools"


def make_batch_dir(root, batch):
    """A new, empty batch folder ``root/batch``; refused if it exists or would leave ``root``."""
    if not ID_RE.match(batch):
        raise ManifestError(f"batch {batch!r} must match {ID_RE.pattern}")
    root = Path(root).expanduser().resolve()
    batch_dir = (root / batch).resolve()
    if batch_dir.parent != root:
        raise ManifestError(f"batch folder {batch_dir} is not directly under {root}")
    if batch_dir.exists():
        raise ManifestError(f"{batch_dir} exists; a batch id is used once (running scripts stay unchanged)")
    batch_dir.mkdir(parents=True)
    return batch_dir


def unit_name(batch):
    return "fm-batch-" + re.sub(r"[^A-Za-z0-9_-]", "_", batch)


def unit_command(batch, batch_dir, python="python3"):
    """``systemd-run`` as a system unit for this user: it survives the end of the wsl session that started it."""
    home, user = os.path.expanduser("~"), os.environ.get("USER", "")
    return ["sudo", "-n", "systemd-run", f"--unit={unit_name(batch)}", "--collect", f"--uid={os.getuid()}",
            f"--gid={os.getgid()}", "-p", f"WorkingDirectory={home}", f"--setenv=HOME={home}",
            f"--setenv=USER={user}", f"--setenv=PATH={os.environ.get('PATH', '')}",
            python, str(Path(batch_dir) / "tools" / "jobs" / "launch.py"), "run", str(Path(batch_dir) / "plan.json")]


def keepalive_seconds(jobs, max_clients=MAX_CLIENTS):
    """A generous bound on the batch's duration: waves of ``max_clients``, each job's timeout plus start, twice."""
    longest = max(job.timeout_min for job in jobs) * 60 + 900
    waves = math.ceil(len(jobs) / max_clients)
    return min(2 * waves * longest + 1800, 24 * 3600)


def start_keepalive(seconds, distro=None):
    """Keep the WSL VM up from the Windows side (it powers off ~17 s after the last wsl.exe session ends)."""
    distro = distro or os.environ.get("WSL_DISTRO_NAME", "Ubuntu")
    if not re.match(r"^[A-Za-z0-9._-]+$", distro):
        raise ManifestError(f"odd WSL distro name {distro!r}")
    ps = shutil.which("powershell.exe")
    if not ps:
        print("warning: powershell.exe not found; no Windows keep-alive (is this WSL?)")
        return False
    subprocess.run([ps, "-NoProfile", "-Command",
                    f"Start-Process wsl.exe -ArgumentList '-d','{distro}','--','sleep','{int(seconds)}' "
                    "-WindowStyle Hidden"], check=True, timeout=60)
    return True


def prune_in_flight(modal_dir, state=None):
    """Drop stale in-flight budget entries (dead clients, expired) of every profile, with the guard's own lock."""
    sys.path.insert(0, str(modal_dir))
    import budget  # noqa: E402  (pure Python)
    state = Path(os.path.expanduser(state or os.environ.get("FASTMACE_MODAL_STATE", "~/.fastmace-modal")))
    for path in sorted(state.glob("modal-in-flight*.json")):
        for key, entry in budget.prune(path).items():
            print(f"{path.name}: dropped stale {key[:8]} (pid {entry.get('pid')}, ${entry.get('worst_case_usd')})")


def _modal_launch(meta, batch_dir):
    modal = os.path.expanduser(meta["modal"])
    app = str(Path(batch_dir) / "tools" / "modal" / "fastmace_modal.py")

    def launch(job, attempt):
        tag = f"{job.id}-{'abcdefgh'[attempt - 1]}"
        out = Path(batch_dir) / tag
        env = dict(os.environ, MODAL_PROFILE=job.profile, FASTMACE_BATCH=meta["batch"],
                   FASTMACE_IDEA="+".join(job.ideas), MODAL_IMAGE_BUILDER_VERSION="2025.06")
        stream = open(Path(batch_dir) / f"{tag}.launch.log", "a")
        proc = subprocess.Popen([modal, "run", f"{app}::job", "--gpu", job.gpu, "--timeout-min", str(job.timeout_min),
                                 "--tarball", meta["tarball"], "--command", job.command(), "--out", str(out)],
                                env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=batch_dir)
        return proc, stream, out, Path(stream.name)

    def poll(handle):
        proc, stream, out, log = handle
        code = proc.poll()
        if code is None:
            return None
        stream.close()
        text = log.read_text(errors="replace")
        if REFUSAL_RE.search(text) and not (out / "out" / "modal.json").exists():
            return "refused"
        try:
            record = json.loads((out / "out" / "modal.json").read_text())
        except (OSError, ValueError):
            return "failed"
        return "ok" if code == 0 and record.get("exit") == 0 else "failed"

    return launch, poll


def run_batch(plan_path):
    plan_path = Path(plan_path).resolve()
    batch_dir = plan_path.parent
    meta, jobs = load_plan(plan_path)
    prune_in_flight(batch_dir / "tools" / "modal")
    launch, poll = _modal_launch(meta, batch_dir)
    log_file = open(batch_dir / "batch.log", "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    runner = Runner(jobs, launch=launch, poll=poll, clients=count_clients, log=log,
                    on_change=lambda: save_plan(batch_dir, jobs, meta))
    log(f"batch {meta['batch']}: {len(jobs)} jobs, at most {MAX_CLIENTS} clients, {LAUNCH_GAP_S} s apart")
    failed = runner.run()
    log(f"batch {meta['batch']} finished: {sum(j.state == 'done' for j in jobs)} done, {len(failed)} failed"
        + (f" ({', '.join(j.id for j in failed)}: look at them, no more automatic retries)" if failed else ""))
    return 1 if failed else 0


def describe(jobs):
    lines = []
    for job in jobs:
        lines.append(f"{job.id}: {job.kind} on {job.gpu}/{job.profile}, {len(job.entries)} switch(es), "
                     f"timeout {job.timeout_min} min, ideas {','.join(job.ideas)}")
        lines.append(f"    {job.command()}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "start"):
        p = sub.add_parser(name)
        p.add_argument("manifest", type=Path)
        p.add_argument("--batch", required=True)
        p.add_argument("--env", action="append", help="template variable for every job, e.g. CELL=16x16x16")
        p.add_argument("--allow-single", action="store_true", help="a one-switch pack that blocks a decision")
        p.add_argument("--p90", type=Path, default=HERE / "p90.json")
        if name == "start":
            p.add_argument("--commit", required=True, help="the source commit (must contain tools/jobs)")
            p.add_argument("--repo", default=str(REPO), help="git repository holding the commit (default: this one)")
            p.add_argument("--root", default="~/fm-runs")
            p.add_argument("--modal", default="~/environments/modal_client/bin/modal")
            p.add_argument("--no-unit", action="store_true", help="run in the foreground (tests, debugging)")
            p.add_argument("--no-keepalive", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("plan", type=Path)
    p = sub.add_parser("status")
    p.add_argument("batch_dir", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return run_batch(args.plan)
    if args.cmd == "status":
        meta, jobs = load_plan(Path(args.batch_dir).expanduser() / "plan.json")
        for job in jobs:
            print(f"{job.id}: {job.state}, attempts {job.attempts}, waits {job.waits}; "
                  f"{job.history[-1][1] if job.history else ''}")
        return 0

    try:
        rows = parse_manifest(args.manifest.read_text())
        jobs = plan(rows, args.batch, load_p90(args.p90), parse_env(args.env), args.allow_single)
    except ManifestError as error:
        raise SystemExit(f"manifest: {error}") from None
    print(describe(jobs))
    if args.cmd == "plan":
        return 0

    import slim_tar  # noqa: E402  (next to this file)
    repo = Path(args.repo).expanduser()
    sha = slim_tar.resolve(args.commit, repo)
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}:tools/jobs/ab_pack.sh"]).returncode:
        raise SystemExit(f"commit {sha[:8]} has no tools/jobs templates; launch from a commit that has them")
    batch_dir = make_batch_dir(args.root, args.batch)
    copy_templates(batch_dir, repo, sha)  # the commit's own versions, before anything launches
    shutil.copy2(args.manifest, batch_dir / "manifest.csv")
    tarball = batch_dir / f"src-{sha[:8]}.tar"
    _, kept, dropped = slim_tar.slim_tar(sha, tarball, repo)
    print(f"slim tarball {tarball}: {kept / 1e6:.1f} MB kept, {dropped / 1e6:.1f} MB dropped")
    meta = dict(version=VERSION, batch=args.batch, commit=sha, tarball=str(tarball), modal=args.modal,
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save_plan(batch_dir, jobs, meta)
    if not args.no_keepalive:
        start_keepalive(keepalive_seconds(jobs))
    if args.no_unit:
        return run_batch(batch_dir / "plan.json")
    subprocess.run(["sudo", "-n", "systemctl", "reset-failed", unit_name(args.batch)], capture_output=True)
    subprocess.run(unit_command(args.batch, batch_dir), check=True)
    print(f"started unit {unit_name(args.batch)}; follow {batch_dir / 'batch.log'} or "
          f"python3 {batch_dir / 'tools/jobs/launch.py'} status {batch_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
