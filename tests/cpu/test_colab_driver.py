"""tools/colab/fastmace_colab.py (plan task ops-colab): command building, parsing, ledger and the
always-stop guarantee, with the colab CLI faked (no network, no VM)."""

import csv
import importlib.util
import io
import json
import re
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = REPOSITORY_ROOT / "tools" / "colab" / "fastmace_colab.py"
    spec = importlib.util.spec_from_file_location("_tool_colab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fc = _load()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """The driver's retry back-off sleeps (transfers, polls) cost nothing here."""
    monkeypatch.setattr(fc.time, "sleep", lambda seconds: None)

USAGE = """\
Current balance: 198.99 compute units
Usage rate: 5.30/hr
Active assignments: 1
"""


def test_arguments_batch_several_commands_and_default_session():
    args = fc.parse_args(["--gpu", "G4", "--command", "pytest -m compat", "--command", "echo hi", "--out", "/x"])
    assert args.gpu == "G4" and args.command == ["pytest -m compat", "echo hi"]
    assert args.commit == "HEAD" and not args.keep and args.session is None
    with pytest.raises(SystemExit):
        fc.parse_args(["--gpu", "V100", "--command", "x", "--out", "/x"])
    name = fc.default_session("G4", datetime(2026, 10, 23, 20, 5, 1, tzinfo=timezone.utc))
    assert re.fullmatch(r"fm-g4-20261023-200501-[0-9a-f]{6}", name)
    # two jobs started in the same second must not share a name (4-T13 lost a VM that way)
    assert len({fc.default_session("A100", datetime(2026, 10, 23, tzinfo=timezone.utc)) for _ in range(50)}) == 50


def test_build_env_needs_no_command():
    args = fc.parse_args(["--gpu", "A100", "--build-env", "--out", "/x"])
    assert args.build_env and args.command == []
    with pytest.raises(SystemExit):
        fc.parse_args(["--gpu", "A100", "--out", "/x"])  # neither a command nor --build-env


def test_colab_command_lines():
    assert fc.new_args("G4", "s1") == ["colab", "new", "--gpu", "G4", "-s", "s1"]
    assert fc.stop_args("s1") == ["colab", "stop", "-s", "s1"]
    assert fc.upload_args("s1", "/l/a", "/content/fm/a") == ["colab", "upload", "-s", "s1", "/l/a", "/content/fm/a"]
    assert fc.download_args("s1", "/content/fm/b", "/l/b") == ["colab", "download", "-s", "s1", "/content/fm/b",
                                                               "/l/b"]
    line = fc.exec_args("s1", "/l/remote_step.py", 1260, {"STEP": "run1", "STAGE": "run", "ARG": "1"})
    assert line[:6] == ["colab", "exec", "-s", "s1", "-f", "/l/remote_step.py"]
    assert line[line.index("--timeout") + 1] == "1260"
    assert [line[i + 1] for i, x in enumerate(line) if x == "--env"] == ["STEP=run1", "STAGE=run", "ARG=1"]


def test_step_environment_sets_every_variable_each_call():
    # --env values persist in the remote kernel, so a later call must overwrite RESET, KILL and ARG.
    env = fc.step_env("probe", "probe", wait_s=300)
    assert env == {"STEP": "probe", "STAGE": "probe", "ARG": "", "WAIT_S": "300", "RESET": "0", "KILL": "0"}
    assert fc.step_env("probe", "probe", wait_s=300, reset=True)["RESET"] == "1"
    assert fc.step_env("run2", "run", arg="2", wait_s=60, kill=True)["KILL"] == "1"


def test_usage_and_step_parsing():
    assert fc.parse_usage(USAGE) == {"balance": 198.99, "rate_per_h": 5.30, "active": 1}
    assert fc.parse_usage("nonsense") == {"balance": None, "rate_per_h": None, "active": None}
    out = "log line\n@@STEP {\"step\": \"a\", \"state\": \"running\"}\nmore\n@@STEP " \
          "{\"step\": \"a\", \"state\": \"done\", \"exit\": 3, \"seconds\": 7}\n"
    assert fc.parse_step(out) == {"step": "a", "state": "done", "exit": 3, "seconds": 7}
    assert fc.parse_step("no marker") is None


def test_cache_key_follows_capability_and_stack():
    a = fc.stack_key("torch==2.14.1+cu130\n")
    assert a == fc.stack_key("torch==2.14.1+cu130\n") and a != fc.stack_key("torch==2.14.2\n") and len(a) == 12
    assert fc.cache_key("12.0", a) == f"sm120-py310-{a}"
    assert fc.cache_key("8.9", a) == f"sm89-py310-{a}"


def test_requirements_take_the_overrides():
    text = "# comment\nase==3.27.0\ncertifi==2026.7.22\n"
    out = fc.requirements_text(text, {"certifi": "2026.10.19"}, ())
    assert "certifi==2026.10.19" in out and "certifi==2026.7.22" not in out and "ase==3.27.0" in out
    assert fc.requirements_text(text, {}, ("dftd3==1.6.0",)).splitlines()[-1] == "dftd3==1.6.0"


def test_ledger_header_once_and_rows(tmp_path):
    path = tmp_path / "results" / "colab-ledger.csv"
    record = dict(date_utc="2026-10-23T20:00:00Z", session="s1", gpu="G4", gpu_name="RTX PRO 6000",
                  capability="12.0", commit="abc", commands=["pytest -m compat", "echo, \"hi\""], exit=0,
                  wall_s=600.5, setup_s=70.2, cache_key="sm120-x", cache_hit=True, cu_before=198.0,
                  cu_after=196.5, out="/o", extra="ignored")
    fc.append_ledger(path, fc.ledger_row(record))
    fc.append_ledger(path, fc.ledger_row(dict(record, session="s2", cu_after=None)))
    rows = list(csv.DictReader(path.open()))
    assert [r["session"] for r in rows] == ["s1", "s2"]
    assert list(rows[0]) == list(fc.LEDGER_FIELDS)
    assert rows[0]["cu_used"] == "1.5" and rows[1]["cu_used"] == ""
    assert rows[0]["commands"] == "pytest -m compat ; echo, \"hi\""


@pytest.fixture(autouse=True)
def _git_independent_of_the_machine(monkeypatch):
    """Every git process of these tests (the temporary repos and the driver's own ``rev-parse`` and ``archive``) reads
    no global or system configuration and gets an identity and ``safe.directory=*`` from the environment (command-line
    scope, which git trusts): on murphy a temporary repo on the NAS (/mnt/homes) failed git's ownership check (exit
    128, "dubious ownership"), and there is no global identity there."""
    for key, value in {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_COUNT": "3",
                       "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*",
                       "GIT_CONFIG_KEY_1": "user.email", "GIT_CONFIG_VALUE_1": "t@t",
                       "GIT_CONFIG_KEY_2": "user.name", "GIT_CONFIG_VALUE_2": "t"}.items():
        monkeypatch.setenv(key, value)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")  # identity and safe.directory: ``_git_independent_of_the_machine``
    (repo / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-q", "-m", "one")
    return repo


def test_payload_holds_source_commands_models_and_baseline(tmp_path):
    repo = _git_repo(tmp_path)
    baseline = tmp_path / "1a-T18"
    (baseline / "Structures").mkdir(parents=True)
    (baseline / "Structures" / "NaCl.cif").write_text("cif\n")
    sha, payload = fc.build_payload(tmp_path / "work", repo, "HEAD", ["echo one", "echo two"], ["mh-1", "medium"],
                                    baseline, "ase==3.27.0\n")
    assert len(sha) == 40
    with tarfile.open(payload) as tar:
        names = set(tar.getnames())
        assert {"source.tar.gz", "baseline.tar.gz", "requirements.txt", "models.txt", "command1.sh",
                "command2.sh"} <= names
        assert tar.extractfile("command2.sh").read().decode() == "echo two\n"
        assert tar.extractfile("models.txt").read().decode() == "mh-1\nmedium\n"
        source = tarfile.open(fileobj=io.BytesIO(tar.extractfile("source.tar.gz").read()))
        assert source.extractfile(".source_commit").read().decode().strip() == sha
        assert source.extractfile("a.txt").read() == b"one\n"
        base = tarfile.open(fileobj=io.BytesIO(tar.extractfile("baseline.tar.gz").read()))
        assert "1a-T18/Structures/NaCl.cif" in base.getnames()


class FakeColab:
    """The colab CLI as the driver sees it: canned results per subcommand, every call recorded."""

    def __init__(self, fail_on=None, interrupt_on=None, capability="12.0", cache_changed=True, live=(), parts=()):
        self.calls, self.fail_on, self.interrupt_on, self.live, self.parts = [], fail_on, interrupt_on, live, parts
        self.capability, self.cache_changed = capability, cache_changed
        self.balance = 198.99

    def __call__(self, argv, timeout=None):
        self.calls.append(argv)
        sub = argv[1]
        stage = next((a.split("=", 1)[1] for a in argv if a.startswith("STAGE=")), None)
        if stage and stage == self.interrupt_on:
            raise KeyboardInterrupt
        if stage and stage == self.fail_on:
            raise subprocess.CalledProcessError(1, argv, "boom")
        if sub == "sessions":
            return "".join(f"[{name}] gpu-x | Hardware: G4\n" for name in self.live)
        if sub == "usage":
            text, self.balance = USAGE.replace("198.99", f"{self.balance:.2f}"), self.balance - 0.5
            return text
        if sub == "download":
            self._download(argv[-2], Path(argv[-1]))
            return ""
        if sub == "exec":
            step = next(a.split("=", 1)[1] for a in argv if a.startswith("STEP="))
            log = ""
            if stage == "probe":
                log = json.dumps(dict(gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition", driver="580.82.07",
                                      capability=self.capability, cuda_driver="13.0", cpus=48, ram_gib=176.0,
                                      cache_parts=list(self.parts)))
            if stage == "pack":
                log = "caches changed: 10M" if self.cache_changed else "caches unchanged"
            return log + "\n@@STEP " + json.dumps(dict(step=step, state="done", exit=0, seconds=3)) + "\n"
        return ""

    def _download(self, remote, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        if remote.endswith("out.tar.gz"):
            with tarfile.open(local, "w:gz") as tar:
                for name, data in {"out/setup.json": json.dumps(dict(python="3.10.19", install_s=60.0,
                                                                        pip_freeze_sha256="f" * 64,
                                                                        versions={"torch": "2.14.1+cu130"})),
                                   "out/cmd1/command.log": "ok\n"}.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data.encode())
                    tar.addfile(info, io.BytesIO(data.encode()))
        else:
            local.write_bytes(b"cache")

    def subcommands(self):
        return [c[1] for c in self.calls]

    def stages(self):
        return [next(a.split("=", 1)[1] for a in c if a.startswith("STAGE=")) for c in self.calls if c[1] == "exec"]


def _job_args(tmp_path, *extra):
    repo = _git_repo(tmp_path)
    baseline = tmp_path / "1a-T18"
    baseline.mkdir()
    (tmp_path / "store").mkdir()
    return fc.parse_args(["--gpu", "G4", "--command", "echo one", "--out", str(tmp_path / "out"), "--session", "t1",
                          "--repo", str(repo), "--baseline", str(baseline), "--store", str(tmp_path / "store"),
                          "--ledger", str(tmp_path / "ledger.csv"), *extra])


def test_job_runs_stages_in_order_and_stops(tmp_path):
    colab = FakeColab()
    args = _job_args(tmp_path)
    record = fc.run_job(args, colab)
    assert colab.subcommands()[:3] == ["usage", "sessions", "new"]
    assert colab.stages() == ["probe", "setup", "restore", "run", "pack"]
    assert colab.calls[-2] == fc.stop_args("t1") and colab.subcommands()[-1] == "usage"
    assert record["exit"] == 0 and record["capability"] == "12.0" and record["cache_hit"] is False
    assert record["cu_before"] == 198.99 and record["cu_after"] == pytest.approx(198.49)
    saved = json.loads((tmp_path / "out" / "colab.json").read_text())
    assert saved["provider"] == "colab" and saved["commands"][0]["command"] == "echo one"
    assert (tmp_path / "out" / "out" / "cmd1" / "command.log").exists()
    key = record["cache_key"]
    assert (tmp_path / "store" / "caches" / key / "000.tar.gz").exists()
    assert list(csv.DictReader((tmp_path / "ledger.csv").open()))[0]["session"] == "t1"


def test_warm_cache_is_uploaded_and_an_unchanged_one_not_downloaded(tmp_path):
    args = _job_args(tmp_path)
    fc.run_job(args, FakeColab())
    colab = FakeColab(cache_changed=False)
    record = fc.run_job(args, colab)
    assert record["cache_hit"] is True and record["cache_saved"] is False
    uploads = [c for c in colab.calls if c[1] == "upload"]
    assert [c[-1] for c in uploads if "cache-in" in c[-1]] == ["/content/fm_cache-in-000.tar.gz"]
    downloads = [c for c in colab.calls if c[1] == "download"]
    assert not any(c[-2].endswith("cache-out.tar.gz") for c in downloads)


@pytest.mark.parametrize("stage", ["setup", "run"])
def test_vm_stops_on_error(tmp_path, stage):
    colab = FakeColab(fail_on=stage)
    with pytest.raises(subprocess.CalledProcessError):
        fc.run_job(_job_args(tmp_path), colab)
    assert fc.stop_args("t1") in colab.calls
    assert list(csv.DictReader((tmp_path / "ledger.csv").open()))[0]["exit"] == "error"


def test_vm_stops_on_ctrl_c(tmp_path):
    colab = FakeColab(interrupt_on="run")
    with pytest.raises(KeyboardInterrupt):
        fc.run_job(_job_args(tmp_path), colab)
    assert fc.stop_args("t1") in colab.calls


def test_keep_leaves_the_vm_running(tmp_path):
    colab = FakeColab()
    fc.run_job(_job_args(tmp_path, "--keep"), colab)
    assert "stop" not in colab.subcommands()


def test_dry_run_prints_and_touches_nothing(tmp_path, capsys):
    colab = FakeColab()
    fc.run_job(_job_args(tmp_path, "--dry-run"), colab)
    assert colab.calls == []
    assert "colab new --gpu G4 -s t1" in capsys.readouterr().out


def test_transfers_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    fake = FakeColab()
    failures = {"n": 0}

    def flaky(argv, timeout=None):
        if argv[1] == "upload" and failures["n"] < 2:
            failures["n"] += 1
            raise subprocess.CalledProcessError(1, argv, "500 Server Error")
        return fake(argv, timeout)

    record = fc.run_job(_job_args(tmp_path), flaky)
    assert record["exit"] == 0 and failures["n"] == 2


def test_changed_caches_add_a_part_and_all_parts_go_up(tmp_path):
    args = _job_args(tmp_path)
    first = fc.run_job(args, FakeColab())
    fc.run_job(args, FakeColab(cache_changed=True))
    parts = fc.cache_parts(tmp_path / "store" / "caches" / first["cache_key"])
    assert [p.name for p in parts] == ["000.tar.gz", "001.tar.gz"]
    colab = FakeColab(cache_changed=False)
    fc.run_job(args, colab)
    assert [c[-1] for c in colab.calls if c[1] == "upload" and "cache-in" in c[-1]] == [
        "/content/fm_cache-in-000.tar.gz", "/content/fm_cache-in-001.tar.gz"]


def test_large_files_are_split_for_upload(tmp_path):
    big = tmp_path / "big.bin"
    data = bytes(range(256)) * 41  # 10496 bytes
    big.write_bytes(data)
    assert fc.split_file(big, tmp_path / "c", len(data)) == [big]
    chunks = fc.split_file(big, tmp_path / "c", 4096)
    assert [c.name for c in chunks] == ["big.bin.chunk000", "big.bin.chunk001", "big.bin.chunk002"]
    assert b"".join(c.read_bytes() for c in chunks) == data


def test_a_large_cache_part_goes_up_in_chunks(tmp_path, monkeypatch):
    args = _job_args(tmp_path)
    first = fc.run_job(args, FakeColab())
    part = tmp_path / "store" / "caches" / first["cache_key"] / "000.tar.gz"
    part.write_bytes(b"x" * 2500)
    monkeypatch.setattr(fc, "CHUNK_BYTES", 1000)
    colab = FakeColab(cache_changed=False)
    fc.run_job(args, colab)
    assert [c[-1] for c in colab.calls if c[1] == "upload" and "cache-in" in c[-1]] == [
        f"/content/fm_cache-in-000.tar.gz.chunk{n:03d}" for n in range(3)]


def test_a_part_claimed_meanwhile_is_not_overwritten(tmp_path):
    cache = tmp_path / "k"
    cache.mkdir()
    (cache / "000.tar.gz").write_bytes(b"a")
    incoming = cache / ".incoming-s"
    incoming.write_bytes(b"b")
    real_link = fc.os.link

    def racing_link(src, dst):  # another job takes 001 between the listing and the link
        if Path(dst).name == "001.tar.gz" and not Path(dst).exists():
            Path(dst).write_bytes(b"other")
        return real_link(src, dst)

    fc.os.link = racing_link
    try:
        part = fc.claim_part(cache, incoming)
    finally:
        fc.os.link = real_link
    assert part.name == "002.tar.gz" and part.read_bytes() == b"b"
    assert (cache / "001.tar.gz").read_bytes() == b"other" and not incoming.exists()


def test_a_live_name_is_refused_and_left_running(tmp_path):
    colab = FakeColab(live=("t1",))
    with pytest.raises(fc.SessionExists):
        fc.run_job(_job_args(tmp_path), colab)
    assert "new" not in colab.subcommands() and "stop" not in colab.subcommands()
    assert list(csv.DictReader((tmp_path / "ledger.csv").open()))[0]["exit"] == "error"


def test_reuse_runs_on_the_live_session(tmp_path):
    colab = FakeColab(live=("t1",))
    record = fc.run_job(_job_args(tmp_path, "--reuse", "--keep"), colab)
    assert record["exit"] == 0 and record["reused_session"] is True
    assert "new" not in colab.subcommands() and "stop" not in colab.subcommands()


def test_an_error_before_allocation_stops_nothing(tmp_path):
    args = _job_args(tmp_path)
    args.commit = "no-such-commit"
    colab = FakeColab()
    with pytest.raises(subprocess.CalledProcessError):
        fc.run_job(args, colab)
    assert "stop" not in colab.subcommands()


SESSIONS = """\
[fm-g4-compat-warm] gpu-g4-s-ft-kkb-euw4a2-hct3ydixyi0m | Hardware: G4 | Shape: Standard | Variant: GPU
[?] gpu-a100-s-usc1f0-1kxof2v4qlkf2 | Hardware: A100 | Shape: Standard | Variant: GPU
"""


def test_stop_endpoint_releases_only_an_unnamed_live_one():
    assert fc.parse_sessions(SESSIONS) == {"gpu-g4-s-ft-kkb-euw4a2-hct3ydixyi0m": "fm-g4-compat-warm",
                                           "gpu-a100-s-usc1f0-1kxof2v4qlkf2": "?"}
    runs = []
    listing = lambda argv, timeout=None: SESSIONS  # noqa: E731
    run = lambda argv, **kw: runs.append(argv)  # noqa: E731
    fc.stop_endpoint("gpu-a100-s-usc1f0-1kxof2v4qlkf2", colab=listing, run=run, python="/py")
    assert runs == [["/py", "-c", fc.UNASSIGN, "gpu-a100-s-usc1f0-1kxof2v4qlkf2"]]
    assert "state.client.unassign(sys.argv[1])" in fc.UNASSIGN
    for refused in ("gpu-g4-s-ft-kkb-euw4a2-hct3ydixyi0m", "gpu-x-not-live", "a;rm -rf /", "-x"):
        with pytest.raises(SystemExit):
            fc.stop_endpoint(refused, colab=listing, run=run, python="/py")
    assert len(runs) == 1


def test_a_hung_poll_reattaches_instead_of_failing_the_job(tmp_path, monkeypatch):
    # 4-T13: one `colab exec` poll of a 2 h command hung past its subprocess timeout; the job must go on.
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    fake = FakeColab()
    hung = {"n": 0}

    def flaky(argv, timeout=None):
        if argv[1] == "exec" and "STAGE=run" in argv and hung["n"] < 2:
            hung["n"] += 1
            if hung["n"] == 1:
                raise subprocess.TimeoutExpired(argv, timeout)
            raise subprocess.CalledProcessError(1, argv, "websocket closed")
        return fake(argv, timeout)

    record = fc.run_job(_job_args(tmp_path), flaky)
    assert record["exit"] == 0 and hung["n"] == 2
    assert record["commands"][0]["exit"] == 0


def test_polls_give_up_after_repeated_failures_but_salvage_the_records(tmp_path, monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    fake = FakeColab()

    def dead(argv, timeout=None):
        if argv[1] == "exec" and "STAGE=run" in argv:
            raise subprocess.TimeoutExpired(argv, timeout)
        return fake(argv, timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        fc.run_job(_job_args(tmp_path), dead)
    assert fc.stop_args("t1") in fake.calls
    assert "pack" in fake.stages()  # the records were packed and downloaded before the stop
    assert (tmp_path / "out" / "out" / "setup.json").exists()


def test_warm_names_its_session_and_keeps_it():
    args = fc.parse_args(["--gpu", "A100", "--warm", "--command", "x", "--out", "/x"])
    assert args.session == "fm-a100-warm" and args.reuse and args.keep and args.idle_stop_min == 20.0
    with pytest.raises(SystemExit):
        fc.parse_args(["--gpu", "A100", "--warm", "--session", "other", "--command", "x", "--out", "/x"])


def test_warm_vm_gets_only_the_cache_parts_it_lacks_and_marks_its_own(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(fc, "start_idle_stop", lambda *a: started.append(a))
    args = _job_args(tmp_path)
    first = fc.run_job(args, FakeColab())  # store part 000
    fc.run_job(args, FakeColab(cache_changed=True))  # store part 001
    key = first["cache_key"]
    warm = fc.argparse.Namespace(**vars(args))
    warm.session, warm.reuse, warm.keep, warm.warm = "fm-g4-warm", True, True, True
    colab = FakeColab(live=("fm-g4-warm",), parts=(f"{key}/000.tar.gz", "sm80-py310-other/001.tar.gz"))
    record = fc.run_job(warm, colab)
    assert [c[-1] for c in colab.calls if c[1] == "upload" and "cache-in" in c[-1]] == [
        "/content/fm_cache-in-001.tar.gz"]  # 000 is on the VM; the other key's 001 is not this one
    assert record["cache_parts_on_vm"] == ["000.tar.gz"] and "new" not in colab.subcommands()
    restore = next(c for c in colab.calls if "STAGE=restore" in c)
    assert f"ARG={key}" in restore
    # the changed caches became part 002, which the kept VM holds already: marked, and the VM stays
    mark = [c for c in colab.calls if "STAGE=mark" in c]
    assert colab.stages()[-1] == "mark" and len(mark) == 1 and f"ARG={key}/002.tar.gz" in mark[0]
    assert "stop" not in colab.subcommands() and started == []
    assert fc.warm_lock_path(warm.store, "fm-g4-warm").exists()


def test_a_warm_vm_allocated_by_a_job_gets_its_idle_stop(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(fc, "start_idle_stop", lambda *a: started.append(a))
    warm = _job_args(tmp_path)
    warm.session, warm.reuse, warm.keep, warm.warm = "fm-g4-warm", True, True, True
    colab = FakeColab()
    fc.run_job(warm, colab)
    assert "new" in colab.subcommands() and "stop" not in colab.subcommands()
    assert len(started) == 1 and started[0][:2] == ("fm-g4-warm", 20.0)


def test_idle_stop_waits_for_jobs_and_idleness(tmp_path):
    store = tmp_path / "store"
    lock = fc.warm_lock_path(store, "w")
    lock.parent.mkdir(parents=True)
    lock.touch()
    now = {"t": lock.stat().st_mtime}
    calls = []

    def colab(argv, timeout=None):
        calls.append(argv)
        return "[w] gpu-x | Hardware: A100\n" if argv[1] == "sessions" else ""

    def sleep(seconds):
        now["t"] += seconds

    assert fc.idle_stop("w", 3, store, colab=colab, clock=lambda: now["t"], sleep=sleep) is True
    assert fc.stop_args("w") in calls
    assert now["t"] - lock.stat().st_mtime >= 180  # it stopped only once three idle minutes had passed
    stops = [c for c in calls if c[1] == "stop"]
    assert len(stops) == 1 and len([c for c in calls if c[1] == "sessions"]) == 3


def test_idle_stop_leaves_a_busy_vm_and_ends_when_it_is_gone(tmp_path):
    store = tmp_path / "store"
    sessions = iter(["[w] x\n", "[w] x\n", ""])
    calls = []

    def colab(argv, timeout=None):
        calls.append(argv)
        return next(sessions) if argv[1] == "sessions" else ""

    with fc.warm_lock(store, "w"):  # a job holds the VM: never stopped under it
        import threading

        result = {}
        thread = threading.Thread(target=lambda: result.update(
            r=fc.idle_stop("w", 0, store, colab=colab, sleep=lambda s: None)))
        thread.start()
        thread.join(10)
    assert result["r"] is False and not any(c[1] == "stop" for c in calls)
