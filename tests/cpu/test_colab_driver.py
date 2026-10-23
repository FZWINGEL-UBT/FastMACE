"""tools/colab/fastmace_colab.py (plan task ops-colab): command building, parsing, ledger and the
always-stop guarantee, with the colab CLI faked (no network, no VM)."""

import csv
import importlib.util
import io
import json
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
    assert name == "fm-g4-20261023-200501"


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
    assert fc.parse_usage(USAGE) == {"balance": 198.99, "rate_per_h": 5.30}
    assert fc.parse_usage("nonsense") == {"balance": None, "rate_per_h": None}
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


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
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

    def __init__(self, fail_on=None, interrupt_on=None, capability="12.0", cache_changed=True):
        self.calls, self.fail_on, self.interrupt_on = [], fail_on, interrupt_on
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
                                      capability=self.capability, cuda_driver="13.0", cpus=48, ram_gib=176.0))
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
