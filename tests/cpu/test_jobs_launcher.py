"""The packed-job launcher (tools/jobs/launch.py; AGENTS.md "Using resources"): manifest parsing, packing,
data-based timeouts and the WSL limits (client cap, launch gap, one retry), with Modal and the clock faked."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "tools" / "jobs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


launch = _load("fastmace_jobs_launch", JOBS / "launch.py")
slim_tar = _load("fastmace_jobs_slim_tar", JOBS / "slim_tar.py")
P90 = launch.load_p90(JOBS / "p90.json")
T = "fastmace.kernels.triton.tabled_outer_sum"
HEADER = "idea,switch,gpu,profile,timeout\n"
HEADER7 = "idea,switch,gpu,profile,timeout,kind,name\n"


def _manifest(*rows, header=HEADER):
    return header + "".join(r + "\n" for r in rows)


def _ab_rows(n, gpu="H200", profile="fourth"):
    return [f"i{k},{T}:CHUNK={k},{gpu},{profile}," for k in range(n)]


# --- manifest -----------------------------------------------------------------------------------------------------

def test_parse_a_manifest_with_comments_and_optional_columns():
    text = ("# batch 7\n" "idea,switch,gpu,profile,timeout,kind,name\n"
            f"i41,{T}:CHUNK=8,H200,fourth,,ab,tosc8\n"
            "\n"
            f"i42,{T}:CHUNK=32 {T}:FORWARD_CHUNK=16,A100-80GB,second,30,gate,\n"
            "i43,,H200,fzwingel,,profile,\n")
    rows = launch.parse_manifest(text)
    assert [(r.idea, r.kind, r.name, r.gpu, r.profile, r.timeout_min) for r in rows] == [
        ("i41", "ab", "tosc8", "H200", "fourth", None), ("i42", "gate", "i42", "A100-80GB", "second", 30),
        ("i43", "profile", "i43", "H200", "fzwingel", None)]
    assert rows[1].switch == f"{T}:CHUNK=32 {T}:FORWARD_CHUNK=16" and rows[1].line == 5


def test_switch_may_quote_a_value_with_csv_quotes():
    (row,) = launch.parse_manifest(_manifest('i1,"m.x:TILE=(32,64)",H200,fourth,'))
    assert row.switch == "m.x:TILE=(32,64)"


def test_ab_rewrite_words_pass_through():
    (row,) = launch.parse_manifest(_manifest("i1,--b radial_tables=off,H200,fourth,"))
    assert row.switch == "--b radial_tables=off"


@pytest.mark.parametrize("header, match", [
    ("idea,switch,gpu,profile\n", "missing"),
    ("idea,switch,gpu,profile,timeout,colour\n", "unknown"),
    ("idea,idea,switch,gpu,profile,timeout\n", "header"),
])
def test_bad_headers_are_refused(header, match):
    with pytest.raises(launch.ManifestError, match=match):
        launch.parse_manifest(header + f"i1,{T}:CHUNK=8,H200,fourth,\n")


@pytest.mark.parametrize("row, match", [
    (f"i1,{T}:CHUNK=8,B200,fourth,", "gpu"),
    (f"i 1,{T}:CHUNK=8,H200,fourth,", "idea"),
    (f"i1,{T}:CHUNK=8,H200,Fourth!,", "profile"),
    (f"i1,{T}:CHUNK=8,H200,fourth,0", "timeout"),
    (f"i1,{T}:CHUNK=8,H200,fourth,1.5", "timeout"),
    ("i1,CHUNK=8,H200,fourth,", "MODULE:NAME=VALUE"),
    ("i1,,H200,fourth,", "empty switch"),
    (f"i1,{T}:CHUNK=8;rm -rf ~,H200,fourth,", "separators or shell"),
    (f"i1,{T}:CHUNK=$(id),H200,fourth,", "separators or shell"),
    (f"i1,{T}:CHUNK=8|x,H200,fourth,", "separators or shell"),
    (f"i1,{T}:CHUNK=8,H200,fourth", "fields"),
])
def test_bad_rows_are_refused_with_their_line(row, match):
    with pytest.raises(launch.ManifestError, match=match) as error:
        launch.parse_manifest(_manifest(f"i0,{T}:CHUNK=4,H200,fourth,", row))
    assert "line 3" in str(error.value)


@pytest.mark.parametrize("row, match", [
    (f"i1,{T}:CHUNK=8,H200,fourth,,ab,a/b", "name"),
    (f"i1,{T}:CHUNK=8,H200,fourth,,bench,", "kind"),
    ("i1,--b radial_tables=off,H200,fourth,,gate,", "MODULE:NAME=VALUE"),
])
def test_bad_kind_and_name_are_refused(row, match):
    with pytest.raises(launch.ManifestError, match=match):
        launch.parse_manifest(_manifest(row, header=HEADER7))


def test_duplicate_switches_are_refused():
    with pytest.raises(launch.ManifestError, match="twice"):
        launch.parse_manifest(_manifest(f"i1,{T}:CHUNK=8,H200,fourth,", f"i2,{T}:CHUNK=8,H200,fourth,"))


def test_empty_manifests_are_refused():
    for text in ("", "# only a comment\n", HEADER):
        with pytest.raises(launch.ManifestError):
            launch.parse_manifest(text)


# --- packing and timeouts -----------------------------------------------------------------------------------------

def test_rows_pack_by_kind_gpu_and_profile_at_most_eight():
    rows = launch.parse_manifest(_manifest(*_ab_rows(11), *_ab_rows(4, "A100-80GB", "second")))
    jobs = launch.plan(rows, "b7", P90)
    assert [(j.id, len(j.entries)) for j in jobs] == [
        ("b7-ab-h200-fourth-1", 8), ("b7-ab-h200-fourth-2", 3), ("b7-ab-a100-second-1", 4)]
    assert jobs[2].ideas == ["i0", "i1", "i2", "i3"]


def test_a_single_switch_pack_needs_allow_single():
    rows = launch.parse_manifest(_manifest(*_ab_rows(9)))  # 8 + 1
    with pytest.raises(launch.ManifestError, match="one switch"):
        launch.plan(rows, "b7", P90)
    assert len(launch.plan(rows, "b7", P90, allow_single=True)) == 2


def test_profiles_run_one_per_job():
    rows = launch.parse_manifest(_manifest("i1,,H200,fourth,,profile", f"i2,{T}:CHUNK=8,H200,fourth,,profile",
                                           header="idea,switch,gpu,profile,timeout,kind\n"))
    jobs = launch.plan(rows, "b7", P90)
    assert [len(j.entries) for j in jobs] == [1, 1]


def test_repeated_names_in_a_pack_get_a_suffix():
    rows = launch.parse_manifest(_manifest(f"i1,{T}:CHUNK=8,H200,fourth,", f"i1,{T}:CHUNK=16,H200,fourth,"))
    (job,) = launch.plan(rows, "b7", P90)
    assert [name for name, _ in job.entries] == ["i1", "i1-2"]


def test_timeout_is_twice_the_p90_rounded_up():
    # ab on the H200: p90 = 90 + 150 x 4 = 690 s -> 1380 s = 23 min (AGENTS.md: "about 20 min for an H200 A/B").
    assert launch.p90_s(P90, "ab", "H200", 4) == 690 and launch.timeout_min(P90, "ab", "H200", 4) == 23
    # A100-80GB uses the A100 fit: 120 + 180 x 4 = 840 s -> 28 min.
    assert launch.timeout_min(P90, "ab", "A100-80GB", 4) == 28
    # a gate pack of 2 switches runs 3 gates.
    assert launch.p90_s(P90, "gate", "H200", 2) == 30 + 3 * 310
    assert launch.timeout_min(P90, "profile", "H200", 1) == 20


def test_an_explicit_timeout_wins_and_the_largest_counts():
    rows = launch.parse_manifest(_manifest(f"i1,{T}:CHUNK=8,H200,fourth,40", f"i2,{T}:CHUNK=16,H200,fourth,",
                                           f"i3,{T}:CHUNK=32,H200,fourth,55"))
    (job,) = launch.plan(rows, "b7", P90)
    assert job.timeout_min == 55


def test_every_template_and_gpu_has_a_p90():
    for kind in launch.KINDS:
        for gpu in launch.GPUS:
            assert launch.p90_s(P90, kind, gpu, 4) > 0


def test_the_command_quotes_the_pack_and_names_the_template():
    rows = launch.parse_manifest(_manifest('i1,"m.x:TILE=(32,64)",A100-80GB,second,',
                                           f"i2,{T}:CHUNK=8,A100-80GB,second,"))
    (job,) = launch.plan(rows, "b7", P90, env={"CELL": "16x16x16"})
    words = shlex.split(job.command())
    assert words[-2:] == ["bash", "tools/jobs/ab_pack.sh"]
    env = dict(w.split("=", 1) for w in words[:-2])
    assert env == {"CELL": "16x16x16", "EXPECT_GPU": "A100", "PACK": f"i1|m.x:TILE=(32,64);i2|{T}:CHUNK=8"}


def test_template_env_is_limited_to_known_variables():
    assert launch.parse_env(["CELL=16x16x16", "BLOCKS=4"]) == {"CELL": "16x16x16", "BLOCKS": "4"}
    for bad in ("PATH=/tmp", "CELL=1 2", "EXPECT_GPU=L4"):
        with pytest.raises(launch.ManifestError):
            launch.parse_env([bad])


def test_templates_exist_and_are_versioned():
    for name in launch.KINDS.values():
        text = (JOBS / name).read_text()
        assert " v1" in text.splitlines()[1] and "EXPECT_GPU" in text and "set -uo pipefail" in text
    for name in launch.TEMPLATE_FILES:
        assert (JOBS / name).is_file(), name
    for name in launch.MODAL_FILES:
        assert (ROOT / "tools" / "modal" / name).is_file(), name


# --- the runner: client cap, launch gap, retries ------------------------------------------------------------------

class FakeModal:
    """Jobs that take ``duration`` fake seconds; ``outcomes[job_id]`` lists the outcome of each launch."""

    def __init__(self, outcomes=None, duration=600.0, others=0):
        self.now = 0.0
        self.outcomes = outcomes or {}
        self.duration = duration
        self.others = others  # Modal clients of other batches on the machine
        self.live = {}
        self.launches = []
        self.peak = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def clients(self):
        return self.others + len(self.live)

    def launch(self, job, attempt):
        handle = (job.id, attempt, len(self.launches))
        self.launches.append((self.now, job.id, attempt))
        self.live[handle] = self.now + self.duration
        self.peak = max(self.peak, self.clients())
        return handle

    def poll(self, handle):
        if self.now < self.live[handle]:
            return None
        del self.live[handle]
        queue = self.outcomes.get(handle[0], [])
        return queue.pop(0) if queue else "ok"


def _jobs(n):
    rows = launch.parse_manifest(_manifest(*[f"i{k},{T}:CHUNK={k},H200,p{k},2" for k in range(n)]))
    return launch.plan(rows, "b9", P90, allow_single=True)


def _runner(fake, jobs, **kwargs):
    return launch.Runner(jobs, launch=fake.launch, poll=fake.poll, clients=fake.clients, clock=fake.clock,
                         sleep=fake.sleep, log=lambda msg: None, **kwargs)


def test_the_limits_are_the_owners():
    assert launch.MAX_CLIENTS == 6 and launch.LAUNCH_GAP_S == 25 and launch.MAX_ATTEMPTS == 2
    assert launch.MAX_PACK == 8


def test_at_most_six_clients_launched_25_s_apart():
    fake = FakeModal()
    jobs = _jobs(14)
    assert _runner(fake, jobs).run() == []
    assert fake.peak == 6
    times = [t for t, _, _ in fake.launches]
    assert len(times) == 14 and all(b - a >= 25 for a, b in zip(times, times[1:]))
    assert all(job.state == "done" and job.attempts == 1 for job in jobs)


def test_clients_of_other_batches_count_against_the_cap():
    fake = FakeModal(others=4)
    _runner(fake, _jobs(6)).run()
    assert fake.peak == 6  # 4 others + at most 2 of ours


def test_no_launch_while_the_machine_is_full():
    fake = FakeModal(others=6)
    runner = _runner(fake, _jobs(2))
    for _ in range(20):
        runner.step()
        fake.sleep(5)
    assert fake.launches == []


def test_a_failed_job_is_retried_once_then_left_for_a_person():
    fake = FakeModal(outcomes={"b9-ab-h200-p0-1": ["failed", "failed"], "b9-ab-h200-p1-1": ["failed", "ok"]})
    jobs = _jobs(3)
    failed = _runner(fake, jobs).run()
    assert [j.id for j in failed] == ["b9-ab-h200-p0-1"]
    by_id = {j.id: j for j in jobs}
    assert by_id["b9-ab-h200-p0-1"].attempts == 2 and by_id["b9-ab-h200-p1-1"].state == "done"
    assert sum(1 for _, job_id, _ in fake.launches if job_id == "b9-ab-h200-p0-1") == 2


def test_budget_refusals_wait_without_using_the_retry():
    fake = FakeModal(outcomes={"b9-ab-h200-p0-1": ["refused", "refused", "failed", "ok"]}, duration=60)
    (job,) = _jobs(1)
    assert _runner(fake, [job], wait_s=90).run() == []
    assert job.state == "done" and job.waits == 2 and job.attempts == 2
    launches = [t for t, _, _ in fake.launches]
    assert launches[1] - launches[0] >= 60 + 90


def test_refusals_stop_after_max_waits():
    fake = FakeModal(outcomes={"b9-ab-h200-p0-1": ["refused"] * 10}, duration=1)
    (job,) = _jobs(1)
    failed = _runner(fake, [job], wait_s=1, max_waits=3).run()
    assert failed == [job] and job.waits == 3


def test_count_clients_reads_proc(tmp_path):
    def proc(pid, *args):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
    proc(10, "/home/u/environments/modal_client/bin/python", "/home/u/environments/modal_client/bin/modal", "run",
         "/home/u/fm-runs/b1/tools/modal/fastmace_modal.py::job", "--gpu", "H200")
    proc(11, "bash", "x.sh")
    proc(12, "modal", "run", "tools/modal/fastmace_modal.py::job")
    (tmp_path / "self").mkdir()
    assert launch.count_clients(tmp_path) == 2


# --- the batch folder and the unit --------------------------------------------------------------------------------

def test_templates_are_copied_into_a_new_batch_folder(tmp_path):
    batch_dir = launch.make_batch_dir(tmp_path, "b7")
    launch.copy_templates(batch_dir)
    for name in launch.TEMPLATE_FILES:
        assert (batch_dir / "tools" / "jobs" / name).read_bytes() == (JOBS / name).read_bytes()
    for name in launch.MODAL_FILES:
        assert (batch_dir / "tools" / "modal" / name).is_file()
    with pytest.raises(launch.ManifestError, match="exists"):
        launch.make_batch_dir(tmp_path, "b7")


@pytest.mark.parametrize("batch", ["../b7", "b7/x", "", ".hidden", "a b"])
def test_batch_ids_cannot_leave_the_root(tmp_path, batch):
    with pytest.raises(launch.ManifestError):
        launch.make_batch_dir(tmp_path, batch)


def test_plan_round_trips_through_the_batch_folder(tmp_path):
    jobs = _jobs(3)
    jobs[0].history.append([1.0, "launched attempt 1"])
    launch.save_plan(tmp_path, jobs, {"batch": "b9", "tarball": "/x.tar"})
    meta, loaded = launch.load_plan(tmp_path / "plan.json")
    assert meta == {"batch": "b9", "tarball": "/x.tar"}
    assert [j.command() for j in loaded] == [j.command() for j in jobs]
    assert loaded[0].history == [[1.0, "launched attempt 1"]]


@pytest.mark.skipif(not hasattr(__import__("os"), "getuid"), reason="POSIX only")
def test_the_unit_is_a_collected_system_unit_running_the_batch_copy(tmp_path):
    words = launch.unit_command("b7", tmp_path)
    assert words[:3] == ["sudo", "-n", "systemd-run"] and "--collect" in words and "--unit=fm-batch-b7" in words
    assert any(w.startswith("--uid=") for w in words) and "--user" not in words
    assert words[-3:] == [str(tmp_path / "tools" / "jobs" / "launch.py"), "run", str(tmp_path / "plan.json")]


def test_keepalive_covers_the_batch():
    jobs = _jobs(7)  # two waves of 6
    assert launch.keepalive_seconds(jobs) >= 2 * (2 * 60 + 900)
    assert launch.keepalive_seconds(jobs * 100) == 24 * 3600


def test_slim_tarball_keeps_code_and_the_job_inputs():
    assert slim_tar.keep("fastmace/engine.py") and slim_tar.keep("tools/jobs/ab_pack.sh")
    assert slim_tar.keep("results/loup-a100-20261103/SrTiO3.cif")
    assert not slim_tar.keep("docs/design/conversations.zip") and not slim_tar.keep("results/x/profile.json")
    assert not slim_tar.keep("runs/a.log")


def test_the_plan_command_validates_without_launching(tmp_path, capsys):
    manifest = tmp_path / "m.csv"
    manifest.write_text(_manifest(*_ab_rows(5)))
    assert launch.main(["plan", str(manifest), "--batch", "b7"]) == 0
    out = capsys.readouterr().out
    assert "b7-ab-h200-fourth-1: ab on H200/fourth, 5 switch(es), timeout 28 min" in out
    assert json.loads((JOBS / "p90.json").read_text())["version"] == 1
