"""tools/ci/gpu_idle.py (plan task 2-T18): the GPU CI runner's idle decision, with nvidia-smi and ps mocked."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_tool_gpu_idle", REPOSITORY_ROOT / "tools" / "ci" / "gpu_idle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gi = _load()

# nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
APPS_NONE = ""
APPS_TWO = "4242, 1830\n5151, 512\n"
APPS_NO_PERMISSION = "4242, [Insufficient Permissions]\n"
# ps -o pid=,user= -p 4242,5151
PS_BOTH = " 4242 margraf\n 5151 zwingel\n"
PS_OWN = " 5151 zwingel\n"
# nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
GPU_IDLE = "NVIDIA RTX 4000 Ada Generation, 0, 1, 20475\n"
GPU_DESKTOP = "NVIDIA GeForce RTX 5060 Ti, 3, 4300, 16311\n"
GPU_LOADED = "NVIDIA GeForce RTX 5060 Ti, 97, 9100, 16311\n"


def test_parsers():
    assert gi.parse_compute_apps(APPS_TWO) == [4242, 5151]
    assert gi.parse_compute_apps(APPS_NO_PERMISSION) == [4242]
    assert gi.parse_compute_apps(APPS_NONE) == []
    assert gi.parse_compute_apps("No running processes found\n") == []
    assert gi.parse_ps(PS_BOTH) == {4242: "margraf", 5151: "zwingel"}
    (gpu,) = gi.parse_gpus(GPU_DESKTOP)
    assert gpu == dict(name="NVIDIA GeForce RTX 5060 Ti", utilization_percent=3.0, memory_used_MiB=4300.0,
                       memory_total_MiB=16311.0)
    assert gi.parse_gpus("garbage\n") == []


def test_no_process_is_idle():
    idle, reasons, processes = gi.decide([], {}, gi.parse_gpus(GPU_IDLE), own_user="zwingel")
    assert idle and reasons == [] and processes == []


def test_another_users_process_is_busy():
    idle, reasons, processes = gi.decide([4242], {4242: "margraf"}, gi.parse_gpus(GPU_IDLE), own_user="zwingel")
    assert not idle
    assert processes == [dict(pid=4242, user="margraf", owner="other user")]
    assert "margraf" in reasons[0]


def test_own_process_from_another_task_is_busy_too():
    # Conductor answer Q2: zwingel's own GPU work (another task) also makes the runner wait.
    idle, reasons, processes = gi.decide([5151], gi.parse_ps(PS_OWN), gi.parse_gpus(GPU_IDLE), own_user="zwingel")
    assert not idle
    assert processes == [dict(pid=5151, user="zwingel", owner="own")]


def test_a_process_ps_cannot_resolve_counts_as_another_users():
    idle, reasons, processes = gi.decide([7777], {}, gi.parse_gpus(GPU_IDLE), own_user="zwingel")
    assert not idle
    assert processes == [dict(pid=7777, user=None, owner="other user")]
    assert "unknown" in reasons[0]


def test_the_checks_own_processes_are_left_out():
    idle, _, processes = gi.decide([31, 32], {31: "zwingel", 32: "zwingel"}, gi.parse_gpus(GPU_IDLE),
                                   own_user="zwingel", ignore_pids={31, 32})
    assert idle and processes == []


def test_no_gpu_listed_is_not_idle():
    idle, reasons, _ = gi.decide([], {}, [], own_user="zwingel")
    assert not idle
    assert "no GPU" in reasons[0]


def test_memory_and_utilization_limits():
    # The local runner (WSL) cannot see Windows programs or other WSL processes by pid through nvidia-smi,
    # so it also judges the GPU's memory and utilization against the desktop's.
    desktop = gi.parse_gpus(GPU_DESKTOP)
    loaded = gi.parse_gpus(GPU_LOADED)
    assert gi.decide([], {}, desktop, own_user="fzwin", max_memory_mib=6000, max_utilization=10)[0]
    idle, reasons, _ = gi.decide([], {}, loaded, own_user="fzwin", max_memory_mib=6000, max_utilization=10)
    assert not idle
    assert any("memory" in reason for reason in reasons)
    assert any("utilization" in reason for reason in reasons)
    # Without the limits (murphy) memory and utilization are not judged.
    assert gi.decide([], {}, loaded, own_user="zwingel")[0]


def test_available_memory_limit(tmp_path):
    # The PC's resource rule: at least 10 GB available in WSL before a GPU suite.
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16000000 kB\nMemAvailable:    7340032 kB\n")
    assert gi.available_gib(meminfo) == 7.0
    assert gi.available_gib(tmp_path / "missing") is None
    gpus = gi.parse_gpus(GPU_DESKTOP)
    idle, reasons, _ = gi.decide([], {}, gpus, own_user="fzwin", available_gib=7.0, min_available_gib=10)
    assert not idle and "available" in reasons[0]
    assert gi.decide([], {}, gpus, own_user="fzwin", available_gib=12.0, min_available_gib=10)[0]
    assert not gi.decide([], {}, gpus, own_user="fzwin", available_gib=None, min_available_gib=10)[0]


def test_unknown_utilization_under_a_limit_is_busy():
    gpus = gi.parse_gpus("NVIDIA GeForce RTX 5060 Ti, [N/A], 4300, 16311\n")
    assert not gi.decide([], {}, gpus, own_user="fzwin", max_utilization=10)[0]


def test_dxg_holders(tmp_path):
    # WSL: CUDA processes hold /dev/dxg open; nvidia-smi lists none of them.
    proc = tmp_path / "proc"
    for pid, target in ((101, "/dev/dxg"), (102, "/dev/null"), (103, "/dev/dxg")):
        (proc / str(pid) / "fd").mkdir(parents=True)
        os.symlink(target, proc / str(pid) / "fd" / "5")
    (proc / "self").mkdir()
    (proc / "104" / "fd").mkdir(parents=True)  # a process whose fds cannot be listed or that has none
    assert gi.dxg_holders(proc) == [101, 103]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def record(idle, reasons=(), processes=()):
    return dict(idle=idle, reasons=list(reasons), processes=list(processes), gpus=[])


def test_wait_returns_at_once_when_idle():
    clock = Clock()
    samples = iter([record(True)])
    result = gi.wait_for_idle(lambda: next(samples), wait_s=600, interval_s=30, clock=clock, sleep=clock.sleep)
    assert result["idle"] and result["waited_s"] == 0 and result["samples"] == 1


def test_wait_until_the_other_user_is_gone():
    clock = Clock()
    samples = iter([record(False, ["margraf"])] * 3 + [record(True)])
    result = gi.wait_for_idle(lambda: next(samples), wait_s=600, interval_s=30, clock=clock, sleep=clock.sleep)
    assert result["idle"] and result["waited_s"] == 90 and result["samples"] == 4
    assert result["busy_reasons_seen"] == ["margraf"]


def test_busy_past_the_limit_gives_up():
    clock = Clock()
    result = gi.wait_for_idle(lambda: record(False, ["margraf"]), wait_s=120, interval_s=30, clock=clock,
                              sleep=clock.sleep)
    assert not result["idle"]
    assert result["waited_s"] == 120
    assert result["reasons"] == ["margraf"]


def test_idle_samples_must_follow_each_other():
    clock = Clock()
    pattern = [True, False, True, True]
    samples = iter([record(value, [] if value else ["load"]) for value in pattern])
    result = gi.wait_for_idle(lambda: next(samples), wait_s=600, interval_s=20, idle_samples=2, clock=clock,
                              sleep=clock.sleep)
    assert result["idle"] and result["samples"] == 4 and result["waited_s"] == 60


def test_no_wait_checks_once():
    clock = Clock()
    result = gi.wait_for_idle(lambda: record(False, ["margraf"]), wait_s=0, interval_s=30, clock=clock,
                              sleep=clock.sleep)
    assert not result["idle"] and result["samples"] == 1


def fake_runner(outputs):
    """A ``run(argv)`` that answers nvidia-smi and ps from ``outputs`` (None: the command failed)."""
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[0] == "ps":
            return outputs.get("ps")
        if any(arg.startswith("--query-compute-apps") for arg in argv):
            return outputs.get("apps")
        return outputs.get("gpus")

    run.calls = calls
    return run


def test_sample_with_mocked_commands():
    run = fake_runner(dict(apps=APPS_TWO, ps=PS_BOTH, gpus=GPU_IDLE))
    result = gi.sample(run, own_user="zwingel", source="nvidia-smi")
    assert not result["idle"]
    assert [p["owner"] for p in result["processes"]] == ["other user", "own"]
    assert ["ps", "-o", "pid=,user=", "-p", "4242,5151"] in run.calls


def test_sample_when_nvidia_smi_fails_is_busy():
    result = gi.sample(fake_runner(dict(apps=None, gpus=None)), own_user="zwingel", source="nvidia-smi")
    assert not result["idle"]
    assert any("nvidia-smi" in reason for reason in result["reasons"])


def test_sample_wsl_uses_dxg_holders(tmp_path):
    proc = tmp_path / "proc"
    (proc / "900" / "fd").mkdir(parents=True)
    os.symlink("/dev/dxg", proc / "900" / "fd" / "3")
    run = fake_runner(dict(ps=" 900 fzwin\n", gpus=GPU_DESKTOP))
    result = gi.sample(run, own_user="fzwin", source="wsl", proc_root=proc)
    assert not result["idle"]
    assert result["processes"] == [dict(pid=900, user="fzwin", owner="own")]
    assert not any(arg.startswith("--query-compute-apps") for argv in run.calls for arg in argv)


def test_a_live_local_reservation_is_busy(tmp_path):
    # The local runner treats someone else's live reservation (tools/ci/local_gpu.py) as a busy GPU.
    proc = tmp_path / "proc"
    (proc / "4040").mkdir(parents=True)  # the holder: alive, no /dev/dxg open
    (proc / "4040" / "stat").write_text("4040 (bash) S 1 " + "0 " * 17 + "5 0 0\n")
    marker = tmp_path / "gpu"
    marker.mkdir()
    (marker / "holder.json").write_text(json.dumps(dict(owner="fzwin", purpose="2-T05 OOM cases", pid=4040,
                                                        start_ticks=5, started="T")))
    run = fake_runner(dict(gpus=GPU_DESKTOP))
    result = gi.sample(run, own_user="fzwin", source="wsl", proc_root=proc, reservation=marker)
    assert not result["idle"]
    assert any("2-T05 OOM cases" in reason for reason in result["reasons"])
    assert gi.sample(run, own_user="fzwin", source="wsl", proc_root=proc, reservation=tmp_path / "none")["idle"]


def test_check_writes_github_output_and_summary(tmp_path, monkeypatch):
    output, summary = tmp_path / "out", tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(gi, "_run", fake_runner(dict(apps="4242, 100\n", ps=" 4242 margraf\n", gpus=GPU_IDLE)))
    status = gi.main(["check", "--wait-minutes", "0", "--user", "zwingel", "--source", "nvidia-smi",
                      "--record", str(tmp_path / "idle.json")])
    assert status == 0  # busy is not an error: the tests job is skipped
    assert "idle=false" in output.read_text().splitlines()
    assert "margraf" in summary.read_text()
    assert json.loads((tmp_path / "idle.json").read_text())["idle"] is False


def test_check_idle(tmp_path, monkeypatch):
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(gi, "_run", fake_runner(dict(apps="", gpus=GPU_IDLE)))
    assert gi.main(["check", "--wait-minutes", "0", "--user", "zwingel", "--source", "nvidia-smi"]) == 0
    assert "idle=true" in output.read_text().splitlines()


def test_watch_records_only_other_users(tmp_path, monkeypatch):
    # Conductor answer Q3: another user who starts during the run is recorded, the run goes on.
    out = tmp_path / "watch.jsonl"
    samples = iter([
        dict(apps="", ps=""),
        dict(apps="5151, 900\n", ps=" 5151 zwingel\n"),
        dict(apps="5151, 900\n4242, 300\n", ps=" 5151 zwingel\n 4242 margraf\n"),
    ])
    current = {}

    def run(argv):
        if argv[0] == "ps":
            return current.get("ps")
        if any(arg.startswith("--query-compute-apps") for arg in argv):
            current.clear()
            current.update(next(samples))
            return current["apps"]
        return GPU_IDLE

    monkeypatch.setattr(gi, "_run", run)
    gi.watch(out, own_user="zwingel", source="nvidia-smi", interval_s=0, rounds=3, sleep=lambda s: None)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["processes"] == [dict(pid=4242, user="margraf", owner="other user")]
    assert "margraf" in gi.summarize_watch(out)
    assert "No other user" in gi.summarize_watch(tmp_path / "missing.jsonl")


@pytest.mark.parametrize("tier, expression", [("fast", "not slow and not acceptance"), ("full", "not acceptance"),
                                              ("acceptance", "acceptance")])
def test_marker_expressions_come_from_the_tiers(tier, expression, capsys):
    assert gi.main(["marks", tier]) == 0
    assert capsys.readouterr().out.strip() == expression
