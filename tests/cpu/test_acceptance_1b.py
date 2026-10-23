"""Acceptance tools of plan task 1b-T23, without a GPU.

- ``fastmace.bench.versions``: the 1a release as a renamed package (``fastmace_1a``) beside 1b in one
  process, and the swap of the process-wide patches both versions install (conductor answer to the
  prestart report: accepted only if the duplicated patches are handled the way ``bench.ab`` handles
  Inductor's custom-pass slot);
- ``tools/ab_fastmace.py``: 1b against 1a, its refusals and its plan check;
- ``tools/ab_frontends.py``: ``Simulation``, ``fm.md`` and ``fastmace.ase.Langevin`` in one process;
- ``tools/library_gate.py``: the in-library reduced gate per label;
- the 1b suites of ``tools/acceptance/suites.py``.

The rebuilt force/stress/energy gate driver is tested in test_acceptance_tools.py and its rule in
test_bench.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fastmace.bench import step_time, versions

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY_ROOT / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TOOLS))  # the tools import each other by file name, as when run as scripts
ab_fastmace = _load(TOOLS / "ab_fastmace.py", "_tool_ab_fastmace_t23")
ab_frontends = _load(TOOLS / "ab_frontends.py", "_tool_ab_frontends_t23")
library_gate = _load(TOOLS / "library_gate.py", "_tool_library_gate_t23")
sys.path.insert(0, str(TOOLS / "acceptance"))
import suites  # noqa: E402

# --- the renamed 1a package --------------------------------------------------------------------------------


@pytest.mark.parametrize("source, expected", [
    ("from fastmace.model import loading\n", "from fastmace_1a.model import loading\n"),
    ("import fastmace  # noqa\n", "import fastmace_1a  # noqa\n"),
    ("import fastmace.engine.graph as g\n", "import fastmace_1a.engine.graph as g\n"),
    ('@torch.library.custom_op("fastmace::edge_outer_sum_forward", mutates_args=())',
     '@torch.library.custom_op("fastmace_1a::edge_outer_sum_forward", mutates_args=())'),
    ("y = torch.ops.fastmace.inplace_block_linear(x)", "y = torch.ops.fastmace_1a.inplace_block_linear(x)"),
    ('_RECORDED_ELSEWHERE = (("polar_longrange", "fastmace.longrange.polar"),)',
     '_RECORDED_ELSEWHERE = (("polar_longrange", "fastmace_1a.longrange.polar"),)'),
    # its own compiler-cache key: the two versions never share Inductor's custom-pass key
    ('UUID = "fastmace-unfused-full-addmm-v1"', 'UUID = "fastmace_1a-unfused-full-addmm-v1"'),
    # environment variables, other identifiers and paths stay
    ('os.environ["FASTMACE_CACHE"]', 'os.environ["FASTMACE_CACHE"]'),
    ("~/environments/fastmace_cpu", "~/environments/fastmace_cpu"),
    ("myfastmace.x", "myfastmace.x"),
])
def test_rename_source(source, expected):
    assert versions.rename_source(source, "fastmace_1a") == expected


def test_rename_refuses_a_bad_name():
    with pytest.raises(ValueError):
        versions.rename_source("import fastmace", "fastmace")
    with pytest.raises(ValueError):
        versions.rename_source("import fastmace", "fast-mace")


def _git(repository, *args):
    # safe.directory=*: on NFS (murphy's NAS --basetemp) the files' owner uid differs from the login's, and git
    # refuses the repository as of "dubious ownership" otherwise (exit 128).
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repository),
           "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*"}
    done = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repository), "-c", "core.autocrlf=false",
                           *args], capture_output=True, text=True, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({done.returncode}): {done.stderr.strip()}")
    return done.stdout.strip()


@pytest.fixture
def old_release(tmp_path):
    """A repository whose tag ``fastmace-1a`` holds a tiny ``fastmace`` package."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repository = tmp_path / "repo"
    package = repository / "fastmace"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text('VERSION = "1a"\n')
    (package / "sub" / "__init__.py").write_text("")
    (package / "sub" / "mod.py").write_text('from fastmace import VERSION\nOP = "fastmace::op"\n'
                                            'CACHE = "FASTMACE_CACHE"\n')
    (package / "sub" / "kernel.cuh").write_text("// fastmace kernel, unchanged\n")
    (repository / "README.md").write_text("not exported\n")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "1a")
    _git(repository, "tag", "fastmace-1a")
    (package / "__init__.py").write_text('VERSION = "1b"\n')
    _git(repository, "commit", "-q", "-am", "1b")
    return repository


def test_export_renamed_takes_the_tagged_tree(old_release, tmp_path):
    destination = tmp_path / "versions"
    source = versions.export_renamed(old_release, "fastmace-1a", destination, name="fastmace_1a")
    package = destination / "fastmace_1a"
    assert source["name"] == "fastmace_1a" and source["ref"] == "fastmace-1a"
    assert source["commit"] == _git(old_release, "rev-parse", "fastmace-1a^{commit}")
    assert (package / "__init__.py").read_text() == 'VERSION = "1a"\n'  # the tag, not the work tree
    assert (package / "sub" / "mod.py").read_text() == ('from fastmace_1a import VERSION\nOP = "fastmace_1a::op"\n'
                                                       'CACHE = "FASTMACE_CACHE"\n')
    assert (package / "sub" / "kernel.cuh").read_text() == "// fastmace kernel, unchanged\n"  # only .py files
    assert not (destination / "README.md").exists()
    recorded = json.loads((package / versions.SOURCE_FILE).read_text())
    assert recorded == source and recorded["tree_sha256"] == versions.tree_digest(package)
    with pytest.raises(FileExistsError):
        versions.export_renamed(old_release, "fastmace-1a", destination, name="fastmace_1a")


def test_load_renamed_checks_the_commit_and_the_tree(old_release, tmp_path):
    destination = tmp_path / "versions"
    source = versions.export_renamed(old_release, "fastmace-1a", destination, name="fastmace_1a")
    script = f"""
import json, sys
from fastmace.bench import versions
module, record = versions.load_renamed({str(destination)!r}, "fastmace_1a", expected_commit={source['commit'][:7]!r})
import fastmace_1a.sub.mod as mod
print(json.dumps([module.VERSION, mod.OP, record["commit"], module.__file__]))
try:
    versions.load_renamed({str(destination)!r}, "fastmace_1a", expected_commit="0000000")
except versions.VersionError as error:
    print("refused", error)
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPOSITORY_ROOT)}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True)
    first, second = out.stdout.strip().splitlines()
    version, op, commit, file = json.loads(first)
    assert (version, op, commit) == ("1a", "fastmace_1a::op", source["commit"])
    assert Path(file).resolve().parent == (destination / "fastmace_1a").resolve()
    assert second.startswith("refused")


def test_tree_digest_orders_by_posix_path(tmp_path):
    """The copy is exported on one platform and checked on another: the order must not depend on the platform
    (Windows sorts Path objects casefolded, so "B.py" came after "a.py" there)."""
    import hashlib

    (tmp_path / "sub").mkdir()
    for name, data in (("a.py", b"1"), ("B.py", b"2"), ("sub/c.py", b"3"), (versions.SOURCE_FILE, b"{}")):
        (tmp_path / name).write_bytes(data)
    digest = hashlib.sha256()
    for name, data in (("B.py", b"2"), ("a.py", b"1"), ("sub/c.py", b"3")):  # code-point order
        digest.update(name.encode() + b"\0" + data + b"\0")
    assert versions.tree_digest(tmp_path) == digest.hexdigest()


def test_load_renamed_refuses_a_changed_tree(old_release, tmp_path):
    destination = tmp_path / "versions"
    versions.export_renamed(old_release, "fastmace-1a", destination, name="fastmace_1a")
    (destination / "fastmace_1a" / "sub" / "mod.py").write_text("OP = 'edited'\n")
    with pytest.raises(versions.VersionError, match="tree"):
        versions.check_renamed(destination, "fastmace_1a")


# --- the process-wide patches of two versions ---------------------------------------------------------------


@pytest.fixture
def fake_targets(monkeypatch):
    first = types.ModuleType("_fake_mace_blocks")
    first.mask_head = "mace's"
    second = types.ModuleType("_fake_inductor_config")
    second.post_grad_custom_pre_pass = None
    monkeypatch.setitem(sys.modules, first.__name__, first)
    monkeypatch.setitem(sys.modules, second.__name__, second)
    return first, second, ((first.__name__, "mask_head"), (second.__name__, "post_grad_custom_pre_pass"))


def test_process_patches_give_each_version_its_own_patches(fake_targets):
    blocks, inductor, targets = fake_targets
    patches = versions.ProcessPatches(targets)
    patches.enter("1b")
    blocks.mask_head, inductor.post_grad_custom_pre_pass = "1b's", "1b key"  # 1b's build installs its patches
    patches.leave("1b")
    patches.enter("1a")  # 1a starts from MACE's own objects, so its qualification hashes MACE's source
    assert (blocks.mask_head, inductor.post_grad_custom_pre_pass) == ("mace's", None)
    blocks.mask_head, inductor.post_grad_custom_pre_pass = "1a's", "1a key"
    patches.leave("1a")
    patches.enter("1b")
    assert (blocks.mask_head, inductor.post_grad_custom_pre_pass) == ("1b's", "1b key")
    patches.enter("1a")
    assert (blocks.mask_head, inductor.post_grad_custom_pre_pass) == ("1a's", "1a key")
    record = patches.as_dict()
    assert record["targets"] == ["_fake_mace_blocks:mask_head", "_fake_inductor_config:post_grad_custom_pre_pass"]
    assert record["sides"]["1a"]["_fake_mace_blocks:mask_head"] == repr("1a's")
    patches.restore_pristine()
    assert (blocks.mask_head, inductor.post_grad_custom_pre_pass) == ("mace's", None)


def test_process_patches_leave_records_a_change_between_blocks(fake_targets):
    blocks, _, targets = fake_targets
    patches = versions.ProcessPatches(targets)
    patches.enter("1b")
    blocks.mask_head = "1b's"
    patches.leave("1b")
    patches.enter("1b")
    blocks.mask_head = "something else"  # a side that changed a patch while timed
    assert patches.leave("1b") == ["_fake_mace_blocks:mask_head"]


def test_patch_targets_cover_every_patch_both_versions_install():
    """Every ``_put(module, "attribute", ...)`` of fastmace._patches is a swapped target (1b, and 1a, whose
    _patches.py, read from the tag). PolarMACE's class patch is not swapped: ab_fastmace refuses polar models."""
    attributes = {attribute for _, attribute in versions.PATCH_TARGETS}
    for source in (REPOSITORY_ROOT / "fastmace" / "_patches.py").read_text(), _tag_source("fastmace/_patches.py"):
        if source is None:
            continue
        found = set(re.findall(r'_put\(\w+,\s*"(\w+)"', source))
        assert found and found <= attributes, found - attributes


def _tag_source(path):
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(REPOSITORY_ROOT), "show",
                               f"fastmace-1a:{path}"], check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None  # a checkout without the tag (the NAS copy)


# --- tools/ab_fastmace.py: 1b against the 1a tag ------------------------------------------------------------


def _ab_args(tmp_path, *extra):
    return ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v"), *extra]


def test_ab_fastmace_defaults_are_the_acceptance_protocol(tmp_path):
    args = ab_fastmace.parse_args(_ab_args(tmp_path))
    assert args.sides == ["1a", "1b"] and args.blocks == 12 and args.block == 300 and args.warmup == 300
    assert args.model == "mh1-r2scan" and args.backend == "hybrid" and args.ensemble == "nvt"
    assert args.expected_commit == ab_fastmace.FASTMACE_1A_COMMIT == "bb35363"
    assert args.planner_1b == "minimal"


@pytest.mark.parametrize("extra, reason", [
    (("--model", "polar1-s"), "PolarMACE"),
    (("--backend", "cueq"), "cuEq"),
    (("--ensemble", "npt"), "NPT"),
])
def test_ab_fastmace_refuses_what_one_process_cannot_hold(tmp_path, capsys, extra, reason):
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_ab_args(tmp_path, *extra))
    error = capsys.readouterr().err
    assert reason in error and "alternating processes" in error
    # one side per process (the alternating-process fallback) takes them
    args = ab_fastmace.parse_args(_ab_args(tmp_path, *extra, "--sides", "1a"))
    assert args.sides == ["1a"]


def test_ab_fastmace_needs_the_renamed_copy_for_a_1a_side(tmp_path):
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(["--output", str(tmp_path), "--checkpoint", "m"])
    args = ab_fastmace.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--sides", "1b"])
    assert args.renamed is None


def test_ab_fastmace_side_names_and_build_order(tmp_path):
    args = ab_fastmace.parse_args(_ab_args(tmp_path, "--sides", "1b", "1a", "--pattern", "abba"))
    assert args.sides == ["1b", "1a"] and args.pattern == "abba"
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_ab_args(tmp_path, "--sides", "1b", "1b"))


def test_ab_fastmace_ratio_is_1b_over_1a_whatever_the_order():
    assert ab_fastmace.ratio_1b_over_1a({"1a": 10.0, "1b": 10.1}) == pytest.approx(1.01)
    assert ab_fastmace.ratio_1b_over_1a({"1b": 10.1, "1a": 10.0}) == pytest.approx(1.01)


def test_ab_fastmace_plan_check():
    same = ab_fastmace.compare_options({"a": True, "b": False}, {"a": True, "b": False})
    assert (same["equal"], same["differences"], same["as_expected"]) == (True, {}, True)
    different = ab_fastmace.compare_options({"a": True, "b": False}, {"a": True, "b": True, "c": True})
    assert (different["equal"], different["differences"]) == (False, {"b": [False, True], "c": [None, True]})
    assert not different["as_expected"]  # with no allowed changes (2-T23), every difference is unexpected


def test_ab_fastmace_regression_suite():
    machine = suites.MACHINES["murphy"]
    jobs = suites.regression_jobs(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    names = [job.name for job in jobs]
    assert names[:3] == ["regression-ab-abab", "regression-ab-1b-first", "regression-ab-abba"]
    for job in jobs[:3]:
        argv = list(job.argv)
        assert argv[argv.index("--structures") + 1:argv.index("--structures") + 3] == ["NaCl", "NbCl5"]
        assert argv[argv.index("--blocks") + 1] == "12"
        assert "--renamed" in argv and "oeq_no_nvcc.sh" not in " ".join(argv)  # FastMACE only: no launcher
    assert list(jobs[1].argv)[list(jobs[1].argv).index("--sides") + 1:][:2] == ["1b", "1a"]
    assert {job.name for job in jobs[3:]} == {"regression-control-1a", "regression-control-1b"}


# --- tools/ab_frontends.py: Simulation, fm.md and fastmace.ase in one process -------------------------------


class _Runner:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __call__(self, steps):
        self.log.append((self.name, steps))


def test_chunk_side_runs_a_block_as_one_call():
    """A front end advances in run(n) calls: the timed block is one run(block) (time_block times it all)."""
    log = []
    side = ab_frontends.ChunkSide(_Runner(log, "sim"), synchronize=lambda: log.append(("sync",)))
    clock = iter(range(100))
    ms = step_time.time_block(side, 5, clock=lambda: float(next(clock)))
    assert log == [("sync",), ("sim", 5), ("sync",)]
    assert ms > 0
    side.finish_block()  # nothing pending: no call
    assert log[-1] == ("sync",)


def test_round_robin_blocks_and_ratios_against_the_first_side():
    clock = step_time.TimingProtocol(warmup=2, block=3, blocks=2)
    log = []
    now = [0.0]

    def make(name, seconds):
        def run(steps):
            now[0] += seconds * steps
            log.append((name, steps))
        return ab_frontends.ChunkSide(run, synchronize=lambda: None)

    sides = {"simulation": make("simulation", 0.010), "md": make("md", 0.0101), "ase": make("ase", 0.0099)}
    result = ab_frontends.run_round_robin(sides, clock, clock=lambda: now[0])
    assert log[:3] == [("simulation", 2), ("md", 2), ("ase", 2)]  # warm-ups
    assert [name for name, _ in log[3:]] == ["simulation", "md", "ase"] * 2
    assert result["ratios_over_first"]["md"] == pytest.approx(1.01)
    assert result["ratios_over_first"]["ase"] == pytest.approx(0.99)
    assert result["order"] == ["simulation", "md", "ase"] * 2


def test_round_robin_needs_two_sides():
    with pytest.raises(ValueError):
        ab_frontends.run_round_robin({"a": None}, step_time.TimingProtocol())


def test_plans_equal_allows_one_ulp_of_friction():
    """T21's conversion from ASE's friction may differ by one ulp (conductor answer, T21)."""
    from fastmace.config import IntegratorConfig
    from fastmace.planner.plan import Plan

    a = Plan(tier="graph", integrator=IntegratorConfig(friction=0.02))
    b = Plan(tier="graph", integrator=IntegratorConfig(friction=np.nextafter(0.02, 1.0)))
    c = Plan(tier="graph", integrator=IntegratorConfig(friction=0.021))
    assert ab_frontends.plans_equal(a, a) and ab_frontends.plans_equal(a, b)
    assert not ab_frontends.plans_equal(a, c)


def test_gate_records_equal_ignores_timings():
    from fastmace.planner import selfcheck

    a = selfcheck.GateRecord(status="passed", check="gate", force_eV_per_A=1e-5, seconds={"production_s": 1.0})
    b = selfcheck.GateRecord(status="passed", check="gate", force_eV_per_A=1e-5, seconds={"production_s": 2.0})
    c = selfcheck.GateRecord(status="passed", check="gate", force_eV_per_A=2e-5)
    assert ab_frontends.gate_records_equal(a, b) and not ab_frontends.gate_records_equal(a, c)


def test_frontends_args():
    args = ab_frontends.parse_args(["--output", "o", "--model", "mh1-r2scan"])
    assert args.sides == ["simulation", "md", "ase"] and args.blocks == 12 and args.structure == "NaCl"
    assert args.check == "skip"
    with pytest.raises(SystemExit):
        ab_frontends.parse_args(["--output", "o", "--sides", "simulation"])


def test_frontend_builders_name_the_task_they_wait_for(monkeypatch):
    import fastmace.quick as quick

    def stub(*args, **kwargs):
        raise NotImplementedError(quick._TASK)

    monkeypatch.setattr(quick, "md", stub)
    with pytest.raises(ab_frontends.FrontEndUnavailable, match="1b-T20"):
        ab_frontends.require_front_end("md")
    monkeypatch.delitem(sys.modules, "fastmace.ase.md", raising=False)
    if importlib.util.find_spec("fastmace.ase.md") is None:
        with pytest.raises(ab_frontends.FrontEndUnavailable, match="1b-T21"):
            ab_frontends.require_front_end("ase")
    ab_frontends.require_front_end("simulation")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_two_simulation_sides_on_the_cpu(tmp_path, monkeypatch):
    """The harness's Simulation builder and round-robin timing on a tiny model (eager CPU engine)."""
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.model import loading
    from fastmace.planner import minimal

    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    loading._LICENCES_SHOWN.add(str(path))
    loaded = loading.load_pristine(str(path), trusted=True)
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    args = ab_frontends.parse_args(["--output", str(tmp_path / "runs"), "--sides", "simulation", "md", "ase"])
    built = {name: ab_frontends.BUILDERS[name](args, atoms, loaded) for name in args.sides}
    sides = {name: ab_frontends.ChunkSide(runner, synchronize=lambda: None)
             for name, (_, runner, _, _) in built.items()}
    result = ab_frontends.run_round_robin(sides, step_time.TimingProtocol(warmup=2, block=3, blocks=2))
    assert set(result["ratios_over_first"]) == {"md", "ase"} and result["order"] == ["simulation", "md", "ase"] * 2
    plans = {name: entry[2] for name, entry in built.items()}
    gates = {name: entry[3] for name, entry in built.items()}
    for name in ("md", "ase"):
        assert ab_frontends.plans_equal(plans["simulation"], plans[name]), name
        assert ab_frontends.gate_records_equal(gates["simulation"], gates[name]), name
    sims = {"simulation": built["simulation"][0], "md": built["md"][0].simulation, "ase": built["ase"][0].simulation}
    steps = {name: sim.committed().step for name, sim in sims.items()}
    # fm.md and the ASE shim ran one step to build; then the same warm-up and blocks
    assert steps == {"simulation": 2 + 2 * 3, "md": 1 + 2 + 2 * 3, "ase": 1 + 2 + 2 * 3}
    np.testing.assert_array_equal(sims["md"].committed().positions, sims["ase"].committed().positions)
    built["ase"][0].close()
    for name in ("simulation", "md"):
        sims[name].close()


# --- tools/library_gate.py: the in-library reduced gate per label -------------------------------------------


def test_library_gate_args_and_record():
    args = library_gate.parse_args(["--output", "o", "--model", "polar1-s", "--structure", "NaCl"])
    assert args.ensembles == ["nvt", "npt"] and args.check == "gate" and args.repeat == [3, 3, 3]
    record = library_gate.case_record("npt", _FakeSimulation())
    assert record["status"] == "passed" and record["ensemble"] == "npt"
    assert record["gate"]["copies"] == "2 rattled (0.05 Å), 2 strained (±2 %, hydrostatic)"
    assert record["plan"].startswith("graph")
    json.dumps(record)


def _gate_runner(cases):
    """A stand-in for subprocess.run: per ensemble, (status in the RESULT line or None for no line, exit code)."""
    def run(argv, stdout, **kwargs):
        status, code = cases[argv[argv.index("case") + 1]]
        if status is not None:
            stdout.write("RESULT " + json.dumps({"ensemble": argv[argv.index("case") + 1], "status": status}) + "\n")
        else:
            stdout.write("FileNotFoundError: checkpoint MACE-POLAR-1-S.model not found\n")
        return types.SimpleNamespace(returncode=code)

    return run


@pytest.mark.parametrize("cases,status", [
    ({"nvt": ("passed", 0), "npt": ("passed", 0)}, 0),
    ({"nvt": ("passed", 0), "npt": ("failed", 0)}, 1),          # a gate that ran and failed
    ({"nvt": ("passed", 0), "npt": (None, 1)}, 2),               # a case that errored (missing checkpoint)
    ({"nvt": ("passed", 3), "npt": ("passed", 0)}, 2),           # a child that crashed after its result line
], ids=["passed", "gate-failed", "case-error", "child-exit"])
def test_library_gate_exit_status_is_red_unless_every_case_passed(tmp_path, cases, status):
    """4-T12: a run whose cases errored (the first Modal try: no checkpoint on the volume) exited 0."""
    argv = ["--output", str(tmp_path / "out"), "--model", "polar1-s", "--structure", "NaCl"]
    meta = library_gate.main(argv, runner=_gate_runner(cases))
    assert library_gate.exit_status(meta) == status
    assert meta["passed"] == (status == 0)
    assert library_gate.exit_status({"cases": {}}) == 2  # nothing ran is not a pass


def test_library_gate_exits_non_zero_without_the_checkpoint(tmp_path):
    """The real tool, end to end on the CPU: no checkpoint under FASTMACE_MODELS, so every case errors."""
    env = dict(os.environ, FASTMACE_MODELS=str(tmp_path / "no-models"), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(TOOLS.parent), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run([sys.executable, str(TOOLS / "library_gate.py"), "--output", str(tmp_path / "out"),
                           "--model", "polar1-s", "--structure", "NaCl", "--ensembles", "nvt"],
                          env=env, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 2, done.stdout + done.stderr
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["cases"]["nvt"]["status"] == "error" and not meta["passed"]
    assert "not found" in (tmp_path / "out" / "nvt.log").read_text()


class _FakeSimulation:
    def __init__(self):
        from fastmace.planner import selfcheck

        self.gate = selfcheck.GateRecord(status="passed", check="gate", copies="2 rattled (0.05 Å), 2 strained "
                                         "(±2 %, hydrostatic)", energy_meV_per_atom=0.001, mode="graph (captured)")
        self.plan = types.SimpleNamespace(describe=lambda: "graph plan")
        self._setup_s = {"plan": 0.1, "gate": 2.0}


def test_library_gate_suite_one_job_per_label():
    machine = suites.MACHINES["murphy"]
    jobs = suites.library_gate_jobs(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    assert len(jobs) == 19
    by_label = {list(job.argv)[list(job.argv).index("--model") + 1]: job for job in jobs}
    structure = {label: list(job.argv)[list(job.argv).index("--structure") + 1] for label, job in by_label.items()}
    assert structure["mh1-r2scan"] == "NaCl" and structure["omol0-xl"] == "NaCl"
    assert structure["off24-medium"].endswith("urea_cod1008787.cif")
    assert all("--check" not in job.argv or "gate" in job.argv for job in jobs)


def test_gate_1b_suite_is_the_1a_layout_with_the_minimal_planner():
    machine = suites.MACHINES["murphy"]
    jobs = suites.gate_1b_jobs(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    assert len({job.name for job in jobs}) == len(jobs)
    for job in jobs:
        argv = list(job.argv)
        assert argv[argv.index("--planner") + 1] == "minimal"
        assert "--structures-root" in argv and "--baseline" not in argv
    subset = suites.gate_1b_jobs(suites.MACHINES["local"], Path("/home/u/fastmace-acceptance/r"),
                                 labels=suites.LOCAL_GATE_SUBSET)
    assert {list(job.argv)[list(job.argv).index("--model") + 1] for job in subset} == {
        "mh1-r2scan", "polar1-s", "omol0-xl", "off24-medium"}


def test_the_1b_suites_are_registered():
    for name in ("gate-1b", "gate-1b-subset", "library-gate", "regression", "frontends", "gpu-suite",
                 "adapter-tests", "compile-checks"):
        assert name in suites.SUITES


def test_workstation_paths_of_the_1b_suites_stay_in_the_nas_home():
    machine = suites.MACHINES["murphy"]
    root = Path("/mnt/homes/zwingel/fastmace-acceptance/r")
    for name in ("gate-1b", "library-gate", "regression", "frontends", "gpu-suite", "adapter-tests", "compile-checks"):
        for job in suites.SUITES[name](machine, root):
            for word in job.argv:
                assert not str(word).startswith(("/tmp", "--basetemp=/tmp")), (name, word)
            assert job.output.startswith("/mnt/homes/zwingel/"), (name, job.output)
    with pytest.raises(suites.LocationError):
        suites.gpu_suite_jobs(machine, Path("/tmp/r"))


def test_adapter_tests_only_read_research_module():
    job = suites.adapter_jobs(suites.MACHINES["local"], Path("/home/u/fastmace-acceptance/r"))[0]
    argv = list(job.argv)
    assert "-p" in argv and argv[argv.index("-p") + 1] == "no:cacheprovider"
    basetemp = next(word for word in argv if word.startswith("--basetemp="))
    assert "ResearchModule" not in basetemp and job.env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert argv[3].endswith("Experiments/Langevin_Thermostat/tests/core/test_fastmace_engine.py")


def test_compile_checks_args_and_suite():
    compile_checks = _load(TOOLS / "compile_checks.py", "_tool_compile_checks_t23")
    args = compile_checks.parse_args(["polar-npt", "--output", "o"])
    assert (args.model, args.chunks, args.chunk, args.repeat) == ("polar1-s", 5, 200, [3, 3, 3])
    args = compile_checks.parse_args(["calc", "--output", "o"])
    assert (args.model, args.calls, args.repeat) == ("mh1-r2scan", 6, [2, 2, 2])
    jobs = suites.compile_checks_jobs(suites.MACHINES["murphy"], Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    environments = {job.name: Path(job.argv[0]).parts[-3] for job in jobs}
    assert environments == {"compile-polar1s-npt-restart": "mace_cueq_oeq", "compile-fm-calc-hybrid": "mace_cueq_oeq",
                            "setup-mace-imported-first": "mace_cueq_oeq",
                            "compile-fm-calc-cueq": "mace_cueq_latest"}
    assert all(job.argv[2] in ("polar-npt", "calc", "mace-first") for job in jobs)
