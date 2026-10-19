"""The software block once per process, and the git commit read from files (plan task 1b-T28).

``provenance.fastmace_block`` costs about 0.13 s from ext4 and about 13 s from a ``/mnt/d`` checkout
(``git status`` over WSL's 9P), and ``Simulation.run`` wrote it at the end of every call (1b-T21b). Now
its static part (versions, git commit, kernel probes, GPU) is built once per process and device, the
commit is read from ``.git`` without starting git, and the dirtiness check runs once per checkout
with a short timeout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fastmace import provenance

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

FASTMACE_BLOCK_KEYS = ["version", "git_commit", "git_dirty", "patches", "runtime", "kernels", "versions", "gpu"]


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch):
    monkeypatch.setattr(provenance, "_SOFTWARE", {})
    monkeypatch.setattr(provenance, "_DIRTY", {})


def _git(directory, *args):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "HOME": str(directory), "PATH": os.environ.get("PATH", "")}
    result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=True, env=env)
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("no git executable")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("a\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "first")
    (root / "a.txt").write_text("b\n")
    _git(root, "commit", "-q", "-am", "second")
    return root


# --------------------------------------------------------------------------- the commit from files


def test_loose_ref(repository):
    assert (repository / ".git" / "refs" / "heads" / "main").is_file()
    assert provenance.head_commit(repository) == _git(repository, "rev-parse", "HEAD")


def test_packed_ref(repository):
    _git(repository, "pack-refs", "--all")
    assert not (repository / ".git" / "refs" / "heads" / "main").exists()
    assert provenance.head_commit(repository) == _git(repository, "rev-parse", "HEAD")


def test_detached_head(repository):
    first = _git(repository, "rev-parse", "HEAD~1")
    _git(repository, "checkout", "-q", "--detach", first)
    assert provenance.head_commit(repository) == first


def test_a_worktrees_git_file(repository, tmp_path):
    tree = tmp_path / "tree"
    _git(repository, "worktree", "add", "-q", "-b", "side", str(tree))
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-q", "-m", "on the side")
    assert (tree / ".git").is_file()
    assert provenance.head_commit(tree) == _git(tree, "rev-parse", "HEAD")
    assert provenance.head_commit(tree) != provenance.head_commit(repository)
    _git(repository, "pack-refs", "--all")
    assert provenance.head_commit(tree) == _git(tree, "rev-parse", "HEAD")


def test_git_commit_starts_git_only_for_the_dirtiness_check_once(repository, monkeypatch):
    head = _git(repository, "rev-parse", "HEAD")
    calls = []
    original = subprocess.run

    def run(argv, *args, **kwargs):
        calls.append(list(argv))
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(provenance.subprocess, "run", run)
    for _ in range(3):
        assert provenance.git_commit(repository) == (head, False)
    assert len(calls) == 1 and "status" in calls[0]


def test_dirty_is_true_for_a_changed_tracked_file(repository):
    (repository / "a.txt").write_text("changed\n")
    assert provenance.git_commit(repository)[1] is True


def test_the_dirtiness_timeout_is_honoured(repository, monkeypatch):
    seen = {}

    def run(argv, *args, timeout=None, **kwargs):
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(provenance.subprocess, "run", run)
    commit, dirty = provenance.git_commit(repository)
    assert commit == provenance.head_commit(repository) and dirty == "unknown"
    assert seen["timeout"] == provenance.GIT_STATUS_TIMEOUT_S <= 2.0


def test_source_commit_without_git(tmp_path):
    (tmp_path / ".source_commit").write_text("43ce026+segmented-generic-experiment\n")
    assert provenance.git_commit(tmp_path) == ("43ce026+segmented-generic-experiment", None)


def test_nothing_outside_a_checkout(tmp_path):
    assert provenance.git_commit(tmp_path) == (None, None)
    (tmp_path / ".git").write_text("gitdir: /no/such/place\n")
    assert provenance.git_commit(tmp_path) == (None, None)


@pytest.mark.skipif(Path("/").anchor != "/" or not Path("/mnt").is_dir(), reason="WSL path mapping only")
def test_a_windows_gitdir_maps_to_mnt_under_wsl(tmp_path):
    assert provenance._host_path("D:/Margraf/FastMACE/.git/worktrees/x", tmp_path) == \
        Path("/mnt/d/Margraf/FastMACE/.git/worktrees/x")
    assert provenance._host_path("../..", tmp_path / "a" / "b") == tmp_path / "a" / "b" / ".." / ".."


def test_this_checkout_matches_git_rev_parse():
    root = Path(provenance.fastmace.__file__).resolve().parent.parent
    if not (root / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:  # a Windows worktree seen from WSL: git cannot read it, the files can
        pytest.skip(f"git cannot read this checkout: {result.stderr.strip()}")
    assert provenance.git_commit()[0] == result.stdout.strip()


# --------------------------------------------------------------------------- once per process


def test_fastmace_block_keys_and_meaning_are_unchanged():
    block = provenance.fastmace_block("cpu")
    assert list(block) == FASTMACE_BLOCK_KEYS
    assert block["gpu"] is None
    commit, dirty = provenance.git_commit()
    assert (block["git_commit"], block["git_dirty"]) == (commit, dirty)


def test_fastmace_block_is_built_once_per_device_and_returns_copies(monkeypatch):
    calls = []
    original = provenance._software

    def software(device):
        calls.append(device)
        return original(device)

    monkeypatch.setattr(provenance, "_software", software)
    first = provenance.fastmace_block("cpu")
    first["versions"]["python"] = "changed"
    first["kernels"].clear()
    second = provenance.fastmace_block("cpu")
    assert second["versions"]["python"] != "changed" and second["kernels"]
    assert calls == ["cpu"]


def test_patches_are_read_at_every_call(monkeypatch):
    from fastmace import _patches

    provenance.fastmace_block("cpu")
    record = _patches.PatchRecord(name="late_patch", value=True, targets=("module:attribute",), detail="test")
    monkeypatch.setitem(_patches._RECORDS, "late_patch", record)
    assert "late_patch" in [entry["name"] for entry in provenance.fastmace_block("cpu")["patches"]]


def test_a_simulation_builds_the_block_once_across_ten_runs(monkeypatch, tmp_path):
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace import integrators
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.io.output import Output
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.simulation import Simulation

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    path = tmp_path / "tiny.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    loaded = loading.load_pristine(str(path), trusted=True)
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    calls = []
    original = provenance._software

    def software(device):
        calls.append(device)
        return original(device)

    monkeypatch.setattr(provenance, "_software", software)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    sim = Simulation(atoms, ForceField(loaded, atoms), seed=1, output=Output(str(tmp_path / "run")),
                     integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01))
    assert calls == ["cpu"]
    assert "provenance" in sim.report.setup_s  # built during setup, shown with the setup time
    for _ in range(10):
        sim.run(steps=1)
    assert calls == ["cpu"]
    import json

    written = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert written["timing"]["steps"] == 10  # the per-call write stays current
