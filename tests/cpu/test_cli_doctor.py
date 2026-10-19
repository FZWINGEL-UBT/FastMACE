"""1b-T19: ``fastmace`` CLI main (error printing, exit codes) and ``fastmace doctor`` (SPEC §7.5, §14)."""

import dataclasses
import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fastmace import cli
from fastmace.cli import doctor
from fastmace.errors import CacheLocationError, FastMACEError, HybridUnavailableError
from fastmace.kernels import Availability
from fastmace.model import labels

REPOSITORY = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- main and error printing

def _raising_command(error):
    def run(args):
        raise error
    return run


@pytest.fixture
def failing_command(monkeypatch):
    """A ``doctor`` whose run raises the given error."""
    def install(error):
        monkeypatch.setattr(doctor, "run", _raising_command(error))
    return install


def test_command_table_holds_doctor_and_models():
    assert cli.COMMANDS["doctor"] == "fastmace.cli.doctor"
    assert cli.COMMANDS["models"] == "fastmace.cli.models"


def test_fastmace_error_prints_message_shape_alone_and_exits_1(failing_command, capsys):
    error = FastMACEError("the run stopped", why="because of a test.", next_step="Do nothing.",
                          record="run/metadata.json")
    failing_command(error)
    assert cli.main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == str(error)
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_environment_error_names_doctor(failing_command, capsys):
    failing_command(HybridUnavailableError("no OEq"))
    assert cli.main(["doctor"]) == 1
    assert "fastmace doctor" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--debug", "doctor"], ["doctor", "--debug"]])
def test_debug_shows_the_traceback(failing_command, capsys, argv):
    error = FastMACEError("the run stopped", why="because of a test.")
    failing_command(error)
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "_raising_command" in err or "in run" in err
    assert str(error) in err


def test_other_exceptions_keep_their_traceback(failing_command):
    failing_command(KeyError("not ours"))
    with pytest.raises(KeyError):
        cli.main(["doctor"])


def test_no_command_prints_help_and_exits_2(capsys):
    assert cli.main([]) == 2
    assert "doctor" in capsys.readouterr().out


# --------------------------------------------------------------------------- probes mocked

@pytest.fixture
def healthy(monkeypatch, tmp_path):
    """Every probe passes: a GPU, all packages, a toolchain, a fresh cache root; returns the environ."""
    specs = {"torch", "mace", "cuequivariance", "cuequivariance_torch", "cuequivariance_ops_torch",
             "openequivariance", "graph_longrange", "torch_dftd", "triton"}
    monkeypatch.setattr(doctor, "_find_spec", lambda name: name in specs)
    monkeypatch.setattr(doctor, "_version", lambda *names: "1.2.3")
    monkeypatch.setattr(doctor, "_ops_wheels", lambda: {"cuequivariance-ops-torch-cu13": ("0.12.0", []),
                                                        "cuequivariance-ops-cu13": ("0.12.0", [])})
    monkeypatch.setattr(doctor, "_cuda", lambda: {"available": True, "devices": ["NVIDIA RTX 4000 Ada"],
                                                  "cuda": "13.0", "driver": "580.1", "error": None})
    monkeypatch.setattr(doctor, "_triton", lambda: Availability(True, "triton 3.5 is importable"))
    monkeypatch.setattr(doctor, "_nvrtc", lambda: Availability(True, "NVRTC 13.0 loads through cuda.bindings"))
    monkeypatch.setattr(doctor, "_which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(doctor, "_oeq_runtime_problems", lambda environ: [])
    monkeypatch.setattr(doctor, "_on_nfs", lambda path: False)
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache"), "HOME": str(tmp_path / "home")}
    return environ


def _item(name, environ, **options):
    context = doctor.Context(environ=environ, **options)
    (item,) = [check(context) for check in doctor.CHECKS if check.__name__ == f"check_{name}"]
    return item


CHECK_NAMES = ("gpu", "torch", "mace", "cueq", "oeq", "triton", "nvrtc", "graph_longrange", "torch_dftd",
               "cache_root", "file_locks", "models")


def test_checks_in_spec_order():
    assert tuple(check.__name__.removeprefix("check_") for check in doctor.CHECKS) == CHECK_NAMES


@pytest.mark.parametrize("name", CHECK_NAMES)
def test_every_check_passes_when_its_probes_pass(healthy, name):
    item = _item(name, healthy)
    assert item.ok, item
    assert item.fix is None
    assert item.detail


def test_all_pass_exits_0(healthy, capsys):
    assert doctor.run_checks(doctor.Context(environ=healthy)) == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and "[warn]" not in out
    assert "all required checks passed" in out


FAILURES = {
    # name: (what the probe change does, required, a phrase of the fix)
    "gpu": (lambda m: m.setattr(doctor, "_cuda", lambda: {"available": False, "devices": [], "cuda": None,
                                                          "driver": None, "error": "no CUDA device"}),
            False, "nvidia-smi"),
    "torch": (lambda m: m.setattr(doctor, "_find_spec", lambda name: name != "torch"), True, "pip install torch"),
    "mace": (lambda m: m.setattr(doctor, "_find_spec", lambda name: name != "mace"), True, "pip install mace-torch"),
    "cueq": (lambda m: m.setattr(doctor, "_ops_wheels", lambda: {
                 "cuequivariance-ops-torch-cu13": ("0.12.0", []),
                 "cuequivariance-ops-cu13": ("0.12.0", ["cuequivariance_ops/lib/libcue_ops.so"])}),
             True, "cuequivariance-ops-torch-cu13"),
    "oeq": (lambda m: m.setattr(doctor, "_find_spec", lambda name: name != "openequivariance"),
            False, "fastmace[oeq]"),
    "triton": (lambda m: m.setattr(doctor, "_triton", lambda: Availability(False, "triton is not importable")),
               False, "pip install triton"),
    "nvrtc": (lambda m: m.setattr(doctor, "_nvrtc", lambda: Availability(False, "cuda.bindings.nvrtc missing")),
              False, "cuda-bindings"),
    "graph_longrange": (lambda m: m.setattr(doctor, "_find_spec", lambda name: name != "graph_longrange"),
                        False, "fastmace[polar]"),
    "torch_dftd": (lambda m: m.setattr(doctor, "_find_spec", lambda name: name != "torch_dftd"),
                   False, "fastmace[d3]"),
}


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_each_failure_names_its_fix(healthy, monkeypatch, name):
    change, required, fix = FAILURES[name]
    change(monkeypatch)
    item = _item(name, healthy)
    assert not item.ok
    assert item.required is required
    assert fix in item.fix


def test_cueq_missing_package(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_find_spec", lambda name: name != "cuequivariance_torch")
    item = _item("cueq", healthy)
    assert not item.ok and item.required
    assert "pip install cuequivariance-torch" in item.fix


def test_cueq_ops_wheel_missing(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_ops_wheels", lambda: {"cuequivariance-ops-cu13": ("0.12.0", [])})
    item = _item("cueq", healthy)
    assert not item.ok and item.required
    assert "naive kernels" in item.detail


def test_cueq_lack_of_library_named(healthy, monkeypatch):
    FAILURES["cueq"][0](monkeypatch)
    assert "libcue_ops.so" in _item("cueq", healthy).detail


def test_cueq_ops_wheels_read_from_metadata():
    """The real probe, on whatever this environment has: versions and missing libraries, nothing imported."""
    before = set(sys.modules)
    wheels = doctor._ops_wheels()
    assert set(wheels) <= set(doctor.OPS_DISTRIBUTIONS)
    for version, missing in wheels.values():
        assert version and missing == []
    assert not {name for name in set(sys.modules) - before if name.startswith("cuequivariance_ops")}


def test_heavy_modules_never_imported(healthy, monkeypatch):
    """OEq, MACE and the cuEq ops are looked up, never imported (importing them writes files)."""
    imported = []
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: imported.append(name) or real(name, *a))
    for check in doctor.CHECKS:
        check(doctor.Context(environ=healthy))
    assert not [name for name in imported if name.split(".")[0] in
                ("openequivariance", "mace", "cuequivariance_ops_torch", "cuequivariance_ops")]


def test_oeq_missing_toolchain_names_the_document(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_which", lambda name: None)
    item = _item("oeq", healthy)
    assert not item.ok and not item.required
    assert "g++" in item.detail and "ninja" in item.detail
    assert "docs/design/oeq-runtime.md" in item.fix


def test_oeq_runtime_problem_reported(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_oeq_runtime_problems", lambda environ: ["no libcuda.so.1 found"])
    item = _item("oeq", healthy)
    assert not item.ok
    assert "no libcuda.so.1 found" in item.detail


def test_oeq_cached_extension_reported_as_built(healthy):
    built = Path(healthy["FASTMACE_CACHE"]) / "oeq" / "torch-extensions-abc" / "oeq_ext"
    built.mkdir(parents=True)
    (built / "oeq_ext.so").write_bytes(b"")
    item = _item("oeq", healthy)
    assert item.ok
    assert "built" in item.detail and str(built / "oeq_ext.so") in item.detail
    assert "host toolchain" in item.detail


def test_oeq_without_cached_extension(healthy):
    item = _item("oeq", healthy)
    assert item.ok
    assert "not built yet" in item.detail


def test_cache_root_inside_environment(healthy, monkeypatch):
    def refuse(environ=None):
        raise CacheLocationError("cache root /env/cache lies inside the Python environment")
    monkeypatch.setattr(doctor._cache, "cache_root", refuse)
    item = _item("cache_root", healthy)
    assert not item.ok and item.required
    assert "FASTMACE_CACHE" in item.fix


def test_cache_root_not_writable(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_writable", lambda path: False)
    item = _item("cache_root", healthy)
    assert not item.ok and item.required
    assert "FASTMACE_CACHE" in item.fix


def test_cache_root_not_created(healthy):
    item = _item("cache_root", healthy)
    assert item.ok
    assert "not created yet" in item.detail
    assert not Path(healthy["FASTMACE_CACHE"]).exists()


def test_kernel_locks_on_nfs_fail(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_on_nfs", lambda path: True)
    monkeypatch.setattr(doctor, "_runtime_info", lambda: doctor.RuntimeInfo(
        file_locks="kernel", file_lock_source="kernel: torch was imported before fastmace",
        file_lock_switch_late=True, cuda_driver_cache="disabled (on NFS)"))
    item = _item("file_locks", healthy)
    assert not item.ok and item.required
    assert "import fastmace" in item.fix


def test_driver_cache_enabled_on_nfs_fails(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_on_nfs", lambda path: True)
    monkeypatch.setattr(doctor, "_runtime_info", lambda: doctor.RuntimeInfo(
        file_locks="soft", file_lock_source="switched at import", file_lock_switch_late=False,
        cuda_driver_cache="CUDA_CACHE_DISABLE=0 set by the user"))
    item = _item("file_locks", healthy)
    assert not item.ok
    assert "CUDA_CACHE_DISABLE" in item.fix


def test_soft_locks_on_nfs_pass(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_on_nfs", lambda path: True)
    monkeypatch.setattr(doctor, "_runtime_info", lambda: doctor.RuntimeInfo(
        file_locks="soft", file_lock_source="switched at import", file_lock_switch_late=False,
        cuda_driver_cache="disabled (on NFS)"))
    item = _item("file_locks", healthy)
    assert item.ok
    assert "soft" in item.detail and "NFS" in item.detail


# --------------------------------------------------------------------------- model files

def _model_file(directory, label, content=b"weights"):
    directory.mkdir(parents=True, exist_ok=True)
    name = labels.LABELS[label].checkpoint.url.rsplit("/", 1)[-1]
    path = directory / name
    path.write_bytes(content)
    return path


def test_models_wrong_hash_warns(healthy):
    path = _model_file(Path(healthy["FASTMACE_CACHE"]) / "models", "mp0-medium")
    item = _item("models", healthy)
    assert not item.ok and not item.required
    assert path.name in item.detail and "does not match" in item.detail
    assert "download=True" in item.fix


def test_models_matching_hash(healthy, monkeypatch):
    directory = Path(healthy["HOME"]) / "models"
    path = _model_file(directory, "mp0-medium", b"exact")
    entry = labels.LABELS["mp0-medium"].checkpoint
    exact = dataclasses.replace(entry, sha256=hashlib.sha256(b"exact").hexdigest())
    monkeypatch.setattr(doctor, "_expected_files", lambda: {path.name: exact})
    item = _item("models", {**healthy, "FASTMACE_MODELS": str(directory)})
    assert item.ok
    assert path.name in item.detail and "SHA-256 matches" in item.detail


def test_models_no_hash_skips_hashing(healthy, monkeypatch):
    _model_file(Path(healthy["FASTMACE_CACHE"]) / "models", "mp0-medium")
    monkeypatch.setattr(doctor, "_sha256", lambda path: pytest.fail("hashed with --no-hash"))
    item = _item("models", healthy, hash_models=False)
    assert item.ok
    assert "not hashed" in item.detail


def test_models_unknown_file_listed(healthy):
    directory = Path(healthy["FASTMACE_CACHE"]) / "models"
    directory.mkdir(parents=True)
    (directory / "mine.model").write_bytes(b"x")
    item = _item("models", healthy)
    assert item.ok
    assert "mine.model" in item.detail and "not in the label table" in item.detail


def test_models_none_found(healthy):
    item = _item("models", healthy)
    assert item.ok
    assert "no model files" in item.detail


def test_required_failure_exits_1_optional_warns(healthy, monkeypatch, capsys):
    FAILURES["triton"][0](monkeypatch)
    assert doctor.run_checks(doctor.Context(environ=healthy)) == 0
    out = capsys.readouterr().out
    assert "[warn] Triton" in out and "fix: pip install triton" in out
    FAILURES["torch"][0](monkeypatch)
    assert doctor.run_checks(doctor.Context(environ=healthy)) == 1
    out = capsys.readouterr().out
    assert "[FAIL] torch" in out and "1 required check failed" in out


def test_doctor_disables_cuda_driver_cache_before_cuda(healthy, monkeypatch):
    monkeypatch.delenv("CUDA_CACHE_DISABLE", raising=False)
    seen = []
    monkeypatch.setattr(doctor, "_cuda", lambda: seen.append(os.environ.get("CUDA_CACHE_DISABLE")) or {
        "available": False, "devices": [], "cuda": None, "driver": None, "error": "none"})
    for key, value in healthy.items():
        monkeypatch.setenv(key, value)
    assert cli.main(["doctor", "--no-hash"]) == 0
    assert seen == ["1"]


# --------------------------------------------------------------------------- doctor changes nothing

def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_doctor_writes_no_file(tmp_path):
    """The real probes, in a fresh process with a temporary home and cache root."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "keep.txt").write_text("x")
    environ = {**os.environ, "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
               "FASTMACE_CACHE": str(tmp_path / "cache"), "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join([str(REPOSITORY), os.environ.get("PYTHONPATH", "")]),
               "TRITON_CACHE_DIR": str(home / ".triton")}
    environ.pop("FASTMACE_MODELS", None)
    before = _listing(tmp_path)
    result = subprocess.run([sys.executable, "-m", "fastmace.cli", "doctor"], cwd=home, env=environ,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode in (0, 1), result.stderr
    assert "fastmace doctor" in result.stdout
    assert "Traceback" not in result.stderr
    assert _listing(tmp_path) == before
