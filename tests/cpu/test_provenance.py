"""Minimal provenance (plan task 1a-T16, SPEC §13): today's run.json and metadata.json blocks plus FastMACE's.

The reference files in ``tests/_support/data`` are a research-engine smoke run of 16 October 2026
(MH-1 r2SCAN, plain cuEq, float64, ``cueq_conv_fusion`` on, RTX 5060 Ti):
``run_cueq_conv_fusion_20261016.json`` is its ``run.json`` and
``metadata_cueq_conv_fusion_20261016.json`` the ``metadata.json`` of its one crystal.
"""

import fastmace  # noqa: F401  (before torch: it switches PyTorch's file locks, SPEC §14)

# isort: split
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fastmace import _patches, kernels, provenance, runtime_info
from fastmace.config import EngineConfig, NeighbourConfig, PolarConfig, RewriteConfig

DATA = Path(__file__).resolve().parents[1] / "_support" / "data"
REFERENCE_RUN = json.loads((DATA / "run_cueq_conv_fusion_20261016.json").read_text())
REFERENCE_METADATA = json.loads((DATA / "metadata_cueq_conv_fusion_20261016.json").read_text())
REFERENCE_MODEL = REFERENCE_METADATA["model"]
ROOT = Path(__file__).resolve().parents[2]

# Keys whose value is None when the hardware or package is absent: the CPU suite has no GPU, CPU torch
# has no CUDA version, and the cuEquivariance packages may be missing from the CPU environment.
ABSENT_HERE = {"gpu", "cuda_version", "cuequivariance", "cuequivariance_torch", "cuequivariance_ops_torch"}
# Keys the baseline code at research-engine-baseline-20261017 writes that the 16 October run predates.
LATER_THAN_REFERENCE = {"file_locks"}

ON = "on"
OFF = "off"


def _resolved_options():
    """Options as the reference run resolved them (plain cuEq on MH-1: everything off but the defaults)."""
    rewrites = RewriteConfig(
        factor_species=OFF, fold_cutoff=OFF, density_first=OFF, reassociate_first=OFF, joint_radial_input=OFF,
        fused_folded_backward=OFF, segmented_first_sum=OFF, streaming_conv=OFF, species_skip=OFF,
        fused_radial_hidden=OFF, block_linear=OFF, grouped_linear=OFF, merged_residual=OFF,
        single_graph_strain=ON, polar_uvu=True)
    engine = EngineConfig(backend="cueq", cueq_conv_fusion=ON)
    neighbours = NeighbourConfig(gpu_neighbour_list=ON)
    return dict(rewrites=rewrites, engine=engine, neighbours=neighbours, polar=PolarConfig())


@pytest.fixture(scope="module")
def loaded():
    """A LoadedModel around a tiny random MACE model, shaped like the reference run's."""
    pytest.importorskip("mace")
    from _support.tiny_models import tiny_mace_calculator
    from fastmace.model.loading import LoadedModel

    calculator = tiny_mace_calculator([11, 17])
    return LoadedModel(
        calculator=calculator, path=Path("/models/mace-mh-1.model"), sha256="a" * 64, trust="label",
        checkpoint_source="explicit", label="mh1-r2scan", head="matpes_r2scan",
        heads=("matpes_r2scan", "omat_pbe"), model_type="MACE", family="mh1", functional="r2scan",
        licence="ASL", backend="cueq", dtype="float64", device="cpu", cueq_conv_fusion=True,
        fused_cueq={"device": "cpu", "fused_cueq_modules": 0, "checked": False,
                    "cuequivariance_ops_torch_imports": True},
        embedding_specs=None, charge=None, spin_multiplicity=None)


def _assert_same_types(produced, reference, absent=frozenset()):
    for key, expected in reference.items():
        assert key in produced, f"missing key {key!r}"
        value = produced[key]
        if key in absent and value is None:
            continue
        assert type(value) is type(expected), f"{key}: {type(value).__name__}, reference {type(expected).__name__}"


# ------------------------------------------------------------------------ run.json


def test_run_record_has_every_key_of_the_reference_run_json(tmp_path):
    source = tmp_path / "main.py"
    source.write_text("print('hello')\n")
    record = provenance.run_record(REFERENCE_RUN["protocol"], code_files=[source], root=tmp_path)
    _assert_same_types(record, REFERENCE_RUN)
    assert record["protocol"] == REFERENCE_RUN["protocol"]
    assert record["code_sha256"] == {"main.py": provenance.sha256_file(source)}
    assert record["created_utc"].endswith("+00:00")


def test_run_record_carries_fastmace_additions():
    record = provenance.run_record({"seed": 1})
    assert record["fastmace"]["version"] == fastmace.__version__


def test_run_record_turns_paths_into_strings(tmp_path):
    record = provenance.run_record({"checkpoint": tmp_path / "x.model", "seed": 3})
    assert record["protocol"] == {"checkpoint": str(tmp_path / "x.model"), "seed": 3}
    json.dumps(record)


# ------------------------------------------------------------------------ metadata.json model block


def test_model_block_has_every_key_of_the_reference_model_block(loaded):
    block = provenance.model_block(loaded, **_resolved_options(), compute_stress=False,
                                   auto_options_off={"species_skip": "no MACE-MP-0 or density-family interaction"})
    _assert_same_types(block, REFERENCE_MODEL, absent=ABSENT_HERE)
    for key in LATER_THAN_REFERENCE:
        assert key in block
    json.dumps(block)


def test_model_block_values_follow_the_options_and_the_model(loaded):
    block = provenance.model_block(loaded, **_resolved_options(), compute_stress=False)
    expected = {key: REFERENCE_MODEL[key] for key in (
        "model", "mace_mp_name", "head", "dtype", "dispersion_d3", "enable_cueq", "enable_oeq",
        "equivariance_backend", "compute_stress", "single_graph_strain", "cueq_conv_fusion", "md_backend",
        "factor_species_radial", "fold_radial_cutoff", "density_before_linear", "reassociate_first_interaction",
        "joint_radial_input", "fused_folded_backward", "segmented_first_sum", "gpu_neighbour_list", "block_linear",
        "grouped_linear", "merged_residual", "species_skip", "polar_uvu", "polar_fukui_regularisation",
        "polar_dipole_cap", "streaming_conv", "fused_radial_hidden", "openequivariance")}
    assert {key: block[key] for key in expected} == expected
    assert block["checkpoint"] == str(loaded.path)
    assert block["checkpoint_sha256"] == loaded.sha256
    assert block["heads"] == list(loaded.heads)
    assert block["device"] == "cpu"
    assert block["gpu"] is None  # a CPU device: no GPU name
    assert block["cueq_kernels"] == loaded.fused_cueq
    assert block["file_locks"] == runtime_info().file_locks
    assert set(block["convolution"]) == {"conv_tp", "conv_fusion"}
    assert len(block["convolution"]["conv_fusion"]) == len(loaded.model.interactions)
    assert "auto_options_off" not in block  # as in the baseline: only when something was switched off


def test_model_block_records_dipole_cap_and_polar_uvu_off(loaded):
    options = _resolved_options()
    options["rewrites"] = RewriteConfig(**{**_fields(options["rewrites"]), "polar_uvu": False})
    options["polar"] = PolarConfig(fukui_regularisation=0.5, dipole_cap=(1.0, 2.0))
    block = provenance.model_block(loaded, **options, compute_stress=True)
    assert block["polar_uvu"] == "off"
    assert block["polar_dipole_cap"] == [1.0, 2.0]
    assert block["polar_fukui_regularisation"] == 0.5
    assert block["compute_stress"] is True


def test_model_block_refuses_unresolved_toggles(loaded):
    options = _resolved_options()
    options["rewrites"] = RewriteConfig()  # all "auto"
    with pytest.raises(ValueError, match="auto"):
        provenance.model_block(loaded, **options, compute_stress=False)


def test_model_block_hybrid_records_openequivariance(loaded, monkeypatch):
    import dataclasses

    hybrid = dataclasses.replace(loaded, backend="hybrid", cueq_conv_fusion=None)
    monkeypatch.setattr(provenance, "_distribution_version", lambda *names: f"{names[0]}-version")
    options = _resolved_options()
    options["engine"] = EngineConfig(backend="hybrid", cueq_conv_fusion=OFF)
    block = provenance.model_block(hybrid, **options, compute_stress=False)
    assert block["equivariance_backend"] == "hybrid"
    assert block["enable_oeq"] is True and block["enable_cueq"] is True
    assert block["openequivariance"] == "openequivariance-version"
    assert block["md_backend"] == "hybrid"
    assert set(block["oeq_wrapper"]) == set(provenance.OEQ_ENVIRONMENT_VARIABLES)


def _fields(config):
    import dataclasses

    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


# ------------------------------------------------------------------------ FastMACE's additions


def test_fastmace_block_has_version_commit_patches_locks_and_kernels():
    block = provenance.fastmace_block()
    assert block["version"] == fastmace.__version__
    assert block["patches"] == _patches.provenance()
    info = runtime_info()
    assert block["runtime"] == {"file_locks": info.file_locks, "file_lock_source": info.file_lock_source,
                                "file_lock_switch_late": info.file_lock_switch_late,
                                "cuda_driver_cache": info.cuda_driver_cache}
    assert set(block["kernels"]) == {"triton", "nvrtc", "streaming"}
    for name, probe in (("triton", kernels.triton_available), ("nvrtc", kernels.nvrtc_available),
                        ("streaming", kernels.streaming_available)):
        result = probe()
        assert block["kernels"][name] == {"available": result.available, "reason": result.reason}
    assert {"python", "torch", "numpy"} <= set(block["versions"])
    json.dumps(block)


def test_fastmace_block_records_the_git_commit_of_a_source_checkout():
    if not (ROOT / ".git").exists():
        pytest.skip("not running from a git checkout")
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:  # e.g. a Windows worktree read from WSL: its .git file names a D:/ path
        pytest.skip(f"git cannot read this checkout: {result.stderr.strip()}")
    head = result.stdout.strip()
    block = provenance.fastmace_block()
    assert block["git_commit"] == head
    assert block["git_dirty"] in (True, False, "unknown")  # "unknown": git status ran out of time (1b-T28)


def test_git_commit_is_none_outside_a_checkout(tmp_path):
    assert provenance.git_commit(tmp_path) == (None, None)


def test_patches_in_force_appear_in_the_fastmace_block(monkeypatch):
    record = _patches.PatchRecord(name="test_patch", value=True, targets=("module:attribute",), detail="test")
    monkeypatch.setitem(_patches._RECORDS, "test_patch", record)
    names = [entry["name"] for entry in provenance.fastmace_block()["patches"]]
    assert "test_patch" in names


def test_model_block_and_run_record_carry_the_fastmace_block(loaded):
    block = provenance.model_block(loaded, **_resolved_options(), compute_stress=False)
    assert block["fastmace"]["version"] == fastmace.__version__
    assert "patches" in block["fastmace"]


# ------------------------------------------------------------------------ isolation


_ISOLATION = """
import fastmace
import sys, json
import torch
calls = []
def refuse(*args, **kwargs):
    calls.append("cuda")
    raise RuntimeError("CUDA touched")
for name in ("get_device_name", "get_device_capability", "init", "current_device", "device_count"):
    setattr(torch.cuda, name, refuse)
from fastmace import provenance
everything = provenance.collect(device="cuda:0")
record = provenance.run_record({"seed": 1})
json.dumps(everything); json.dumps(record)
print(json.dumps({"mace": sorted(m for m in sys.modules if m == "mace" or m.startswith("mace.")),
                  "cuda_initialized": torch.cuda.is_initialized(), "calls": calls,
                  "gpu": everything["environment"]["gpu"]}))
"""


def test_collection_without_a_model_imports_no_mace_and_touches_no_cuda():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)}
    result = subprocess.run([sys.executable, "-c", _ISOLATION], capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"mace": [], "cuda_initialized": False, "calls": [], "gpu": None}


def test_collect_records_the_installed_versions():
    """In a MACE environment (mace_cueq_latest): the real versions, as the baseline's model_info reads them."""
    mace = pytest.importorskip("mace")
    import importlib.metadata

    import ase
    import torch

    environment = provenance.collect(device="cpu")["environment"]
    assert environment["mace_torch"] == mace.__version__
    assert environment["torch"] == torch.__version__
    assert environment["ase"] == ase.__version__
    assert environment["cuda_version"] == torch.version.cuda
    try:
        cueq = importlib.metadata.version("cuequivariance")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("cuequivariance is not installed here")
    block = provenance.cueq_versions()
    assert block["cuequivariance"] == cueq
    assert block["cuequivariance_torch"] == importlib.metadata.version("cuequivariance-torch")
    assert block["cuequivariance_ops_torch"] is None or block["cuequivariance_ops_torch"].endswith(("(cu12)", "(cu13)"))


def test_collect_without_a_model_has_environment_and_fastmace_blocks():
    everything = provenance.collect(device="cpu")
    assert set(everything) == {"environment", "fastmace"}
    for key in ("torch", "cuda_version", "gpu", "threads", "mace_torch", "ase", "host"):
        assert key in everything["environment"]
