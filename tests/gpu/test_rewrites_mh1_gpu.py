"""The MH-1 family on CUDA (plan task 1a-T07): Triton kernels, cuEq's GPU convolutions, streaming, MH-1.

- Tiny MH-1-class models converted to cuEquivariance on the GPU, with and without MACE's fused
  convolution, in float32 and float64: the package through the registry, each self-check at its
  SPEC §6.4 threshold, then energy and forces against the unrewritten model. The Triton kernels
  (folded backward, segmented first sum, edge outer sum) run in float32 only.
- The hybrid backend (OEq convolutions) in a fresh interpreter (``FASTMACE_HYBRID_PYTHON``, as the
  loading tests do): the streaming kernels attach to every folded interaction, pass their
  self-check and keep the model's energy and forces.
- MH-1 itself (``FASTMACE_MODELS``, head ``matpes_r2scan``): plain cuEq in float64 on NaCl 2x2x2 at
  the per-model level, and the hybrid in float32 with streaming.

Two models are loaded instead of deep-copying one: a deep copy of a fused cuEq or OEq model loses
MACE's convolution adapters (plan task 1a-T11 restores them).
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from fastmace.rewrites import linear, mh1
from fastmace.rewrites import registry as reg

pytest.importorskip("cuequivariance_torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable
Z_VALUES = [8, 11, 17, 26]
MH1 = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward")
FIRST = ("reassociate_first", "segmented_first_sum")
LINEAR = ("block_linear", "grouped_linear", "merged_residual")


def _hybrid_has(module):
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec(module) is not None
    code = f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


needs_oeq = pytest.mark.skipif(not _hybrid_has("openequivariance"),
                               reason="OpenEquivariance is not installed (hybrid environment needed)")


def registry():
    out = reg.Registry()
    for family in (mh1, linear):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def structure():
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.1, seed=8)
    atoms.numbers[3] = 8
    return atoms


def relative(got, expected):
    return float((got - expected).abs().max()) / float(expected.abs().max())


def energy_and_forces(model, data):
    out = model(dict(data), training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()


def tiny_cueq(family, dtype, device):
    """A tiny model converted on the GPU; device "cuda" fuses the convolution (MACE 0.3.16), "cuda:0" does not."""
    from mace.cli.convert_e3nn_cueq import run

    from _support.tiny_models import FAMILIES, tiny_mace_calculator, tiny_mace_model

    batch = tiny_mace_calculator(Z_VALUES, **FAMILIES[family])._atoms_to_batch(structure()).to("cuda")
    model = run(tiny_mace_model(Z_VALUES, **FAMILIES[family]).to("cuda", dtype), device=device, return_model=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = {key: value.to(dtype) if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in batch.to_dict().items()}
    return model, data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
@pytest.mark.parametrize("family", ["mh1", "omol"])
def test_package_on_cueq_gpu_models(family, device, dtype):
    original, data = tiny_cueq(family, dtype, device)
    model, _ = tiny_cueq(family, dtype, device)
    assert all(hasattr(i, "conv_fusion") for i in model.interactions) is (device == "cuda")
    regs = registry()
    resolution = regs.resolve(model, {"fused_radial_hidden": "on"}, strict=False)
    context = reg.Context(inputs=reg.structure_inputs(data))
    applied = regs.apply(model, resolution, context, original=original)
    names = [entry.name for entry in applied]
    threshold = reg.threshold_for(dtype)
    assert names[:5] == list(MH1) and "fused_radial_hidden" in names and names[-3:] == list(LINEAR)
    assert not resolution.decision("streaming_conv").enabled  # plain cuEq: no OEq convolution
    for entry in applied:
        assert entry.result.passed and entry.result.threshold == threshold, (entry.name, entry.result)
    reassociated = set(FIRST) <= set(names)
    print(json.dumps({"family": family, "device": device, "dtype": str(dtype), "reassociated": reassociated,
                      "reason": resolution.decision("reassociate_first").reason,
                      "errors": {e.name: e.result.max_relative_error for e in applied}}))
    energy_and_forces(original, data)
    energy, forces = energy_and_forces(original, data)
    new_energy, new_forces = energy_and_forces(model, data)
    assert relative(new_energy, energy) <= threshold and relative(new_forces, forces) <= threshold


# --------------------------------------------------------------------------- fresh hybrid interpreters


def run_fresh(code, timeout=1200):
    """Run code in a fresh hybrid interpreter (HYBRID_PYTHON); returns the JSON its last stdout line prints."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    result = subprocess.run([HYBRID_PYTHON, "-c", textwrap.dedent(code)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


HYBRID_CHILD = """
import json
import fastmace
from fastmace.model import loading

def load():
    return loading.load({source!r}, {head}backend="hybrid", dtype={dtype!r}, device="cuda:0", trusted=True)

original, loaded = load(), load()
import torch
from fastmace.rewrites import linear, mh1
from fastmace.rewrites import registry as reg
from _support.structures import nacl

atoms = nacl(repeat=2, rattle=0.1, seed=8)
if {oxygen}:
    atoms.numbers[3] = 8
calc = loaded.calculator
data = calc._atoms_to_batch(atoms).to_dict()
data = {{k: v.to(getattr(torch, {dtype!r})) if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in data.items()}}
# The streaming kernels need receiver-sorted edges (workload constraint sort_receivers); MACE's
# neighbour list sorts by sender.
order = torch.sort(data["edge_index"][1], stable=True).indices
data["edge_index"] = data["edge_index"][:, order]
for key in ("shifts", "unit_shifts"):
    data[key] = data[key][order]
for model in (original.model, loaded.model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
regs = reg.Registry()
for family in (mh1, linear):
    for record in family.RECORDS.values():
        regs.register(record)
resolution = regs.resolve(loaded.model, strict=False)
applied = regs.apply(loaded.model, resolution, reg.Context(inputs=reg.structure_inputs(data)),
                     original=original.model)

def run(model):
    out = model(dict(data), training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()

def relative(a, b):
    return float((a - b).abs().max()) / float(b.abs().max())

run(original.model)
energy, forces = run(original.model)
new_energy, new_forces = run(loaded.model)
streaming = [i for i, x in enumerate(loaded.model.interactions) if getattr(x, "streaming_kernels", None) is not None]
edges = int(data["edge_index"].shape[1])
bound = mh1.bind_streaming_count(loaded.model, torch.tensor(edges, device="cuda"))
bound_energy, bound_forces = run(loaded.model)
print(json.dumps({{
    "enabled": list(resolution.enabled), "reasons": resolution.reasons(),
    "checks": {{e.name: [e.result.passed, e.result.max_relative_error, e.result.threshold] for e in applied}},
    "streaming": streaming, "bound": bound, "energy": float(energy), "atoms": len(atoms),
    "energy_error": float((new_energy - energy).abs().max()), "energy_relative": relative(new_energy, energy),
    "force_error": float((new_forces - forces).abs().max()), "force_relative": relative(new_forces, forces),
    # sender gradients are added atomically: repeated evaluations agree to rounding, not bitwise
    "bound_relative": max(relative(bound_energy, new_energy), relative(bound_forces, new_forces)),
}}))
"""


def _check_hybrid(out, dtype):
    threshold = reg.THRESHOLDS[dtype]
    assert {"streaming_conv", *MH1, *LINEAR} <= set(out["enabled"]), out["reasons"]
    for name, (passed, error, limit) in out["checks"].items():
        assert passed and limit == threshold, (name, error, limit)
    assert out["streaming"] == [0, 1] and out["bound"] == 2 and out["bound_relative"] <= threshold
    assert out["energy_relative"] <= threshold and out["force_relative"] <= threshold, out


@needs_oeq
@pytest.mark.parametrize("dtype", ["float32", pytest.param("float64", marks=pytest.mark.slow)])
def test_streaming_attaches_on_a_tiny_hybrid_model(tmp_path, dtype):
    from _support.tiny_models import FAMILIES, tiny_mace_model

    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model(Z_VALUES, **FAMILIES["mh1"]), path)
    out = run_fresh(HYBRID_CHILD.format(source=str(path), head="", dtype=dtype, oxygen=True))
    print(json.dumps(out, indent=1))
    _check_hybrid(out, dtype)


# --------------------------------------------------------------------------- MH-1


def _has_mh1():
    if not MODELS:
        return False
    from fastmace.model import labels

    return (Path(MODELS) / labels.LABELS["mh1-r2scan"].checkpoint.file_name).is_file()


needs_mh1 = pytest.mark.skipif(not _has_mh1(), reason="FASTMACE_MODELS has no MH-1 checkpoint")


@pytest.fixture
def mace_conv_fusion_rule():
    """Restore MACE's own conv-fusion rule after the test: loading.load(backend="cueq") patches it process-wide."""
    yield
    from fastmace import _patches

    _patches.set_cueq_conv_fusion(None)


@pytest.mark.slow
@needs_mh1
def test_mh1_float64_package_agrees_with_the_unrewritten_model(mace_conv_fusion_rule):
    """MH-1 (matpes_r2scan), plain cuEq, float64, NaCl 2x2x2: the stacked package against the unrewritten model."""
    from _support.structures import nacl
    from fastmace.model import loading

    original = loading.load("mh1-r2scan", backend="cueq", dtype="float64", device="cuda:0")
    loaded = loading.load("mh1-r2scan", backend="cueq", dtype="float64", device="cuda:0")
    atoms = nacl(repeat=2, rattle=0.1, seed=8)
    data = loaded.calculator._atoms_to_batch(atoms).to_dict()
    for model in (original.model, loaded.model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    regs = registry()
    resolution = regs.resolve(loaded.model, strict=False)
    applied = regs.apply(loaded.model, resolution, reg.Context(inputs=reg.structure_inputs(data)),
                         original=original.model)
    assert [e.name for e in applied][:5] == list(MH1) and [e.name for e in applied][-3:] == list(LINEAR)
    for entry in applied:
        assert entry.result.passed and entry.result.threshold == 1e-10, (entry.name, entry.result)
    energy_and_forces(original.model, data)
    energy, forces = energy_and_forces(original.model, data)
    new_energy, new_forces = energy_and_forces(loaded.model, data)
    report = {"enabled": list(resolution.enabled), "reasons": resolution.reasons(),
              "checks": {e.name: e.result.max_relative_error for e in applied},
              "energy_eV": float(energy), "energy_error_eV": float((new_energy - energy).abs().max()),
              "energy_relative": relative(new_energy, energy),
              "force_error_eV_per_A": float((new_forces - forces).abs().max()),
              "force_relative": relative(new_forces, forces)}
    print(json.dumps(report, indent=1))
    assert report["energy_relative"] <= 1e-10 and report["force_relative"] <= 1e-10


@pytest.mark.slow
@needs_mh1
@needs_oeq
def test_mh1_hybrid_float32_with_streaming():
    out = run_fresh(HYBRID_CHILD.format(source="mh1-r2scan", head="", dtype="float32", oxygen=False))
    print(json.dumps(out, indent=1))
    _check_hybrid(out, "float32")
