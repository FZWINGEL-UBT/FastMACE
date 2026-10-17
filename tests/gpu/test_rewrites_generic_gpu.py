"""The generic family on CUDA (plan task 1a-T08): tiny models and released checkpoints.

The stacked generic rewrites (every record that applies, as ``auto`` resolves them) must agree
with the unrewritten model in float64 at the SPEC §6.4 level (relative error 1e-10 in energy,
forces and stress; 1e-4 in float32), on MACE-MP-0 medium (NaCl) and one MACE-OFF model (an
ethanol molecule). Plain cuEquivariance runs in this process; the hybrid backend, where
``generic_streaming`` applies, runs in a fresh interpreter (``FASTMACE_HYBRID_PYTHON``), because
the in-process OEq runtime must be prepared before anything imports MACE. Each model is loaded
twice and the second copy rewritten, so no deep copy of a hybrid model is needed. Edges are
sorted by receiver, as the streaming convolution requires (``sort_receivers``).

Real checkpoints need ``FASTMACE_MODELS``.
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from fastmace.model import labels as table

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable
THRESHOLDS = {"float64": 1e-10, "float32": 1e-4}
MP0, OFF = "mp0-medium", "off23-small"

STACKED = '''
import json
import torch

def stacked(label, backend, dtype):
    """Stacked generic rewrites on a released model against the same model unrewritten (JSON-friendly dict)."""
    import fastmace  # noqa: F401
    from fastmace.model import loading
    from fastmace.rewrites import generic
    from fastmace.rewrites import registry as reg
    original = loading.load(label, backend=backend, dtype=dtype, device="cuda:0")
    rewritten = loading.load(label, backend=backend, dtype=dtype, device="cuda:0")
    if label.startswith("off"):
        from ase.build import molecule
        atoms = molecule("CH3CH2OH")
        atoms.rattle(0.03, seed=3)
        stress = False
    else:
        from _support.structures import nacl
        atoms = nacl(repeat=1, rattle=0.05, seed=17)
        stress = True
    batch = original.calculator._atoms_to_batch(atoms).to("cuda:0")
    order = torch.sort(batch.edge_index[1], stable=True).indices  # receiver-sorted edges
    batch.edge_index, batch.shifts = batch.edge_index[:, order], batch.shifts[order]
    batch.unit_shifts = batch.unit_shifts[order]
    data = {key: value.to(getattr(torch, dtype)) if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in batch.to_dict().items()}
    for model in (original.model, rewritten.model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    registry = reg.Registry(list(generic.RECORDS.values()))
    resolution = registry.resolve(rewritten.model, strict=False)
    context = reg.Context(inputs={generic.NODE_ATTRS: data["node_attrs"]})
    applied = registry.apply(rewritten.model, resolution, context, original=original.model)
    edges = int(data["edge_index"].shape[1])
    for interaction in rewritten.model.interactions:
        if getattr(interaction, "streaming_kernels", None) is not None:
            interaction.streaming_count = torch.tensor(edges, device="cuda:0")
    outputs = []
    for model in (original.model, rewritten.model, rewritten.model):
        out = model(dict(data), training=False, compute_force=True, compute_stress=stress)
        outputs.append({key: out[key].detach().double() for key in ("energy", "forces", "stress")
                        if out.get(key) is not None and (stress or key != "stress")})
    def relative(key):
        a, b = outputs[0][key], outputs[1][key]
        return float((b - a).abs().max()) / float(a.abs().max())
    return {"label": label, "backend": backend, "dtype": dtype, "atoms": len(atoms), "edges": edges,
            "interactions": [type(i).__name__ for i in rewritten.model.interactions],
            "applied": {entry.name: {"detail": entry.detail, "passed": entry.result.passed,
                                     "error": entry.result.max_relative_error, "modules": entry.result.modules}
                        for entry in applied},
            "reasons": resolution.reasons(),
            "relative": {key: relative(key) for key in outputs[0]},
            "repeat_equal": all(torch.equal(outputs[1][k], outputs[2][k]) for k in outputs[1]),
            "energy": float(outputs[0]["energy"].sum())}
'''


def has_checkpoint(label):
    return bool(MODELS) and (Path(MODELS) / table.LABELS[label].checkpoint.file_name).is_file()


def _hybrid_has(module):
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec(module) is not None
    code = f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


def mh1_available():
    try:
        importlib.import_module("fastmace.rewrites.mh1")
    except ImportError:
        return False
    return True


needs_models = pytest.mark.skipif(not MODELS, reason="FASTMACE_MODELS is not set; no real checkpoints")
needs_oeq = pytest.mark.skipif(not _hybrid_has("openequivariance"),
                               reason="OpenEquivariance is not installed (hybrid environment needed)")
needs_mh1 = pytest.mark.skipif(not mh1_available(), reason="fastmace.rewrites.mh1 (plan task 1a-T07) is not merged")


def report(result):
    """Print the agreement for the task report (pytest -s)."""
    print("GENERIC_GPU", json.dumps(result, default=str))


def check(result, labels_applied):
    threshold = THRESHOLDS[result["dtype"]]
    assert sorted(result["applied"]) == sorted(labels_applied), result["reasons"]
    assert all(entry["passed"] for entry in result["applied"].values()), result["applied"]
    for key, error in result["relative"].items():
        assert error <= threshold, (key, error, result)


def expected_records(label, backend):
    names = ["generic_species_skip", "generic_fused_radial_backward"]
    if not label.startswith("off"):
        names.append("generic_reassociate")
    if backend == "hybrid":
        names.append("generic_streaming")
    return names


@needs_mh1
@needs_models
@pytest.mark.parametrize("label", [MP0, OFF])
def test_stacked_generic_rewrites_plain_cueq_float64(label):
    if not has_checkpoint(label):
        pytest.skip(f"{label} not in FASTMACE_MODELS")
    from fastmace import _patches

    namespace = {}
    exec(STACKED, namespace)
    fusion = _patches.record_of("cueq_conv_fusion")
    try:
        result = namespace["stacked"](label, "cueq", "float64")
    finally:
        # loading.load fuses later plain cuEq conversions too; a deep copy of a fused model loses
        # MACE's convolution adapter, which the tiny-model tests in this process do not expect.
        if fusion is None:
            _patches.restore("cueq_conv_fusion")
    report(result)
    check(result, expected_records(label, "cueq"))
    if label.startswith("off"):  # the first block is not generic: nothing touches interactions.0
        assert result["interactions"][0] == "RealAgnosticInteractionBlock"
        assert all("interactions.0" not in entry["detail"] for entry in result["applied"].values())
        assert "RealAgnosticInteractionBlock" in result["reasons"]["generic_reassociate"]


@needs_mh1
@needs_models
@needs_oeq
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("label", [MP0, OFF])
def test_stacked_generic_rewrites_hybrid_in_a_fresh_process(label, dtype):
    if not has_checkpoint(label):
        pytest.skip(f"{label} not in FASTMACE_MODELS")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    code = STACKED + textwrap.dedent(f"""
        print(json.dumps(stacked({label!r}, "hybrid", {dtype!r})))
    """)
    result = subprocess.run([HYBRID_PYTHON, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    result = json.loads(result.stdout.strip().splitlines()[-1])
    report(result)
    check(result, expected_records(label, "hybrid"))


# --------------------------------------------------------------------------- tiny models on CUDA


def tiny_cueq(family, dtype, fused, z_values=(8, 11, 17)):
    """A tiny plain cuEq model on CUDA, its convolution fused or not, whatever patch is in force."""
    from mace.cli import convert_e3nn_cueq

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_calculator, tiny_mace_model
    from fastmace import _patches

    batch = tiny_mace_calculator(list(z_values), **FAMILIES[family])._atoms_to_batch(nacl(repeat=2, rattle=0.05,
                                                                                          seed=1))
    before = _patches.record_of("cueq_conv_fusion")
    _patches.set_cueq_conv_fusion(fused)
    try:
        # looked up after the switch: the patch replaces the module attribute
        model = convert_e3nn_cueq.run(tiny_mace_model(list(z_values), **FAMILIES[family]).to("cuda", dtype),
                                      device="cuda:0", return_model=True)
    finally:
        _patches.set_cueq_conv_fusion(None if before is None else before.value)
    assert all(hasattr(i, "conv_fusion") == fused for i in model.interactions)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = {key: value.to(dtype) if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in batch.to("cuda").to_dict().items()}
    return model, data


def private_copy(model):
    """A deep copy on CUDA with MACE's fused-convolution adapters re-applied.

    A deep copy leaves cuEq fx constants on the CPU, and a fused cuEq convolution (a
    SegmentedPolynomial, copied through __reduce__) loses MACE's four-argument adapter; the
    research engine re-wraps it (``_restore_cueq_conv_fusion``), and so must the force field.
    """
    import copy

    from mace.modules.wrapper_ops import with_cueq_conv_fusion

    duplicate = copy.deepcopy(model).to("cuda")
    copies = dict(duplicate.named_modules())
    for name, module in model.named_modules():
        if "original_forward" in module.__dict__ and "forward" not in copies[name].__dict__:
            with_cueq_conv_fusion(copies[name])
    return duplicate


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("family", ["residual", "density", "off"])
def test_tiny_models_on_cuda(family, dtype, fused):
    pytest.importorskip("cuequivariance_torch")

    from fastmace.rewrites import generic
    from fastmace.rewrites import registry as reg

    model, data = tiny_cueq(family, dtype, fused)
    names = ["generic_species_skip", "generic_fused_radial_backward"]
    if family != "off" and mh1_available():
        names.insert(1, "generic_reassociate")
    registry = reg.Registry(list(generic.RECORDS.values()))
    rewritten = private_copy(model)
    resolution = registry.resolve(rewritten, {name: "on" if name in names else "off" for name in registry.names()})
    applied = registry.apply(rewritten, resolution, reg.Context(inputs={generic.NODE_ATTRS: data["node_attrs"]}),
                             original=model)
    threshold = reg.threshold_for(dtype)
    assert [entry.name for entry in applied] == names
    assert all(entry.result.passed and entry.result.threshold == threshold for entry in applied)
    expected = model(dict(data), training=False, compute_force=True, compute_stress=True)
    got = rewritten(dict(data), training=False, compute_force=True, compute_stress=True)
    for key in ("energy", "forces", "stress"):
        error = float((got[key] - expected[key]).abs().max()) / float(expected[key].abs().max())
        assert error <= threshold, (key, error)
