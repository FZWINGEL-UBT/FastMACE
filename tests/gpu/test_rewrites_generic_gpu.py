"""The generic family on CUDA (plan task 1a-T08): tiny models and released checkpoints.

The stacked generic rewrites (every record that applies, as ``auto`` resolves them) must agree
with the unrewritten model in float64 at the SPEC §6.4 level (relative error 1e-10 in energy,
forces and stress; 1e-4 in float32), on MACE-MP-0 medium (NaCl) and one MACE-OFF model (an
ethanol molecule). Plain cuEquivariance runs in this process; the hybrid backend, where
``generic_streaming`` applies, runs in a fresh interpreter (``FASTMACE_HYBRID_PYTHON``), because
the in-process OEq runtime must be prepared before anything imports MACE. Each model is loaded
twice and the second copy rewritten, so no deep copy of a hybrid model is needed. Edges are
sorted by receiver, as the streaming convolution requires (``sort_receivers``).

``generic_chunked_conv_backward`` (the streaming convolution on MH-1's folded path with its chunked backward) is
also checked with every cell forced into chunks of 37 edges, on MACE-OMAT-0 medium too
(docs/validation/omat0-capacity.md).

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
MP0, OFF, OMAT = "mp0-medium", "off23-small", "omat0-medium"

STACKED = '''
import json
import torch

def stacked(label, backend, dtype, force_chunks=False, recompute=False):
    """Stacked generic rewrites on a released model against the same model unrewritten (JSON-friendly dict).
    ``force_chunks``: the chunked streaming backward runs in chunks of 37 edges whatever the cell's size.
    ``recompute``: generic_recompute_conv_weights on too (float32)."""
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
    resolution = registry.resolve(rewritten.model, {"generic_recompute_conv_weights": "on"} if recompute else None,
                                  strict=False)
    context = reg.Context(inputs=reg.structure_inputs(data))
    applied = registry.apply(rewritten.model, resolution, context, original=original.model)
    edges = int(data["edge_index"].shape[1])
    chunked = []
    for interaction in rewritten.model.interactions:
        if getattr(interaction, "streaming_kernels", None) is not None:
            interaction.streaming_count = torch.tensor(edges, device="cuda:0")
        if force_chunks and getattr(interaction, "chunked_backward_edges", 0):
            interaction.chunked_backward_edges = 37
            chunked.append(type(interaction).__name__)
    if force_chunks:
        from fastmace.kernels.oeq_streaming import runtime
        runtime.chunks_the_backward = lambda *args: True
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
            "forced_chunks": chunked,
            "energy": float(outputs[0]["energy"].sum())}
'''


TABLES = '''
import json
import torch

def tables(label):
    """generic_recompute_first_table (float32, with the radial tables and generic_recompute_conv_weights, every cell
    in chunks of 37 edges): the energy, forces and stress with the tables' outputs evaluated again in the backward,
    the input gradient consumed per chunk or whole, against the same model keeping the outputs."""
    import fastmace  # noqa: F401
    from fastmace.kernels.oeq_streaming import runtime
    from fastmace.model import loading
    from fastmace.rewrites import generic, radial
    from fastmace.rewrites import registry as reg
    from _support.structures import nacl
    original = loading.load(label, backend="hybrid", dtype="float32", device="cuda:0")
    loaded = loading.load(label, backend="hybrid", dtype="float32", device="cuda:0")
    model = loaded.model
    batch = loaded.calculator._atoms_to_batch(nacl(repeat=2, rattle=0.05, seed=17)).to("cuda:0")
    order = torch.sort(batch.edge_index[1], stable=True).indices
    batch.edge_index, batch.shifts, batch.unit_shifts = (batch.edge_index[:, order], batch.shifts[order],
                                                         batch.unit_shifts[order])
    data = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in batch.to_dict().items()}
    for parameter in [*model.parameters(), *original.model.parameters()]:
        parameter.requires_grad_(False)
    names = ("generic_radial_tables", "generic_recompute_first_table")
    registry = reg.Registry([*generic.RECORDS.values(), *(radial.RECORDS[name] for name in names)])
    asked = {name: "on" for name in ("generic_recompute_conv_weights", *names)}
    resolution = registry.resolve(model, asked, strict=False)
    applied = registry.apply(model, resolution, reg.Context(inputs=reg.structure_inputs(data)),
                             original=original.model)
    edges = int(data["edge_index"].shape[1])
    for interaction in model.interactions:
        if getattr(interaction, "streaming_kernels", None) is not None:
            interaction.streaming_count = torch.tensor(edges, device="cuda:0")
        if getattr(interaction, "chunked_backward_edges", 0):
            interaction.chunked_backward_edges = 37
    runtime.chunks_the_backward = lambda *args: True

    def run(recompute, per_chunk):
        for interaction in model.interactions:
            if hasattr(interaction, "recompute_table"):
                interaction.recompute_table = recompute
        runtime.TABLE_GRADIENT_PER_CHUNK = per_chunk
        out = model(dict(data), training=False, compute_force=True, compute_stress=True)
        return {key: out[key].detach().double() for key in ("energy", "forces", "stress")}

    kept = run(False, True)
    again = run(False, True)
    compared = {}
    for per_chunk in (False, True):
        got = run(True, per_chunk)
        compared["per_chunk" if per_chunk else "whole"] = {
            key: {"relative": float((got[key] - kept[key]).abs().max()) / float(kept[key].abs().max()),
                  "equal": bool(torch.equal(got[key], kept[key]))} for key in kept}
    return {"label": label, "edges": edges,
            "applied": {entry.name: {"detail": entry.detail, "passed": entry.result.passed,
                                     "error": entry.result.max_relative_error} for entry in applied},
            "kept_repeat_equal": all(torch.equal(kept[key], again[key]) for key in kept),
            "compared": compared}
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
        importlib.import_module("fastmace.rewrites.mh1.modules")
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
        names += ["generic_reassociate", "generic_segmented_first_sum"]  # SEGMENTED_FIRST_SUM acts on it (1b-T11)
    if backend == "hybrid":
        names += ["generic_streaming", "generic_chunked_conv_backward"]
    return names


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
@needs_mh1
@needs_models
@needs_oeq
@pytest.mark.parametrize("label", [OMAT, MP0])
def test_generic_recompute_conv_weights_in_forced_chunks(label):
    """generic_recompute_conv_weights (float32): no stored per-edge weights, every cell in chunks of 37 edges; the
    forces as the unrewritten model's at the float32 threshold."""
    if not has_checkpoint(label):
        pytest.skip(f"{label} not in FASTMACE_MODELS")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)
    code = STACKED + textwrap.dedent(f"""
        print(json.dumps(stacked({label!r}, "hybrid", "float32", force_chunks=True, recompute=True)))
    """)
    result = subprocess.run([HYBRID_PYTHON, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    result = json.loads(result.stdout.strip().splitlines()[-1])
    report(result)
    check(result, expected_records(label, "hybrid") + ["generic_recompute_conv_weights"])
    assert result["forced_chunks"], result


@pytest.mark.slow
@needs_mh1
@needs_models
@needs_oeq
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("label", [OMAT, MP0])
def test_generic_chunked_backward_in_forced_chunks(label, dtype):
    """generic_chunked_conv_backward with every cell in chunks of 37 edges: the forces as the unrewritten model's."""
    if not has_checkpoint(label):
        pytest.skip(f"{label} not in FASTMACE_MODELS")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)
    code = STACKED + textwrap.dedent(f"""
        print(json.dumps(stacked({label!r}, "hybrid", {dtype!r}, force_chunks=True)))
    """)
    result = subprocess.run([HYBRID_PYTHON, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    result = json.loads(result.stdout.strip().splitlines()[-1])
    report(result)
    check(result, expected_records(label, "hybrid"))
    assert result["forced_chunks"], result


@pytest.mark.slow
@needs_mh1
@needs_models
@needs_oeq
@pytest.mark.parametrize("label", [OMAT, MP0])
def test_generic_recompute_first_table_against_the_kept_tables(label):
    """generic_recompute_first_table is exact: with the input gradient consumed whole or per chunk, the energy, forces
    and stress equal those of the kept table outputs wherever the kept path repeats itself bitwise (the sender
    gradients' atomic order aside), and stay within 1e-6 relative otherwise."""
    if not has_checkpoint(label):
        pytest.skip(f"{label} not in FASTMACE_MODELS")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)
    code = TABLES + textwrap.dedent(f"""
        print(json.dumps(tables({label!r})))
    """)
    result = subprocess.run([HYBRID_PYTHON, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    result = json.loads(result.stdout.strip().splitlines()[-1])
    report(result)
    assert "generic_recompute_first_table" in result["applied"], result
    assert all(entry["passed"] for entry in result["applied"].values()), result["applied"]
    for mode, keys in result["compared"].items():
        for key, entry in keys.items():
            assert entry["relative"] <= 1e-6, (mode, key, result)
            if result["kept_repeat_equal"]:
                assert entry["equal"], (mode, key, result)


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
    applied = registry.apply(rewritten, resolution, reg.Context(inputs=reg.structure_inputs(data)),
                             original=model)
    threshold = reg.threshold_for(dtype)
    assert [entry.name for entry in applied] == names
    assert all(entry.result.passed and entry.result.threshold == threshold for entry in applied)
    expected = model(dict(data), training=False, compute_force=True, compute_stress=True)
    got = rewritten(dict(data), training=False, compute_force=True, compute_stress=True)
    for key in ("energy", "forces", "stress"):
        error = float((got[key] - expected[key]).abs().max()) / float(expected[key].abs().max())
        assert error <= threshold, (key, error)
