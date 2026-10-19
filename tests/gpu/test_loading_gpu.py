"""Model loading on the GPU with real checkpoints (plan task 1a-T05).

Needs ``FASTMACE_MODELS`` pointing at a directory with the released checkpoints
(workstations: the research engine's model store; pass it at run time). The
hybrid tests run in a fresh interpreter, because the in-process OEq runtime must
be prepared before anything imports MACE, and this suite imports MACE early.
They need the hybrid environment (OpenEquivariance installed): this one, or the
interpreter named by ``FASTMACE_HYBRID_PYTHON`` (for a hybrid environment without
pytest, such as the NAS ``mace_cueq_oeq``).
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from _support.structures import nacl
from fastmace.model import labels as table

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")

needs_models = pytest.mark.skipif(not MODELS, reason="FASTMACE_MODELS is not set; no real checkpoints")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable


def _hybrid_has(module):
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec(module) is not None
    code = f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


needs_oeq = pytest.mark.skipif(not _hybrid_has("openequivariance"),
                               reason="OpenEquivariance is not installed (hybrid environment needed)")


def has_checkpoint(label):
    return bool(MODELS) and (Path(MODELS) / table.LABELS[label].checkpoint.file_name).is_file()


def run_fresh(code, timeout=900):
    """Run code in a fresh hybrid interpreter (HYBRID_PYTHON); returns the JSON its last stdout line prints."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    result = subprocess.run([HYBRID_PYTHON, "-c", textwrap.dedent(code)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def evaluate(calculator, repeat=2, rattle=0.05, seed=17):
    atoms = nacl(repeat=repeat, rattle=rattle, seed=seed)
    atoms.calc = calculator
    return atoms.get_potential_energy(), atoms.get_forces()


@pytest.mark.slow
@needs_models
@pytest.mark.skipif(not has_checkpoint("mh1-r2scan"), reason="MH-1 checkpoint not in FASTMACE_MODELS")
def test_mh1_plain_cueq_float64():
    from fastmace.model import loading
    loaded = loading.load("mh1-r2scan", backend="cueq", dtype="float64", device="cuda:0")
    assert loaded.trust == "label" and loaded.sha256 == table.LABELS["mh1-r2scan"].checkpoint.sha256
    assert loaded.checkpoint_source == "FASTMACE_MODELS" and loaded.head == "matpes_r2scan"
    assert loaded.cueq_conv_fusion is True
    assert loaded.fused_cueq["checked"] and not loaded.fused_cueq["naive"]
    assert all(getattr(i, "conv_fusion", False) for i in loaded.model.interactions)
    assert [r.name for r in loaded.patches] == ["cueq_conv_fusion"]
    from fastmace import _patches
    assert _patches.record_of("cueq_conv_fusion") is None  # the rule acted on this conversion only
    energy, forces = evaluate(loaded.calculator)
    assert np.isfinite(energy) and np.isfinite(forces).all() and forces.shape == (64, 3)


@pytest.mark.slow
@needs_models
@needs_oeq
@pytest.mark.skipif(not has_checkpoint("mh1-r2scan"), reason="MH-1 checkpoint not in FASTMACE_MODELS")
def test_mh1_hybrid_float32_in_a_fresh_process():
    out = run_fresh("""
        import json, sys
        import fastmace
        from fastmace.model import loading
        assert "mace" not in sys.modules
        loaded = loading.load("mh1", head="matpes_r2scan", backend="hybrid", dtype="float32", device="cuda:0")
        from _support.structures import nacl
        atoms = nacl(repeat=2, rattle=0.05, seed=17)
        atoms.calc = loaded.calculator
        e, f = atoms.get_potential_energy(), atoms.get_forces()
        import openequivariance as oeq
        oeq_modules = sum(type(m).__module__.startswith("openequivariance") for m in loaded.model.modules())
        print(json.dumps({"label": loaded.label, "trust": loaded.trust, "fused": loaded.fused_cueq["checked"],
                          "patches": [r.name for r in loaded.patches], "built": oeq.BUILT_EXTENSION,
                          "oeq_modules": oeq_modules, "energy": e, "finite": bool(abs(f).max() < 1e3)}))
    """)
    assert out["label"] == "mh1-r2scan" and out["trust"] == "label" and out["built"] is True
    assert out["patches"] == ["oeq_runtime"] and out["oeq_modules"] > 0 and out["fused"] and out["finite"]


@needs_oeq
@pytest.mark.skipif(not _hybrid_has("graph_longrange"), reason="PolarMACE needs graph_longrange")
def test_hybrid_conversion_keeps_the_last_layer_irreps_of_a_tiny_polar_model(tmp_path):
    out = run_fresh(f"""
        import json
        import fastmace
        from fastmace.model import loading
        from fastmace import _patches
        import os
        _patches.prepare_oeq_runtime(os.path.join({str(tmp_path)!r}, "oeq"))
        import torch
        from _support.tiny_models import FAMILIES, tiny_mace_model
        model = tiny_mace_model([11, 17], **FAMILIES["polar"])
        path = os.path.join({str(tmp_path)!r}, "tiny-polar.model")
        torch.save(model, path)
        from mace.calculators import MACECalculator
        try:  # MACE 0.3.16's converter on its own: keep_last_layer_irreps=False for every model
            MACECalculator(model_paths=path, device="cuda:0", default_dtype="float64", model_type="PolarMACE",
                           enable_cueq=True, enable_oeq=True)
            stock = "converted"
        except Exception as error:
            stock = type(error).__name__ + ": " + str(error)[:200]
        with loading.hybrid_keeps_last_layer_irreps():
            calc = MACECalculator(model_paths=path, device="cuda:0", default_dtype="float64",
                                  model_type="PolarMACE", enable_cueq=True, enable_oeq=True)
        print(json.dumps({{"stock": stock, "fixed": type(calc.models[0]).__name__}}))
    """)
    # leto, 17 October 2026: "einsum(): subscript a has size 7 for operand 1 which does not broadcast ...".
    assert out["stock"] != "converted"
    assert out["fixed"] == "PolarMACE"


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("label", sorted(table.LABELS))
def test_every_label_loads_with_its_hash_and_licence(label):
    if not has_checkpoint(label):
        pytest.skip(f"{table.LABELS[label].checkpoint.file_name} not in FASTMACE_MODELS")
    from fastmace.model import loading
    entry = table.LABELS[label]
    loaded = loading.load(label, backend="cueq", dtype="float64", device="cuda:0")
    assert (loaded.sha256, loaded.trust, loaded.licence) == (entry.checkpoint.sha256, "label", entry.licence)
    assert loaded.model_type == entry.model_type and type(loaded.model).__name__ in ("ScaleShiftMACE", "PolarMACE")
    assert loaded.fused_cueq["checked"] and not loaded.fused_cueq["naive"]
    assert (loaded.embedding_specs is not None) == (label == "omol0-xl")
