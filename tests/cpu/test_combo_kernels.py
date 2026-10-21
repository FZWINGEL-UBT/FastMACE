"""The ``combo_kernels`` record: Inductor's pointwise combo kernels for the compiled model (opt-3 OPT3-T04).

- the record is exact, deterministic, opt-in (``never_automatic``), registered last, and only marks the model
  with its Inductor options; its self-check refuses combo kernels without the pointwise-only restriction;
- ``compile_model`` passes a marked model's options to ``torch.compile(options=...)`` (and refuses them with
  another compile mode), an unmarked model compiles as before.
The force field's compile-cache key holds the enabled records, so the two models are never shared. The
GPU test (``tests/gpu/test_combo_kernels_cuda.py``) checks that the option changes no value.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.model import compile as compiling
from fastmace.rewrites import inductor
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, context, tiny  # noqa: E402


def test_the_record_is_exact_opt_in_and_registered_last():
    record = inductor.RECORDS["combo_kernels"]
    assert (record.default, record.exactness, record.deterministic, record.requires) == ("off", "exact", True, ())
    assert reg.RECORD_SOURCES[-1] == "fastmace.rewrites.inductor"
    registry = reg.default_registry()
    assert registry.order()[-1] == "combo_kernels" and registry.config_field("combo_kernels") == "combo_kernels"
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.combo_kernels"] == "never_automatic"
    assert "combo_kernels" in fixed.REWRITE_FIELDS


def test_the_record_marks_the_model_and_nothing_else():
    model, batch = tiny("mh1", backend="cueq")
    registry = reg.default_registry()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    for request in ("off", "auto", "on"):
        out = copy.deepcopy(model)
        asked = {name: "on" for name in PACKAGE} | {"combo_kernels": request}
        resolution = registry.resolve(out, asked, strict=False)
        applied = registry.apply(out, resolution, context(batch), original=model)
        enabled = resolution.decision("combo_kernels").enabled
        assert enabled is (request == "on")
        assert inductor.inductor_options(out) == (inductor.COMBO_OPTIONS if enabled else {})
        if enabled:
            entry = next(entry for entry in applied if entry.name == "combo_kernels")
            assert entry.result.passed
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_the_self_check_refuses_combos_with_reductions():
    model, _ = tiny("mh1", backend="cueq")
    marked = copy.deepcopy(model)
    inductor.set_options(marked, {"combo_kernels": True})
    assert not inductor.RECORDS["combo_kernels"].selfcheck(model, marked, None).passed
    inductor.set_options(marked, inductor.COMBO_OPTIONS)
    assert inductor.RECORDS["combo_kernels"].selfcheck(model, marked, None).passed


def _capture_compile(monkeypatch):
    calls = []

    def fake(model, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake)
    return calls


def test_compile_model_passes_the_options(monkeypatch):
    model, _ = tiny("mh1", backend="cueq")
    calls = _capture_compile(monkeypatch)
    compiling.compile_model(model, "default", accelerated=True,
                            prepare=lambda private: inductor.set_options(private, inductor.COMBO_OPTIONS))
    compiling.compile_model(model, "default", accelerated=True)
    assert calls == [{"options": inductor.COMBO_OPTIONS, "fullgraph": False},
                     {"mode": "default", "fullgraph": False}]
    with pytest.raises(ValueError, match="compile mode"):
        compiling.compile_model(model, "max-autotune", accelerated=True,
                                prepare=lambda private: inductor.set_options(private, inductor.COMBO_OPTIONS))
