"""``RewriteConfig.sc_generated`` (``rewrites.sc_generated``, SC_GENERATED): off by default, the planner's decision,
and the generated source's structure (branch fastmace/sc0-op, 9 November 2026)."""

from types import SimpleNamespace

import pytest

from fastmace.config import RewriteConfig

H200 = "NVIDIA H200"


def test_off_by_default(monkeypatch):
    from fastmace.planner import fixed
    from fastmace.rewrites import sc_generated

    assert RewriteConfig().sc_generated == "auto"
    assert sc_generated.SC_GENERATED is None and not sc_generated.SC_GENERATED_GPUS
    assert not sc_generated.sc_generated_default(H200)
    assert sc_generated.sc_generated_default(H200, "on") and not sc_generated.sc_generated_default(H200, "off")
    monkeypatch.setattr(sc_generated, "SC_GENERATED", True)  # an A/B side
    assert sc_generated.sc_generated_default(H200, "off")
    assert fixed.AUTO_RULES["RewriteConfig.sc_generated"] == "on_with_tf32x3_gemms_on_listed_gpus_decision_66"


@pytest.mark.parametrize("asked, tf32x3, listed, expected", [
    ("auto", "on", False, "off"), ("auto", "off", True, "off"), ("auto", "on", True, "on"),
    ("on", "off", False, "on"), ("off", "on", True, "off")])
def test_planner_decision(monkeypatch, asked, tf32x3, listed, expected):
    from fastmace.kernels.gpu_models import GPUModels
    from fastmace.planner import fixed
    from fastmace.rewrites import sc_generated

    monkeypatch.setattr(sc_generated, "SC_GENERATED_GPUS", GPUModels((H200,) if listed else ()))
    seen = []
    book = SimpleNamespace(decide=lambda item, choice, reason: seen.append((item, choice)))
    choice = fixed.FixedPlanner._sc_generated(book, asked, tf32x3, SimpleNamespace(name=H200))
    assert choice == expected and seen == [("RewriteConfig.sc_generated", expected)]


def test_precision_passes_the_config(monkeypatch):
    from fastmace.rewrites import precision
    from fastmace.rewrites.registry import Context

    seen = []
    monkeypatch.setattr(precision, "route", lambda model, cueq_projected="auto", sc_generated_config="auto":
                        seen.append(sc_generated_config))
    precision._apply(object(), Context(config=RewriteConfig(sc_generated="on")))
    assert seen == ["on"]


def test_route_skips_models_without_compose_skip():
    import torch

    from fastmace.rewrites import sc_generated

    model = torch.nn.Module()
    model.interactions = torch.nn.ModuleList([torch.nn.Module()])
    model.products = torch.nn.ModuleList([torch.nn.Module()])
    model.interactions[0].first_reassociation = SimpleNamespace(compose_skip=None, sc_generated=True)
    assert sc_generated.route(model, "on") == 0
    assert model.interactions[0].first_reassociation.sc_generated is False


def _terms():
    # weight segments 0..1, inputs 0..2, outputs 0..1: degree 1, 2 and 3 monomials, a repeated index
    return {(0, (0,), 0): 1.0, (1, (0, 1), 0): 0.5, (0, (0, 1, 1), 1): -2.0, (1, (0, 1, 2), 1): 0.25,
            (1, (2, 2), 1): 3.0}


def test_forward_source():
    from fastmace.kernels.cuda.sc_generated import render_forward

    source = render_forward(_terms(), 3, 2, 2, 64)
    assert "__launch_bounds__(64)" in source and "1.0f / (d[a] + 1.0f)" in source
    assert source.count("const float p = ") == 2  # prefixes (0, 1) and (2, 2); (0, 1) once for its two cubics
    assert "y[(long long)a * 128 + 64 + ch] = y1;" in source


def test_backward_source():
    from fastmace.kernels.cuda.sc_generated import render_backward

    source = render_backward(_terms(), 3, 2, 2, 64)
    assert "__shared__ float red[2];" in source and "dd[a] = -s * total;" in source
    assert all(f"dm[(long long)a * 192 + {i * 64} + ch] = dx{i} * s;" in source for i in range(3))
