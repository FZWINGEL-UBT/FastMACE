"""Which models the w' backward switches of 12 November 2026 reach (fastmace/wprime-off-polar).

SKIP_OUTSIDE + HOIST_DX + LATE_PULLBACK (H200) and SKIP_OUTSIDE + HOIST_DX + WEIGHTED_VECTOR (A100 family) are on by
GPU set, measured on OMAT-0 medium only. results/wprime-off-polar-20261112: no MACE-OFF plan runs the w' kernel on
any GPU, no POLAR-1 plan runs it on the H200, and on the A100 family only POLAR-1-M does, through MH-1's class
(``WPrime(generic=False)``: no TABLE_W). There SKIP_OUTSIDE, HOIST_DX and LATE_PULLBACK fall back to off by
themselves; WEIGHTED_VECTOR applies, measured no gain there and is on POLAR-1-M's per-model off-list.
"""

from types import SimpleNamespace

import pytest
import torch

from fastmace.config import IntegratorConfig
from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.oeq_streaming import wprime_pullback
from fastmace.planner.plan import Hardware
from fastmace.model.labels import MACE_OFF_LABELS

pytest.importorskip("mace")

from test_planner import nacl, planner, traits  # noqa: E402
from test_wprime_backward import _quadrupole  # noqa: E402

H200 = Hardware("NVIDIA H200", (9, 0), 141 * 2**30)
A100S = (Hardware("NVIDIA A100-SXM4-80GB", (8, 0), 80 * 2**30), Hardware("NVIDIA A100 80GB PCIe", (8, 0), 80 * 2**30),
         Hardware("NVIDIA A100-SXM4-40GB", (8, 0), 40 * 2**30))
POLAR = ("polar1-s", "polar1-m", "polar1-l")


def _wprime(label, hardware):
    plan = planner().plan(traits(label), nacl(), IntegratorConfig(), hardware, model_label=label)
    return plan.rewrites.wprime_backward


def test_no_mace_off_or_polar_plan_runs_wprime_on_the_h200():
    """The H200's LATE_PULLBACK / SKIP_OUTSIDE / HOIST_DX were measured on OMAT-0 medium only: if an OFF or POLAR
    plan gains the w' kernel on the H200, measure them there first (MD-step A/B both orders + 3x float32 gate)."""
    for label in (*MACE_OFF_LABELS, *POLAR):
        assert _wprime(label, H200) == "off", label


def test_on_the_a100_family_only_polar1_m_runs_wprime_of_off_and_polar():
    for hardware in A100S:
        for label in (*MACE_OFF_LABELS, *POLAR):
            assert _wprime(label, hardware) == ("on" if label == "polar1-m" else "off"), (label, hardware.name)


def test_late_pullback_is_unsupported_without_block_message():
    """MH-1's class has no TABLE_W, so no BLOCK_MESSAGE layout: ``supported`` is False and the kernel keeps its
    spherical form."""
    assert not wprime_pullback.supported(_quadrupole(), "float", None)


class _Compiled(Exception):
    pass


def _decided(monkeypatch, name, capability, label="polar1-m", **constants):
    """A ``WPrimeBackward`` built as POLAR-1-M's (``WPrime(generic=False)``: no TABLE_W, HORNER, FOLD_ALPHA) on
    ``name``, stopped at its compile; the switches it decided."""
    pytest.importorskip("cuda.bindings.driver")
    from fastmace.kernels.oeq_streaming import runtime

    for key, value in constants.items():
        monkeypatch.setattr(wp, key, value)
    monkeypatch.setattr(wp, "_device_name", lambda device: name)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    captured = {}

    def compile_cubin(source, cap, options=()):
        captured["source"] = source
        raise _Compiled

    monkeypatch.setattr(runtime, "compile_cubin", compile_cubin)
    schedule = _quadrupole()
    kernels = SimpleNamespace(source="using scalar_t = float;", schedule=schedule, num_threads=128, cohort=32,
                              shared_memory=0, output_dim=schedule.output_dim)
    kernel = wp.WPrimeBackward.__new__(wp.WPrimeBackward)
    with pytest.raises(_Compiled):
        wp.WPrimeBackward.__init__(kernel, kernels, 6, torch.device("cuda", 0), vector=wp.SLOPE_VECTOR,
                                   table_w=False, model_label=label)
    return kernel, captured["source"]


@pytest.mark.parametrize("name,capability", [("NVIDIA H200", (9, 0)), ("NVIDIA A100-SXM4-80GB", (8, 0)),
                                             ("NVIDIA A100 80GB PCIe", (8, 0))])
def test_mh1_class_kernel_falls_back_from_the_table_w_switches(monkeypatch, name, capability):
    """Defaults, and SKIP_OUTSIDE / LATE_PULLBACK forced on: both need TABLE_W, HOIST_DX by default needs
    SKIP_OUTSIDE, so all three stay off and the source is the one without them."""
    default, source = _decided(monkeypatch, name, capability)
    assert not default.table_w and not default.skip_outside and not default.hoist_dx and not default.late_pullback
    assert "SKIP_OUTSIDE" not in source and "dx_lane" not in source
    forced, forced_source = _decided(monkeypatch, name, capability, SKIP_OUTSIDE=True, LATE_PULLBACK=True)
    assert not forced.skip_outside and not forced.hoist_dx and not forced.late_pullback
    assert forced_source == source


def test_mh1_class_kernel_takes_weighted_vector_on_the_a100_except_polar1_m(monkeypatch):
    """WEIGHTED_VECTOR needs FACTORED only, so it reaches MH-1-class kernels on the A100 family; POLAR-1-M is on its
    per-model off-list (Alex: NaCl 0.9932 / 1.0041, NbCl5 0.9987 / 0.9986 with b = off; no gain)."""
    # _quadrupole's J = 5 path has a scalar input, which SCALAR_PATHS takes first: off here, so the weighted form
    # shows in the source (POLAR-1-M's weighted paths have 1o inputs)
    assert set(wp.WEIGHTED_VECTOR_MODEL_OFF_GPUS) == {"polar1-m"}
    for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):
        polar, polar_source = _decided(monkeypatch, name, (8, 0), SCALAR_PATHS=False)
        assert polar.factored and not polar.weighted_vector and not polar.stage_dout  # both model off-lists
        assert "WEIGHTED_VECTOR" not in polar_source
        other, other_source = _decided(monkeypatch, name, (8, 0), label="mh1-r2scan", SCALAR_PATHS=False)
        assert other.weighted_vector and "WEIGHTED_VECTOR" in other_source
        forced, _ = _decided(monkeypatch, name, (8, 0), WEIGHTED_VECTOR=True, SCALAR_PATHS=False)
        assert forced.weighted_vector  # an A/B side's True wins over the off-list
    h200, h200_source = _decided(monkeypatch, "NVIDIA H200", (9, 0), label="mh1-r2scan", SCALAR_PATHS=False)
    assert not h200.weighted_vector and "WEIGHTED_VECTOR" not in h200_source


def test_weighted_vector_model_off_list(monkeypatch):
    monkeypatch.setattr(wp, "WEIGHTED_VECTOR", None)
    for name, on in (("NVIDIA A100-SXM4-80GB", True), ("NVIDIA A100-SXM4-40GB", True), ("NVIDIA H200", False)):
        monkeypatch.setattr(wp, "_device_name", lambda device, name=name: name)
        assert wp.weighted_vector_default(None) is on and wp.weighted_vector_default(None, "omat0-medium") is on
        assert not wp.weighted_vector_default(None, "polar1-m"), name
        assert not wp.weighted_vector_on(None, True, False, "polar1-m")
