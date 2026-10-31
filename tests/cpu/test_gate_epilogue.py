"""The ``gate_epilogue`` record (plans/opt-10.md, OPT10-T01): the gate's backward in ``linear_2``'s input-gradient
GEMM, opt-in.

On the CPU the op runs PyTorch's operations: the joined products over the gated rows and their gradients equal
``joined_products`` over the same rows built by autograd; on a tiny MH-1 model the record applies after
``segment_gemm``, self-checks, and leaves energy and forces unchanged (also with the chain's shape rule forced to
the joined path). The fused kernel's GPU exactness is tests/gpu/test_gated_products_cuda.py. The planner keeps the
record opt-in and refuses it in float64, without ``segment_gemm`` and beside ``tf32x3_gemms``.
"""

import copy
import dataclasses
import importlib.util

import pytest
import torch

from fastmace.config import EngineConfig, IntegratorConfig, PrecisionConfig, RewriteConfig
from fastmace.kernels.triton.gated_products import _gated_rows, gated_joined_products, gated_products_input_grads
from fastmace.kernels.triton.segment_gemm import joined_products
from fastmace.planner import fixed
from fastmace.planner.fixed import PlanRefused
from fastmace.rewrites import fusion
from fastmace.rewrites import registry as reg
from fastmace.rewrites.fusion import modules as fm

MUL, DIMS = 6, (1, 3, 5)


def _case(atoms, seed=0, dtype=torch.float64):
    """Scalars, two gated segments (1o, 2e) and their gates, and linear_2's weights (0e, 1o, 2e) in a shuffled
    product order, as a ``gated_joined_products`` argument dict."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(shape, generator=g, dtype=torch.float64).to(dtype)

    scalars = r(atoms, MUL).requires_grad_()
    gated = [r(atoms * d, MUL).requires_grad_() for d in DIMS[1:]]
    gates = r(atoms, 2 * MUL).requires_grad_()
    # Products in output order 2e, 0e, 1o, as linear_2's block order may differ from the gate's.
    inputs, dims = [2, 0, 1], [5, 1, 3]
    weights = [r(MUL, MUL + 2) for _ in inputs]
    offsets, at = [], 0
    for d, w in zip(dims, weights):
        offsets.append(at)
        at += d * w.shape[1]
    return dict(scalars=scalars, gated=gated, gates=gates, gate_offsets=[MUL, 0], weights=weights,
                weights_t=[w.t().contiguous() for w in weights], offsets=offsets, dims=dims, inputs=inputs, width=at)


def _stock(case):
    rows = _gated_rows(case["scalars"], case["gated"], case["gates"], case["gate_offsets"])
    return joined_products([rows[i] for i in case["inputs"]], case["weights"], case["weights_t"], case["offsets"],
                           case["dims"], case["width"])


def _folded(case):
    return gated_joined_products(**case)


def test_on_the_cpu_values_and_gradients_equal_joined_products_over_the_gated_rows():
    case = _case(7)
    leaves = (case["scalars"], case["gates"], *case["gated"])
    grad = torch.randn((7, case["width"]), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    values = []
    for function in (_stock, _folded):
        out = function(case)
        values.append((out.detach(), *torch.autograd.grad(out, leaves, grad)))
    assert torch.equal(values[0][0], values[1][0])  # the forward is the same operations
    for a, b in zip(values[0][1:], values[1][1:]):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


def test_gates_no_segment_reads_get_a_zero_gradient():
    case = _case(4, seed=2)
    gates = torch.cat([case["gates"].detach(), torch.ones(4, 3, dtype=torch.float64)], dim=1).requires_grad_()
    case["gates"] = gates
    out = _folded(case)
    (dgates,) = torch.autograd.grad(out, (gates,), torch.ones_like(out))
    assert torch.equal(dgates[:, 2 * MUL:], torch.zeros(4, 3, dtype=torch.float64))


def test_fake_kernel_gives_the_shapes_of_the_gradients():
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = _case(3)
    real_gated, real_gates = [x.detach() for x in case["gated"]], case["gates"].detach()
    with FakeTensorMode() as mode:
        grad = mode.from_tensor(torch.zeros(3, case["width"], dtype=torch.float64))
        gated = [mode.from_tensor(x) for x in real_gated]
        gates = mode.from_tensor(real_gates)
        weights_t = [mode.from_tensor(w) for w in case["weights_t"]]
        out = gated_products_input_grads(grad, gated, gates, weights_t, case["offsets"], case["dims"],
                                         case["inputs"], case["gate_offsets"], "ieee")
    assert [tuple(t.shape) for t in out] == [(3, MUL), (9, MUL), (15, MUL), (3, 2 * MUL)]


def test_record_is_rounding_deterministic_after_segment_gemm():
    record = fusion.RECORDS["gate_epilogue"]
    # off at the registry: the planner turns it on (decision 66), as grouped_launch, so a registry "auto" resolve
    # never lets it take tf32x3_gemms' place
    assert record.default == "off" and record.exactness == "rounding" and record.deterministic
    assert record.requires == ("segment_gemm",) and record.conflicts == ("tf32x3_gemms",)
    regs = reg.default_registry()
    assert regs.config_field("gate_epilogue") == "gate_epilogue"
    order = regs.order()
    assert order.index("gate_epilogue") > order.index("segment_gemm")
    assert fixed.AUTO_RULES["RewriteConfig.gate_epilogue"] == "on_with_segment_gemm_for_mh1_on_decision_66_gpus"
    assert "gate_epilogue" in fixed.REWRITE_FIELDS
    assert RewriteConfig().gate_epilogue == "auto"


# --------------------------------------------------------------------------- the record on a tiny MH-1 model

cueq = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None or
                          importlib.util.find_spec("mace") is None,
                          reason="mace or cuequivariance_torch is not installed")


def _rewritten(epilogue):
    from test_rewrites_fusion import registry, requests
    from test_rewrites_mh1 import context, tiny

    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    asked = requests("on") | {"segment_gemm": "on", "gate_epilogue": epilogue}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return out, batch, resolution, applied


@cueq
def test_record_applies_selfchecks_and_keeps_energy_and_forces_on_the_cpu(monkeypatch):
    from test_rewrites_mh1 import energy_and_forces

    model, batch, resolution, applied = _rewritten("on")
    assert resolution.decision("gate_epilogue").enabled
    entry = next(entry for entry in applied if entry.name == "gate_epilogue")
    assert entry.result.passed and entry.result.max_relative_error <= 1e-12
    assert all(i.gated_rows.gate_epilogue for i in model.interactions)
    without, _, _, _ = _rewritten("off")
    assert not any(i.gated_rows.gate_epilogue for i in without.interactions)
    energy_and_forces(without, batch)  # throwaway evaluation (docs/development.md)
    want = energy_and_forces(without, batch)
    assert all(torch.equal(a, b) for a, b in zip(energy_and_forces(model, batch), want))  # CPU: stock join
    # With an SM count the chains take the joined path; the CPU op is PyTorch's, so the forward is bitwise and
    # the forces equal segment_gemm's to rounding.
    for m in (model, without):
        for interaction in m.interactions:
            interaction.gated_rows.sms = 10 ** 6
    monkeypatch.setattr("fastmace.kernels.triton.tile_rule.uses_triton", lambda m, sms: True)
    calls = []
    original = fm.GatedRows._gate_epilogue

    def counting(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(fm.GatedRows, "_gate_epilogue", counting)
    want = energy_and_forces(without, batch)
    assert not calls
    got = energy_and_forces(model, batch)
    assert len(calls) == len(model.interactions)
    assert torch.equal(got[0], want[0])
    torch.testing.assert_close(got[1], want[1], rtol=1e-12, atol=1e-12)


@cueq
def test_record_needs_segment_gemm():
    from test_rewrites_fusion import registry, requests
    from test_rewrites_mh1 import tiny

    model, _ = tiny("mh1", backend="cueq")
    asked = requests("on") | {"segment_gemm": "off", "gate_epilogue": "on"}
    assert not registry().resolve(copy.deepcopy(model), asked, strict=False).decision("gate_epilogue").enabled


# --------------------------------------------------------------------------- planner


def _plan(rewrites, dtype=None):
    pytest.importorskip("mace")
    from test_planner import GPU, nacl, planner, traits

    configs = dict(rewrites=rewrites, engine=dataclasses.replace(EngineConfig(), backend="auto"))
    if dtype:
        configs["precision"] = PrecisionConfig(dtype=dtype)
    return planner(**configs).plan(traits("mh1-r2scan"), nacl(), IntegratorConfig(), GPU, model_label="mh1-r2scan")


def test_planner_turns_it_on_for_mh1_on_the_measured_gpu_and_keeps_explicit_values():
    assert _plan(RewriteConfig()).rewrites.gate_epilogue == "on"  # the RTX 5060 Ti, decision 66
    assert _plan(RewriteConfig(gate_epilogue="off")).rewrites.gate_epilogue == "off"
    assert _plan(RewriteConfig(gate_epilogue="on")).rewrites.gate_epilogue == "on"
    assert _plan(RewriteConfig(), dtype="float64").rewrites.gate_epilogue == "off"  # auto: never refused


def test_planner_refuses_it_in_float64_without_segment_gemm_and_with_tf32x3():
    with pytest.raises(PlanRefused, match="gate_epilogue.*float32"):
        _plan(RewriteConfig(gate_epilogue="on"), dtype="float64")
    with pytest.raises(PlanRefused, match="gate_epilogue.*segment_gemm"):
        _plan(RewriteConfig(gate_epilogue="on", segment_gemm="off"))
    with pytest.raises(PlanRefused, match="gate_epilogue and tf32x3_gemms"):
        _plan(RewriteConfig(gate_epilogue="on", tf32x3_gemms="on"))
