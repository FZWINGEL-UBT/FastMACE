"""The ``tf32x3_gemms`` record (plans/opt-4.md, OPT4-T03 stage 1): every atom GEMM as 3xTF32, opt-in.

On the CPU the op runs PyTorch's product, so with the record on a tiny cuEquivariance model's energy and forces
are bitwise those without it; the GPU accuracy is docs/validation/opt4-T03.md's force gate. The planner turns
"auto" on only for MH-1 and OMAT-0 on the A100-SXM4-80GB (decision 66, docs/validation/cutlass-tf32x3.md), and
refuses it in float64, below compute capability 8.0 and beside ``grouped_launch``.
"""

import copy
import dataclasses
import importlib.util

import pytest
import torch

from fastmace.config import EngineConfig, IntegratorConfig, PrecisionConfig, RewriteConfig
from fastmace.planner import fixed
from fastmace.planner.fixed import PlanRefused
from fastmace.planner.plan import Hardware
from fastmace.rewrites import precision
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")
cueq = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                          reason="cuequivariance_torch is not installed")

from test_planner import GPU, nacl, planner, traits  # noqa: E402
from test_rewrites_mh1 import context, energy_and_forces, tiny  # noqa: E402

# Records that need the GPU stack (OEq) or only change Inductor; off in the CPU comparison.
GPU_ONLY = ("streaming_conv", "generic_streaming", "combo_kernels")


def test_record_is_opt_in_rounding_deterministic_and_last_of_the_module_records():
    record = precision.RECORDS["tf32x3_gemms"]
    assert record.default == "off" and record.exactness == "rounding" and record.deterministic
    assert record.requires == ("block_linear",) and record.conflicts == ("grouped_launch",)
    regs = reg.default_registry()
    assert regs.config_field("tf32x3_gemms") == "tf32x3_gemms"
    order = regs.order()
    species = [name for name in order if name in ("species_tables", "fold_readout_tail", "fold_product_linear")]
    assert species and all(order.index(name) < order.index("tf32x3_gemms") for name in species)
    assert (fixed.AUTO_RULES["RewriteConfig.tf32x3_gemms"]
            == "on_for_mh1_and_omat0_float32_unless_reproducible_on_decision_66_gpus")
    assert "tf32x3_gemms" in fixed.REWRITE_FIELDS
    assert RewriteConfig().tf32x3_gemms == "auto"


def _energy_and_forces(family, tf32x3):
    model, batch = tiny(family, backend="cueq")
    regs = reg.default_registry()
    out = copy.deepcopy(model)
    asked = {name: "off" if name in GPU_ONLY else "auto" for name in regs.names()} | {"tf32x3_gemms": tf32x3}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return out, applied, energy_and_forces(out, batch)


@cueq
@pytest.mark.parametrize("family", ["mh1", "residual"])
def test_on_the_cpu_every_atom_gemm_is_routed_and_bitwise_the_stock_path(family):
    model, applied, (energy, forces) = _energy_and_forces(family, "on")
    entry = next(entry for entry in applied if entry.name == "tf32x3_gemms")
    assert entry.result.passed and entry.detail
    kinds = {type(module).__name__ for module in model.modules() if getattr(module, "tf32x3", False)}
    assert "BlockLinear" in kinds
    if family == "mh1":
        assert {"ResidualBlockLinear", "GatedRows", "FirstInteractionReassociation"} <= kinds
    else:
        assert "SpeciesSkip" in kinds
    unflagged = [name for name, module in model.named_modules()
                 if type(module).__name__ in precision._TYPES and not module.tf32x3]
    assert not unflagged
    _, _, (stock_energy, stock_forces) = _energy_and_forces(family, "off")
    assert torch.equal(energy, stock_energy) and torch.equal(forces, stock_forces)


def test_tf32x3_matmul_on_the_cpu_is_the_product_and_its_gradients():
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

    generator = torch.Generator().manual_seed(3)
    a = torch.randn((9, 5), generator=generator, dtype=torch.float64).requires_grad_()
    b = torch.randn((5, 4), generator=generator, dtype=torch.float64).requires_grad_()
    grad = torch.randn((9, 4), generator=generator, dtype=torch.float64)
    got = tf32x3_matmul(a, b)
    assert torch.equal(got, a @ b)
    for x, y in zip(torch.autograd.grad(got, (a, b), grad), torch.autograd.grad(a @ b, (a, b), grad)):
        assert torch.equal(x, y)
    (only_a,) = torch.autograd.grad(tf32x3_matmul(a, b.detach(), b.detach().t().contiguous()), (a,), grad)
    assert torch.equal(only_a, grad @ b.detach().t())


def test_the_op_traced_by_fx_has_first_and_second_derivatives():
    """A block linear inside an fx-traced module calls the op itself (OMOL-0's joint embedding on CUDA: "no
    autograd formula"); its gradients must equal the plain product's, twice."""
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

    weight = torch.randn((5, 4), dtype=torch.float64)

    class Linear(torch.nn.Module):
        def forward(self, x):
            return tf32x3_matmul(x, weight, weight.t().contiguous())

    traced = torch.fx.symbolic_trace(Linear())
    for module, product in ((traced, lambda x: x @ weight), (Linear(), lambda x: x @ weight)):
        values = []
        for function in (module, product):
            x = torch.randn((6, 5), generator=torch.Generator().manual_seed(4), dtype=torch.float64,
                            requires_grad=True)
            energy = (torch.tanh(function(x)) ** 2).sum()
            (forces,) = torch.autograd.grad(energy, x, create_graph=True)
            (second,) = torch.autograd.grad(forces.square().sum(), x)
            values.append((forces, second))
        for a, b in zip(*values):
            assert torch.equal(a, b)


def test_supported_needs_cuda():
    from fastmace.kernels.triton.tf32x3_gemm import supported

    assert supported("cpu") == (False, "cpu device: tf32x3 needs CUDA")


# --------------------------------------------------------------------------- planner


def _plan(rewrites, hardware=GPU, dtype=None, label="mh1-r2scan"):
    configs = dict(rewrites=rewrites, engine=dataclasses.replace(EngineConfig(), backend="auto"))
    if dtype:
        configs["precision"] = PrecisionConfig(dtype=dtype)
    return planner(**configs).plan(traits(label), nacl(), IntegratorConfig(), hardware, model_label=label)


A100 = Hardware("NVIDIA A100-SXM4-80GB", (8, 0), 80 * 2**30)


def test_decision_66_turns_auto_on_for_mh1_and_omat0_on_the_a100_only():
    for label in ("mh1-r2scan", "omat0-medium"):
        plan = _plan(RewriteConfig(), A100, label=label)
        assert plan.rewrites.tf32x3_gemms == "on" and plan.rewrites.grouped_launch == "off", label
        assert _plan(RewriteConfig(), A100, dtype="float64", label=label).rewrites.tf32x3_gemms == "off", label
    assert "not a measured GPU, decision 66" in dict(_plan(RewriteConfig()).auto_options_off)["tf32x3_gemms"]
    plan = _plan(RewriteConfig(), A100, label="mp0-medium")
    assert plan.rewrites.tf32x3_gemms == "off"
    assert "not a measured model, decision 66" in dict(plan.auto_options_off)["tf32x3_gemms"]
    plan = _plan(RewriteConfig(grouped_launch="on"), A100, label="omat0-medium")  # an explicit grouped_launch wins
    assert plan.rewrites.tf32x3_gemms == "off" and plan.rewrites.grouped_launch == "on"


def test_decision_66_on_the_a100_pcie_for_omat0_only():
    """The A100 80GB PCIe (owner, 4 November 2026): OMAT-0 medium only, where Benchmark_loup measured it."""
    pcie = Hardware("NVIDIA A100 80GB PCIe", (8, 0), 80 * 2**30)
    plan = _plan(RewriteConfig(), pcie, label="omat0-medium")
    assert plan.rewrites.tf32x3_gemms == "on" and plan.rewrites.grouped_launch == "off"
    plan = _plan(RewriteConfig(), pcie, label="mh1-r2scan")
    assert plan.rewrites.tf32x3_gemms == "off"
    assert dict(plan.auto_options_off)["tf32x3_gemms"] == "not a measured GPU, decision 66 (NVIDIA A100 80GB PCIe)"


def test_planner_keeps_it_off_unless_asked_and_enables_an_explicit_on():
    assert _plan(RewriteConfig()).rewrites.tf32x3_gemms == "off"
    assert _plan(RewriteConfig(tf32x3_gemms="on")).rewrites.tf32x3_gemms == "on"


def test_planner_refuses_it_below_sm80_in_float64_and_with_grouped_launch():
    with pytest.raises(PlanRefused, match="tf32x3_gemms.*compute capability"):
        _plan(RewriteConfig(tf32x3_gemms="on"), Hardware("Tesla T4", (7, 5), 16 * 2**30))
    with pytest.raises(PlanRefused, match="tf32x3_gemms.*float32"):
        _plan(RewriteConfig(tf32x3_gemms="on"), dtype="float64")
    with pytest.raises(PlanRefused, match="tf32x3_gemms and grouped_launch"):
        _plan(RewriteConfig(tf32x3_gemms="on", grouped_launch="on"))


@cueq
def test_an_explicit_on_wins_over_an_auto_record_it_conflicts_with():
    """gate_epilogue comes before tf32x3_gemms in application order and conflicts with it; its record stays default
    off (the planner's decision 66 rule turns it on), so a registry resolve with tf32x3_gemms "on" and the rest
    "auto" keeps tf32x3_gemms and leaves gate_epilogue off, not refused (leto suite, e2-default's first version)."""
    model, _ = tiny("mh1", backend="cueq")
    regs = reg.default_registry()
    assert regs.order().index("gate_epilogue") < regs.order().index("tf32x3_gemms")
    asked = {name: "off" if name in GPU_ONLY else "auto" for name in regs.names()} | {"tf32x3_gemms": "on"}
    resolution = regs.resolve(copy.deepcopy(model), asked, strict=True)
    assert resolution.decision("tf32x3_gemms").enabled
    gate = resolution.decision("gate_epilogue")
    assert not gate.enabled and not gate.refused
