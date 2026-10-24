"""The ``fx_e3nn_linear`` record: e3nn's ``o3.Linear`` code as an fx graph, not TorchScript (opt-4 OPT4-T02).

- the record is rounding-class (Inductor then compiles code that ran eagerly), deterministic, opt-in
  (``never_automatic``) and pins e3nn's code generator;
- it applies to models with TorchScript ``o3.Linear`` modules and refuses models without any, and ones whose
  Linears have ``f_in``/``f_out`` batch dimensions (their code cannot be rebuilt from the module);
- applied, every such Linear runs an ``fx.GraphModule``; the model's energy and forces stay bitwise equal on
  the CPU, the self-check passes and catches a wrong graph, and dynamo traces the Linear without a graph
  break (TorchScript is a break, and one inside a loop sends a whole frame back to eager).
"""

import copy

import pytest
import torch

pytest.importorskip("mace")

from _support.structures import nacl  # noqa: E402
from _support.tiny_models import tiny_family_calculator  # noqa: E402
from fastmace.rewrites import codegen  # noqa: E402
from fastmace.rewrites import registry as reg  # noqa: E402

torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4's constants under torch's weights_only load


def _tiny():
    calc = tiny_family_calculator("residual", [11, 17])
    return calc, calc.models[0]


def _linears(model):
    from e3nn import o3

    return [module for module in model.modules() if type(module) is o3.Linear]


def _outputs(calc, model, atoms):
    batch = calc._atoms_to_batch(atoms).to_dict()
    out = model(batch, compute_stress=False, training=False)
    return out["energy"].detach(), out["forces"].detach()


def test_the_record_is_rounding_opt_in_and_pinned():
    record = codegen.RECORDS["fx_e3nn_linear"]
    assert (record.default, record.exactness, record.deterministic, record.requires) == ("off", "rounding", True, ())
    assert set(record.source_hashes) == set(codegen.SOURCE_HASHES)
    assert reg.qualify(record).supported
    registry = reg.default_registry()
    assert "fx_e3nn_linear" in registry and registry.config_field("fx_e3nn_linear") == "fx_e3nn_linear"
    from fastmace.config import RewriteConfig
    from fastmace.planner import fixed

    assert RewriteConfig().fx_e3nn_linear == "auto"
    assert fixed.AUTO_RULES["RewriteConfig.fx_e3nn_linear"] == "never_automatic"
    assert "fx_e3nn_linear" in fixed.REWRITE_FIELDS


def test_applies_to_torchscript_linears_only():
    _, model = _tiny()
    assert all(isinstance(module._compiled_main, torch.jit.ScriptModule) for module in _linears(model))
    support = codegen.applies_to(model)
    assert support.supported, support.reason
    rewritten = copy.deepcopy(model)
    names = codegen.apply(rewritten)
    assert len(names) == len(_linears(model)) > 0
    assert all(isinstance(module._compiled_main, torch.fx.GraphModule) for module in _linears(rewritten))
    again = codegen.applies_to(rewritten)
    assert not again.supported and "no e3nn o3.Linear runs TorchScript" in again.reason
    assert not codegen.applies_to(torch.nn.Linear(2, 2)).supported


def test_refuses_batched_weights():
    from e3nn import o3

    container = torch.nn.Sequential(o3.Linear("2x0e", "2x0e", f_in=3, f_out=3))
    support = codegen.applies_to(container)
    assert not support.supported and "f_in" in support.reason
    with pytest.raises(ValueError, match="f_in"):
        codegen.apply(container)


def test_bitwise_equal_on_the_cpu():
    calc, model = _tiny()
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    rewritten = copy.deepcopy(model)
    codegen.apply(rewritten)
    for _ in range(2):  # the first call of a process rounds differently (TorchScript's first run)
        expected = _outputs(calc, model, atoms)
        got = _outputs(calc, rewritten, atoms)
    for a, b in zip(expected, got):
        assert torch.equal(a, b)
    # A deep copy stores the code as TorchScript again (e3nn's pickling) and still gives the same values.
    copied = copy.deepcopy(rewritten)
    assert all(isinstance(module._compiled_main, torch.jit.ScriptModule) for module in _linears(copied))
    for _ in range(2):
        again = _outputs(calc, copied, atoms)
    assert all(torch.equal(a, b) for a, b in zip(expected, again))


def test_self_check_passes_and_catches_a_wrong_graph():
    _, model = _tiny()
    rewritten = copy.deepcopy(model)
    codegen.apply(rewritten)
    result = codegen.RECORDS["fx_e3nn_linear"].selfcheck(model, rewritten, reg.Context())
    assert result.passed and result.modules == len(_linears(model)), result

    class Doubled(torch.nn.Module):
        def __init__(self, graph):
            super().__init__()
            self.graph_module = graph

        def forward(self, x, w, b):
            return 2 * self.graph_module(x, w, b)

    target = _linears(rewritten)[0]
    target._compiled_main = torch.fx.symbolic_trace(Doubled(target._compiled_main))
    bad = codegen.RECORDS["fx_e3nn_linear"].selfcheck(model, rewritten, reg.Context())
    assert not bad.passed and bad.max_relative_error > 0.5


def test_dynamo_traces_the_rewritten_linear_without_a_break():
    from e3nn import o3

    linear = o3.Linear("4x0e + 2x1o", "3x0e + 2x1o")
    x = torch.randn(5, linear.irreps_in.dim)
    expected, width = linear(x), linear.irreps_out.dim  # Irreps outside: guards on them need polar's patch

    def twice(inputs):  # a loop, as in PolarMACE's forward: a break inside it skips the whole frame
        out = inputs.new_zeros(inputs.shape[0], width)
        for _ in range(2):
            out = out + linear(inputs)
        return out

    torch._dynamo.reset()
    with pytest.raises(Exception, match="(?i)torchscript|script|getattr"):
        torch.compile(twice, backend="eager", fullgraph=True)(x)
    codegen.apply(torch.nn.Sequential(linear))
    torch._dynamo.reset()
    assert torch.equal(torch.compile(twice, backend="eager", fullgraph=True)(x), 2 * expected)
    torch._dynamo.reset()
