"""e3nn's generated code as fx graphs instead of TorchScript, so dynamo traces it (opt-4 OPT4-T02, ``plans/opt-4.md``).

``RECORDS`` holds ``fx_e3nn_linear``. e3nn's ``o3.Linear`` generates its forward as an ``fx.GraphModule`` and,
with e3nn's default ``jit_script_fx``, scripts it into a TorchScript module (``_compiled_main``). A pickled
model keeps that TorchScript IR and restores it as TorchScript on load (``CodeGenMixin.__setstate__``),
whatever ``mace.tools.compile.disable_e3nn_codegen`` says, so the Linears of every loaded checkpoint run
TorchScript. Dynamo cannot trace TorchScript: each such Linear is a graph break. Inside a loop the break
makes dynamo skip the whole frame ("graph break in a for/while loop"), which is what happens to PolarMACE's
forward (the readouts with ``o3.Linear`` in the interaction and charge-update loops): the forward body runs
eagerly, its submodules compile as 31 separate graphs, and on POLAR-1-S NaCl 0.9 ms of the 3.6 ms force
call (538 of 1,226 kernels) are eager kernels and their eager autograd nodes (``docs/validation/opt4-T02.md``).

- **What it does.** For every ``o3.Linear`` whose ``_compiled_main`` is TorchScript, e3nn's own code
  generator (``_codegen_linear``) is run again on the module's irreps, instructions and weight sharing, and
  the resulting ``fx.GraphModule`` replaces the TorchScript module. These are the arguments e3nn's
  ``__init__`` passed, so it is the same graph that was scripted. The weights are untouched.
- **Rounding, not exact.** Eager, the graph runs the same operators as the TorchScript (bitwise on the CPU,
  tests/cpu/test_fx_e3nn_linear.py). Compiled, Inductor fuses code that ran operator by operator, so the
  force call rounds differently: the float32 force gate decides (``tools/force_stress_gate.py``).
- **Copies.** e3nn pickles the code as TorchScript (``CodeGenMixin.__getstate__``), so a deep copy of a
  rewritten model runs TorchScript again (same values). The force field applies records to its private
  copy just before compiling (``fastmace.model.compile.compile_model``), after the last copy.
- **Refused** for models without a TorchScript ``o3.Linear`` and for Linears with ``f_in``/``f_out``
  (weights with batch dimensions; the module does not keep those sizes).
- **Opt-in** (default off, the planner's ``never_automatic``); the owner decides defaults.
"""

from __future__ import annotations

from fastmace.rewrites.registry import CaseErrors, CheckResult, Rewrite, Support, threshold_for

NAME = "fx_e3nn_linear"
MACE_VERSIONS = ">=0.3.16,<0.3.17"
# The instance marker set by apply().
MARKER = "_fastmace_fx_linear"
# sha256 of inspect.getsource in e3nn 0.4.4: the generator apply() calls, the __init__ whose arguments it
# repeats, the forward that calls the generated code and the pickling that restores TorchScript.
SOURCE_HASHES = {
    "e3nn.o3._linear:_codegen_linear": "f87df78430a8fdd4e57183e4ac1f3ec4bbb5f3aab889ff46737989d09d130360",
    "e3nn.o3._linear:Linear.__init__": "32f3a2bf8f75f657f24a88d5ded225166633170872f317b514458dac56665c30",
    "e3nn.o3._linear:Linear.forward": "dea2c798e7f9576b2442f2d564074f3930b845480f36731a1793848785c2d6ae",
    "e3nn.util.codegen._mixin:CodeGenMixin.__setstate__":
        "8c09269347fcdaaee6f4d5bd5ed3a9722b6107f07573ae40665921da977b6e8f",
}


def _linear_class():
    try:
        from e3nn import o3
    except ImportError:
        return None
    return o3.Linear


def _targets(model):
    """(name, module) of every e3nn ``o3.Linear`` in ``model`` that runs TorchScript."""
    import torch

    linear = _linear_class()
    if linear is None or not hasattr(model, "named_modules"):
        return []
    return [(name, module) for name, module in model.named_modules()
            if type(module) is linear and isinstance(getattr(module, "_compiled_main", None), torch.jit.ScriptModule)]


def _batched(module):
    """True for a Linear built with ``f_in``/``f_out`` (weights or biases with batch dimensions)."""
    return module.weight.dim() > 1 or module.bias.dim() > 1


def applies_to(model):
    """(supported, reason): a model with TorchScript ``o3.Linear`` modules, none of them with batched weights."""
    targets = _targets(model)
    if not targets:
        return Support(False, "no e3nn o3.Linear runs TorchScript")
    batched = [name for name, module in targets if _batched(module)]
    if batched:
        return Support(False, f"o3.Linear with f_in/f_out (batched weights): {', '.join(batched)}")
    return Support(True, "ok")


def graph_of(module):
    """e3nn's ``fx.GraphModule`` for one ``o3.Linear``: ``_codegen_linear`` with the module's own arguments."""
    from e3nn.o3._linear import _codegen_linear

    graph, weight_numel, bias_numel = _codegen_linear(module.irreps_in, module.irreps_out, module.instructions,
                                                      None, None, shared_weights=module.shared_weights,
                                                      optimize_einsums=module._optimize_einsums)
    if (weight_numel, bias_numel) != (module.weight_numel, module.bias_numel):
        raise ValueError(f"the rebuilt code takes {weight_numel} weights and {bias_numel} biases, the module "
                         f"{module.weight_numel} and {module.bias_numel}")
    return graph


def apply(model, context=None):
    """Replace every TorchScript ``o3.Linear`` code of ``model`` by its fx graph; returns the module names."""
    support = applies_to(model)
    if not support.supported:
        raise ValueError(f"{NAME} does not apply: {support.reason}")
    names = []
    for name, module in _targets(model):
        module._compiled_main = graph_of(module)
        object.__setattr__(module, MARKER, True)
        names.append(name)
    return names


def selfcheck(original, rewritten, context=None, rows=7):
    """SPEC §6.4 per-module check: each rewritten Linear's graph against the original's TorchScript, in float64.

    Each pair is copied to float64 on its device and evaluated on the same random rows (``context.seed``).
    """
    import copy

    import torch

    linear = _linear_class()
    seed = getattr(context, "seed", 0)
    generator = torch.Generator().manual_seed(seed)
    originals = dict(original.named_modules())
    errors = CaseErrors()
    for name, module in rewritten.named_modules():
        if not module.__dict__.get(MARKER, False):
            continue
        reference = originals.get(name)
        if type(reference) is not linear:
            return CheckResult(False, float("inf"), threshold_for("float64"), len(errors.errors),
                               f"{name}: no matching o3.Linear in the original", "float64")
        device = module.weight.device
        ref, test = copy.deepcopy(reference).double(), copy.deepcopy(module).double()
        # A deep copy restores TorchScript (e3nn's pickling): put the graph under test back.
        test._compiled_main = module._compiled_main
        x = torch.randn(rows, module.irreps_in.dim, generator=generator, dtype=torch.float64).to(device)
        with torch.no_grad():
            expected = ref._compiled_main(x, ref.weight, ref.bias)
            got = test._compiled_main(x, test.weight, test.bias)
        errors.add(name, expected, got)
    if not errors.errors:
        return CheckResult(False, float("inf"), threshold_for("float64"), 0, "no rewritten o3.Linear to check",
                           "float64")
    return errors.result("float64")


RECORDS = {
    NAME: Rewrite(
        name=NAME, applies_to=applies_to, requires=(), conflicts=(), mace_versions=MACE_VERSIONS,
        source_hashes=SOURCE_HASHES, exactness="rounding", default="off", deterministic=True,
        workload_constraints=(), apply=apply, selfcheck=selfcheck),
}

__all__ = ["MARKER", "NAME", "RECORDS", "SOURCE_HASHES", "applies_to", "apply", "graph_of", "selfcheck"]
