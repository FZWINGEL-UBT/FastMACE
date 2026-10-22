"""Engine-level rewrites: process-wide patches of MACE and Inductor, and the per-model strain path (SPEC §6.2, §8.7).

Three §6.2 rows change functions the whole process shares rather than one model, so each is
installed through ``fastmace._patches`` (idempotent, recorded for provenance, restorable). The
fourth, ``single_graph_strain``, is per model since 2-T08b (docs/design/strain-per-model.md):

- ``capture_safe_mask_head``: MACE's multi-head readout mask without indexed assignment, which
  CUDA graph capture rejects (``mace.modules.blocks.mask_head``). Always on for the graph engine;
  same 0/1 mask, exact.
- ``first_derivative_only`` (``EngineConfig.first_derivative_only``): MACE's ``compute_forces``
  and ``compute_forces_virials`` without the second-derivative graph for ``training=True``
  calls (compiled cuEquivariance models). ``training=False`` goes to MACE's own function. Exact.
- ``unfuse_full_addmm`` (``EngineConfig.unfuse_full_addmm``): Inductor keeps ``mm + full-matrix
  add`` unfused on the GPU; a no-op custom pass with a fixed uuid keys the compiler cache.
  Same formulas, other kernel grouping.
- ``single_graph_strain`` (``RewriteConfig.single_graph_strain``): MACE PR 1498's one-graph
  ``get_symmetric_displacement`` (strained cell used directly, not gathered per edge; the
  per-atom gather gets the same rewrite). Several graphs keep MACE's function. Rounding. Nothing
  is installed in MACE: the force field binds its strain path into its own forward
  (``bind_strain_path``), on or off, so no other instance and no process-wide patch can change
  it. MACE's forwards call ``prepare_graph`` as a global of their module, and ``prepare_graph``
  calls ``get_symmetric_displacement`` as a global of ``mace.modules.utils``; the bound forward
  is the same code with a copy of its module's globals in which ``prepare_graph`` is
  ``strain_prepare_graph``: the same code again, with a copy of ``mace.modules.utils``' globals
  holding the one-graph function or MACE's own. The copies are taken at the first binding of each
  forward and choice, and every later binding reuses them (2-T19b: under torch.compile all builds
  must share one globals dict). So a forward that reads a global some process-wide patch replaces
  is refused (``strain_problems``): the copy would freeze it. MACE's forwards read none
  (``get_outputs`` looks ``compute_forces`` up in ``mace.modules.utils`` itself).
  ``_patches.set_single_graph_strain`` remains for the reference calculators, which describe
  MACE's own calculator.

``PROCESS_RECORDS`` holds all four as §6.1 ``Rewrite`` records, and their self-checks compare the
replacement with MACE's function on random inputs. The ``apply`` of the first three installs the
patch (the model is not touched); ``single_graph_strain``'s installs nothing and touches no model,
because the force field binds the path after its other rewrites (the float64 forward included).
``RECORDS`` holds the one that a ``RewriteConfig`` field switches (``single_graph_strain``), for
the rewrite registry; the other three are switched by ``EngineConfig`` or always on and are
installed with ``install``.
"""

from __future__ import annotations

import functools
import types

from fastmace import _patches
from fastmace.rewrites.registry import CheckResult, Rewrite, Support, check_tensors

MACE_VERSIONS = ">=0.3.16,<0.3.17"
UNFUSED_ADDMM_UUID = "fastmace-unfused-full-addmm-v1"

# sha256 of inspect.getsource in mace-torch 0.3.16 of the functions each patch replaces.
MASK_HEAD_SOURCE = {"mace.modules.irreps_tools:mask_head":
                    "bcb74869e70f9cd7ddfdfb7e133f4aacec058e8dd996b3c6eeb916e6e1b49e8e"}
FORCES_SOURCES = {
    "mace.modules.utils:compute_forces": "b19d3e4acaf2c13eeff68562e22d6bb54cc03d73901336aba63e0b9a02c13565",
    "mace.modules.utils:compute_forces_virials": "8c1d2ab9d98ce99f3b8791c9c78f5f10d0d32ff5508ff48e0a641c174fed5ba0",
}
STRAIN_SOURCE = {"mace.modules.utils:get_symmetric_displacement":
                 "d75106300f967df0872e253c6d9c4607c0783a6dd8fe35045204e958a0b36437",
                 # The per-model path reuses prepare_graph's code with other globals (2-T08b).
                 "mace.modules.utils:prepare_graph":
                 "a369db1b449ebb4f4105cf7e42ad4cebea28950cd9d51a07f7b6cc8ec425c557"}


def _mace_utils():
    from mace.modules import utils

    return utils


# --------------------------------------------------------------------------- replacement functions


def capture_safe_mask_head(x, head, num_heads):
    """Same result as mace.modules.irreps_tools.mask_head, without indexed assignment.

    MACE builds the multi-head readout mask with `mask[idx, :, head] = 1`, which
    CUDA graph capture rejects. Comparing against arange gives the same 0/1 mask.
    """
    import torch

    channels = x.shape[1] // num_heads
    selected = head[:, None] == torch.arange(num_heads, device=x.device)  # [n_nodes, heads]
    mask = selected[:, :, None].expand(-1, -1, channels).reshape(x.shape)
    return x * mask.to(torch.get_default_dtype())


def first_derivative_forces(energy, positions, training=True):
    """mace.modules.utils.compute_forces (mace-torch 0.3.16) without the second-derivative graph.

    training=False is passed to MACE's own function unchanged. With
    training=True the gradient is taken with create_graph=False; retain_graph
    stays True, because inductor fails to trace the compiled model without it
    ("Trying to backward through the graph a second time"). Forces are the same.
    """
    import torch

    if not training:
        return _patches._original(_mace_utils(), "compute_forces")(energy, positions, training=False)
    gradient = torch.autograd.grad(outputs=[energy], inputs=[positions],
                                   grad_outputs=[torch.ones_like(energy)],
                                   retain_graph=True, create_graph=False, allow_unused=True)[0]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def first_derivative_forces_virials(energy, positions, displacement, cell, training=True, compute_stress=False):
    """mace.modules.utils.compute_forces_virials (mace-torch 0.3.16) without the second-derivative graph.

    As first_derivative_forces, for the stress path: training=False is passed to
    MACE's function unchanged; training=True keeps retain_graph but not create_graph.
    """
    import torch

    if not training:
        return _patches._original(_mace_utils(), "compute_forces_virials")(
            energy, positions, displacement, cell, training=False, compute_stress=compute_stress)
    forces, virials = torch.autograd.grad(outputs=[energy], inputs=[positions, displacement],
                                          grad_outputs=[torch.ones_like(energy)],
                                          retain_graph=True, create_graph=False, allow_unused=True)
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))
    return -1 * forces, -1 * virials, stress


def single_graph_symmetric_displacement(positions, unit_shifts, cell, edge_index, num_graphs, batch,
                                        displacement=None):
    """mace.modules.utils.get_symmetric_displacement (mace-torch 0.3.16) with MACE PR 1498's one-graph path.

    Returns (strained positions, strained shifts, displacement) as MACE does.
    With one graph every atom and edge belongs to graph 0, so the gathered
    per-atom strain and per-edge cell are the same 3 x 3 matrices for all rows.
    """
    import torch

    if num_graphs != 1:
        return _patches._original(_mace_utils(), "get_symmetric_displacement")(
            positions, unit_shifts, cell, edge_index, num_graphs, batch, displacement)
    if cell is None:
        cell = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)
    if displacement is None:
        displacement = torch.zeros((1, 3, 3), dtype=positions.dtype, device=positions.device)
        displacement = displacement + positions.sum() * 0.0
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.einsum("be,ec->bc", positions, symmetric_displacement[0])
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum("be,ec->bc", unit_shifts, cell[0])
    return positions, shifts, displacement


def unfused_full_addmm_rule(original, post_grad):
    """Inductor's should_prefer_unfused_addmm (``original``), also unfused for a full-matrix add on the GPU.

    `mm + add` stays unfused on the GPU whenever the added tensor is not bias-like
    (a full matrix, which cuBLAS would first have to copy into the output). For
    bias-like adds torch's own decision stands.
    """
    def should_prefer_unfused_addmm(match):
        if original(match):
            return True
        inp = match.kwargs["inp"]
        if not inp.meta["val"].is_cuda:
            return False
        return not post_grad._is_bias_like_addmm_input(inp, match.output_node())

    should_prefer_unfused_addmm.unfused_full_addmm = True
    return should_prefer_unfused_addmm


@functools.cache
def _key_class():
    from torch._inductor.custom_graph_pass import CustomGraphPass

    class UnfusedFullAddmmKey(CustomGraphPass):
        """No-op pass: puts the addmm choice into the compiler-cache key."""

        def __call__(self, graph):
            pass

        def uuid(self):
            return UNFUSED_ADDMM_UUID

    return UnfusedFullAddmmKey


def unfused_addmm_key():
    """A no-op Inductor custom pass whose uuid keys the compiler cache by the addmm choice."""
    return _key_class()()


# --------------------------------------------------------------------------- the strain path per model

# Where bind_strain_path records the bound path on the model: "one_graph" or "mace".
STRAIN_ATTRIBUTE = "_fastmace_strain_path"


class StrainBindingError(RuntimeError):
    """A forward cannot take a per-model strain path (``strain_problems`` gives why)."""


def _global_names(code):
    """Every name ``code`` and the code objects nested in it may look up as a global."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def _frozen_patches(function):
    """``"module:name"`` of the globals ``function`` reads that a process-wide patch may replace.

    A copy of the module's globals would freeze them at bind time. The strain function itself
    is excluded: it is the global the copy replaces on purpose.
    """
    own = set(_patches.SOURCE_PINS["single_graph_strain"])
    patched = {target for pins in _patches.SOURCE_PINS.values() for target in pins} - own
    module = function.__globals__.get("__name__", "")
    return sorted(f"{module}:{name}" for name in _global_names(function.__code__)
                  if name in function.__globals__ and f"{module}:{name}" in patched)


def strain_problems(function):
    """Why ``function`` (a model forward) cannot take a per-model strain path; [] when it can."""
    code = getattr(function, "__code__", None)
    if code is None or not hasattr(function, "__globals__"):
        return [f"{function!r} is not a Python function"]
    name = f"{function.__globals__.get('__name__', '?')}.{getattr(function, '__qualname__', '?')}"
    problems = []
    if "prepare_graph" not in _global_names(code) or "prepare_graph" not in function.__globals__:
        problems.append(f"{name} does not read prepare_graph as a global of its module")
    problems += [f"{name} reads {target}, which a process-wide patch replaces; a copy of its module's globals "
                 "would freeze it" for target in _frozen_patches(function)]
    return problems


# The rebound functions, made once per process for each (function, strain function); the keys hold
# the originals, so an id is never reused. Dynamo caches a compiled frame on its code object but keeps
# its __compiled_fn_* in the globals of the frame it compiled. When every binding made a new function
# from MACE's code object with a fresh copy of the globals, a later build reused a cached frame in a
# dict without that name (NameError; 2-T23, OMOL-0, plan task 2-T19b). So each (function, strain
# function) gets one rebound function with its own code object (_rebound), reused by every build.
_REBOUND: dict = {}
# The copied globals of every rebound function, by the module they copy ("mace.modules.utils" ->
# [namespace, ...]). A process patch installed or undone after a copy was taken reaches it through
# follow_patch (3-T04b): the copy is the module at bind time except the names it replaces on purpose.
_SNAPSHOTS: dict = {}


def _snapshot(function):
    """A copy of ``function``'s globals that follows later process patches (``follow_patch``)."""
    namespace = dict(function.__globals__)
    _SNAPSHOTS.setdefault(function.__globals__.get("__name__", ""), []).append(namespace)
    return namespace


def follow_patch(target, old, new):
    """``_patches`` replaced ``"module:attribute"`` (``old`` by ``new``): update every copy that still holds ``old``.

    A copy holding something else replaced that name on purpose (the strain function of a variant, the
    ``prepare_graph`` of a forward) and keeps it. The strain patch never reaches a copy: an "off"
    variant keeps MACE's own strain function while the reference calculators' patch is in force.
    """
    if target in _patches.SOURCE_PINS["single_graph_strain"]:
        return
    module, attribute = target.split(":", 1)
    for namespace in _SNAPSHOTS.get(module, ()):
        if attribute in namespace and namespace[attribute] is old:
            namespace[attribute] = new


def _rebound(function, namespace):
    """``function`` with the same bytecode, defaults and closure, and ``namespace`` as its globals.

    The code object is a copy: Dynamo caches compiled frames per code object, so a rebound
    function sharing MACE's (or another strain choice's) code object would reuse a frame compiled
    in other globals (2-T19b).
    """
    code = function.__code__.replace()
    if code is function.__code__:  # an interpreter that returns the same object for an empty replace
        code = function.__code__.replace(co_name=function.__code__.co_name + "")
    out = types.FunctionType(code, namespace, function.__name__, function.__defaults__,
                             function.__closure__)
    out.__kwdefaults__ = dict(function.__kwdefaults__) if function.__kwdefaults__ else None
    out.__qualname__, out.__module__, out.__doc__ = function.__qualname__, function.__module__, function.__doc__
    out.__annotations__ = dict(function.__annotations__)
    out.__dict__.update(function.__dict__)
    return out


def strain_prepare_graph(one_graph):
    """MACE's ``prepare_graph`` with its own strain function: one-graph (True) or MACE's (False).

    The same code, with a copy of ``mace.modules.utils``' globals taken at the first call; only
    ``get_symmetric_displacement`` differs, and MACE's own is the original even while the
    reference calculators' process patch is in force. Other process patches installed or undone
    later reach the copy (``follow_patch``). Made once per process for each choice
    (``_REBOUND``), so all builds share it under torch.compile.
    """
    utils = _mace_utils()
    frozen = _frozen_patches(utils.prepare_graph)
    if frozen:
        raise StrainBindingError("mace.modules.utils.prepare_graph reads " + ", ".join(frozen)
                                 + ", which a process-wide patch replaces")
    strain = (single_graph_symmetric_displacement if one_graph
              else _patches._original(utils, "get_symmetric_displacement"))
    key = (utils.prepare_graph, strain)
    found = _REBOUND.get(key)
    if found is None:
        namespace = _snapshot(utils.prepare_graph)
        namespace["get_symmetric_displacement"] = strain
        found = _REBOUND[key] = _rebound(utils.prepare_graph, namespace)
    return found


def strain_forward(function, one_graph):
    """``function`` (a model forward) whose ``prepare_graph`` is ``strain_prepare_graph(one_graph)``.

    The same code, with a copy of its globals taken at the first call for this function and choice
    (``_REBOUND``: made once per process, so all builds share it under torch.compile). Raises
    ``StrainBindingError`` with ``strain_problems`` when the forward does not qualify.
    """
    problems = strain_problems(function)
    if problems:
        raise StrainBindingError("; ".join(problems))
    prepare_graph = strain_prepare_graph(one_graph)
    key = (function, prepare_graph)
    found = _REBOUND.get(key)
    if found is None:
        namespace = _snapshot(function)
        namespace["prepare_graph"] = prepare_graph
        found = _REBOUND[key] = _rebound(function, namespace)
    return found


def bind_strain_path(model, one_graph):
    """Bind the strain path to this instance (a private copy: never the caller's model); returns it.

    The instance's current forward (the float64 one when installed, else its class's) is rebound
    with ``strain_forward``. A PolarMACE instance keeps the class-level dispatcher of
    ``fastmace.longrange.polar``, and its holder carries the choice instead.
    """
    from fastmace.longrange import polar

    if polar.is_polar(model):
        polar.set_single_graph_strain(model, one_graph)
    else:
        current = model.__dict__.get("forward")
        function = current.__func__ if current is not None else type(model).forward
        model.forward = types.MethodType(strain_forward(function, one_graph), model)
    object.__setattr__(model, STRAIN_ATTRIBUTE, "one_graph" if one_graph else "mace")
    return model


def strain_path(model):
    """The strain path bound to ``model``: "one_graph", "mace", or None (not bound: MACE's, as installed)."""
    return model.__dict__.get(STRAIN_ATTRIBUTE)


# --------------------------------------------------------------------------- installation



def install(engine=None, rewrites=None, oeq=False):
    """Install the process-wide patches a force field needs; returns their PatchRecords.

    ``engine`` is an EngineConfig and ``rewrites`` a RewriteConfig (defaults when None).
    ``capture_safe_mask_head`` is always installed (it must be in place before tracing);
    ``first_derivative_only`` when asked for and not on OEq models, which run with
    training=False; ``unfuse_full_addmm`` when asked for (nothing on a torch that does not need
    it). ``rewrites`` is accepted as before, but ``single_graph_strain`` is no longer a process
    patch: the force field binds it per model (``bind_strain_path``). Nothing is undone here: a
    field that is off leaves an installed patch in force (a process constraint, SPEC §8.7).

    A patch whose replaced source does not hash (another MACE or torch, SPEC §6.3) is not
    installed, and the reason is logged (plan task 2-T04): MACE's own function stays. Without
    the capture-safe mask a multi-head model cannot be captured
    (``fastmace.rewrites.versions.capture_support``).
    """
    from fastmace.config import EngineConfig

    engine = engine or EngineConfig()
    wanted = ["capture_safe_mask_head"]
    if engine.first_derivative_only and not oeq:
        wanted.append("first_derivative_only")
    if engine.unfuse_full_addmm:
        wanted.append("unfuse_full_addmm")
    records = [_patches.install_if_qualified(name) for name in wanted]
    return tuple(record for record in records if record is not None)


# --------------------------------------------------------------------------- self-checks


def _selfcheck_mask_head(original, rewritten, context):
    import torch
    from mace.modules import irreps_tools

    generator = torch.Generator().manual_seed(context.seed)
    cases = []
    for heads, channels in ((1, 4), (3, 5)):
        x = torch.randn((13, heads * channels), generator=generator, dtype=torch.float64)
        head = torch.randint(heads, (13,), generator=generator)
        with _default_dtype(torch.float64):
            cases.append((f"mask_head[{heads}]", irreps_tools.mask_head(x, head, heads),
                          capture_safe_mask_head(x, head, heads)))
    return check_tensors(cases, torch.float64)


class _default_dtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        import torch

        self.saved = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, *exc):
        import torch

        torch.set_default_dtype(self.saved)


def _energy(positions, displacement=None):
    """A nonlinear scalar of the positions (and the strain), as a model energy would be."""
    import torch

    strained = positions if displacement is None else positions + positions @ displacement[0]
    distances = torch.cdist(strained, strained) + torch.eye(len(strained), dtype=strained.dtype)
    return (torch.exp(-distances) * torch.sin(strained.sum(dim=1))[:, None]).sum().reshape(1)


def _selfcheck_first_derivative(original, rewritten, context):
    import torch

    utils = _mace_utils()
    forces = _patches._original(utils, "compute_forces")
    virials = _patches._original(utils, "compute_forces_virials")
    generator = torch.Generator().manual_seed(context.seed)
    cases = []
    for training in (True, False):
        positions = torch.randn((7, 3), generator=generator, dtype=torch.float64, requires_grad=True)
        displacement = (0.02 * torch.randn((1, 3, 3), generator=generator, dtype=torch.float64)).requires_grad_()
        cell = torch.eye(3, dtype=torch.float64) * 5
        expected = forces(_energy(positions), positions, training=training)
        got = first_derivative_forces(_energy(positions), positions, training=training)
        cases.append((f"compute_forces(training={training})", expected.detach(), got.detach()))
        expected = virials(_energy(positions, displacement), positions, displacement, cell, training=training,
                           compute_stress=True)
        got = first_derivative_forces_virials(_energy(positions, displacement), positions, displacement, cell,
                                              training=training, compute_stress=True)
        cases += [(f"compute_forces_virials(training={training})", a.detach(), b.detach())
                  for a, b in zip(expected, got)]
    return check_tensors(cases, torch.float64)


def _selfcheck_unfuse(original, rewritten, context):
    return CheckResult(True, 0.0, 1e-10, 0, "no numerical check: Inductor's kernel grouping only", "float64")


def strain_inputs(num_graphs=1, atoms=7, edges=40, seed=0, strain=0.0):
    """Random get_symmetric_displacement inputs; strain > 0 gives a nonzero (non-symmetric) displacement leaf."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    positions = torch.randn(atoms, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    unit_shifts = torch.randint(-2, 3, (edges, 3), generator=generator).double()
    cell = (torch.eye(3, dtype=torch.float64) * 5 + 0.3 * torch.rand(3, 3, generator=generator,
                                                                    dtype=torch.float64)).repeat(num_graphs, 1)
    edge_index = torch.randint(0, atoms, (2, edges), generator=generator)
    batch = torch.sort(torch.randint(0, num_graphs, (atoms,), generator=generator)).values
    displacement = (strain * torch.randn((num_graphs, 3, 3), generator=generator, dtype=torch.float64)
                    ).requires_grad_(True)
    return dict(positions=positions, unit_shifts=unit_shifts, cell=cell, edge_index=edge_index,
                num_graphs=num_graphs, batch=batch, displacement=displacement)


def strained_outputs_and_gradients(function, inputs, seed=1):
    """Strained positions and shifts, and the gradients of a random linear function of them."""
    import torch

    positions, shifts, displacement = function(**inputs)
    generator = torch.Generator().manual_seed(seed)
    weight_p = torch.randn(positions.shape, generator=generator, dtype=torch.float64)
    weight_s = torch.randn(shifts.shape, generator=generator, dtype=torch.float64)
    total = (positions * weight_p).sum() + (shifts * weight_s).sum()
    grad_x, grad_d = torch.autograd.grad(total, [inputs["positions"], displacement])
    return positions.detach(), shifts.detach(), grad_x, grad_d


def _selfcheck_strain(original, rewritten, context):
    import torch

    mace_function = _patches._original(_mace_utils(), "get_symmetric_displacement")
    cases = []
    for seed, strain in ((0, 0.0), (1, 0.03)):
        inputs = strain_inputs(seed=context.seed + seed, strain=strain)
        expected = strained_outputs_and_gradients(mace_function, inputs)
        got = strained_outputs_and_gradients(single_graph_symmetric_displacement, inputs)
        cases += [(f"get_symmetric_displacement(strain={strain})", a, b) for a, b in zip(expected, got)]
    return check_tensors(cases, torch.float64)


# --------------------------------------------------------------------------- records


def _always(model):
    return Support(True, "ok")


def _record(name, apply, selfcheck, exactness, default, sources, constraints=()):
    # Each switches how MACE's own torch code runs; nothing sums atomically (SPEC §9.3, 2-T16).
    return Rewrite(name=name, applies_to=_always, requires=(), conflicts=(), mace_versions=MACE_VERSIONS,
                   source_hashes=sources, exactness=exactness, default=default,
                   workload_constraints=constraints, apply=apply, selfcheck=selfcheck, deterministic=True)


def _apply_mask_head(model, context):
    return _patches.set_capture_safe_mask_head(True)


def _apply_first_derivative(model, context):
    return _patches.set_first_derivative_only(True)


def _apply_unfuse(model, context):
    return _patches.set_unfuse_full_addmm(True)


def _apply_strain(model, context):
    # Nothing installed, no model touched: the force field binds the path (bind_strain_path), so a
    # failing self-check has nothing process-wide to undo.
    return "per model: bound by the force field (rewrites.engine.bind_strain_path)"


PROCESS_RECORDS = {
    "capture_safe_mask_head": _record("capture_safe_mask_head", _apply_mask_head, _selfcheck_mask_head,
                                      "exact", "on", MASK_HEAD_SOURCE),
    "first_derivative_only": _record("first_derivative_only", _apply_first_derivative, _selfcheck_first_derivative,
                                     "exact", "on", FORCES_SOURCES),
    "unfuse_full_addmm": _record("unfuse_full_addmm", _apply_unfuse, _selfcheck_unfuse, "exact", "on", {}),
    "single_graph_strain": _record("single_graph_strain", _apply_strain, _selfcheck_strain, "rounding", "auto",
                                   STRAIN_SOURCE, ("single_graph",)),
}
# Where each is switched (docs/design/switch-map.md); None: always on for the graph engine.
CONFIG_FIELDS = {
    "capture_safe_mask_head": None,
    "first_derivative_only": "EngineConfig.first_derivative_only",
    "unfuse_full_addmm": "EngineConfig.unfuse_full_addmm",
    "single_graph_strain": "RewriteConfig.single_graph_strain",
}
# The records the rewrite registry can switch (a RewriteConfig field).
RECORDS = {"single_graph_strain": PROCESS_RECORDS["single_graph_strain"]}

__all__ = [
    "CONFIG_FIELDS", "PROCESS_RECORDS", "RECORDS", "STRAIN_ATTRIBUTE", "StrainBindingError", "UNFUSED_ADDMM_UUID",
    "bind_strain_path", "capture_safe_mask_head", "first_derivative_forces", "first_derivative_forces_virials",
    "install", "single_graph_symmetric_displacement", "strain_forward", "strain_inputs", "strain_path",
    "strain_prepare_graph", "strain_problems", "strained_outputs_and_gradients", "unfused_addmm_key",
    "unfused_full_addmm_rule",
]
