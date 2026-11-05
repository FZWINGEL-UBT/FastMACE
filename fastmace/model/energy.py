"""Float64 energy accumulation (SPEC §7.3 "Energy", §6.2 ``float64_energy``; plan task 1b-T03).

MACE sums the potential energy in the model's precision. With all-electron E0 references a
float32 total of NbCl5 3x3x3 is near -2.08e7 eV, where one float32 step is 2 eV, so stock float32
energies are off by up to 102 meV/atom (§4.3). Here the energy is reduced into float64:

- the E0 table is an exact float64 copy of the checkpoint's table (``LoadedModel``'s
  ``atomic_energies``, captured before MACE's calculator converts the model to float32), looked
  up per atom by ``node_e0_float64`` and passed in the batch as ``data[NODE_E0_KEY]``; without it
  the model's own table is upcast (exact for float64 models and float32 checkpoints);
- every other term is computed in the force precision, as MACE computes it, and reduced over
  atoms (or k-vectors) into float64: interaction and pair energies, the embedding readout energy
  of OMOL-0, PolarMACE's local electron energy, electrostatic energy (per-k and self terms,
  ``fastmace.longrange.polar``) and field term.

Forces and stress still come from MACE's autograd of its own float32 scalar (design A, SPEC §7.3
"Energy and forces are one scalar"): the gradient of a sum is 1 whatever the accumulator, so the
two are the same derivative, and keeping MACE's backward keeps every force and stress bit.

The reduction is a reference treatment, not a rewrite (§6.2). It replaces blocks of MACE
0.3.16's ``MACE.forward`` and ``ScaleShiftMACE.forward`` (``PolarMACE.forward``: in
``fastmace.longrange.polar``) in their installed source, pinned by sha256 like the rewrites; a
model whose forward is not the qualified one is refused with ``EnergyReductionError``. The
rewritten forward returns, besides MACE's outputs:

- ``"energy"``: the float64 total ``(n_graphs,)``, replacing MACE's;
- ``"node_energy_float64"``: per-atom float64 energies of the atom-resolved terms;
- ``"energy_unresolved_float64"``: the float64 sum of the terms that are not atom-resolved
  (k-space electrostatics and the field term of PolarMACE; zero otherwise).

MACE's ``"node_energy"`` is left as it is and never used: in 0.3.16 it omits the embedding
readout energy of ``ScaleShiftMACE`` and the electron, electrostatic and field energies of
``PolarMACE``.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import textwrap

from fastmace.errors import FastMACEError
from fastmace.kernels.gpu_models import GPUModels

# Batch key of the per-atom float64 E0 (n_atoms,); the force field fills it once per structure.
NODE_E0_KEY = "fastmace_node_e0"

# sha256 of inspect.getsource of the forwards rewritten here, in MACE 0.3.16.
SOURCE_HASHES = {
    "mace.modules.models:MACE.forward": "22ae72645d7a669b5e07130947634042e3af7ed8c29b5da64f5f7ba39f76783a",
    "mace.modules.models:ScaleShiftMACE.forward": "7f4daea32cb9b55c59db5ee623f8310ef65e41cf0f717278f743ae8bd2782e1d",
}

# Terms per model class: atom-resolved (in node_energy_float64) and not (energy_unresolved_float64).
TERMS = {
    "MACE": (("e0", "pair", "interaction", "embedding_readout"), ()),
    "ScaleShiftMACE": (("e0", "pair", "interaction", "embedding_readout"), ()),
    "PolarMACE": (("e0", "interaction", "local_electron"), ("electrostatic", "field")),
}

_EMBEDDING_EDITS = (
    ("""        if hasattr(self, "joint_embedding"):
""", """        embedding_node_energy64 = torch.zeros_like(node_e0, dtype=torch.float64)
        if hasattr(self, "joint_embedding"):
"""),
    ("""                e0 += embedding_energy
""", """                e0 += embedding_energy
                embedding_node_energy64 = embedding_node_energy.double()
"""),
)

_RETURN_EDIT = ("""            "energy": total_energy,
""", """            "energy": energy64,
            "node_energy_float64": node_energy64,
            "energy_unresolved_float64": unresolved64,
""")

_OUTPUTS = "        forces, virials, stress, hessian, edge_forces = get_outputs(\n"

# MACE.forward sums [node_e0, pair, readout_1, ...]; ScaleShiftMACE.forward has node_inter_es after
# its scale and shift (pair energy included).
EDITS = {
    "MACE": _EMBEDDING_EDITS + (
        (_OUTPUTS, """        node_energy64 = _fastmace_node_e0(data, node_e0) + embedding_node_energy64
        for node_term in node_energies_list[1:]:
            node_energy64 = node_energy64 + node_term.double()
        unresolved64 = torch.zeros(num_graphs, dtype=torch.float64, device=node_energy64.device)
        energy64 = scatter_sum(node_energy64, data["batch"], dim=-1, dim_size=num_graphs) + unresolved64
""" + _OUTPUTS),
        _RETURN_EDIT),
    "ScaleShiftMACE": _EMBEDDING_EDITS + (
        (_OUTPUTS, """        node_energy64 = (
            _fastmace_node_e0(data, node_e0) + node_inter_es.double() + embedding_node_energy64
        )
        unresolved64 = torch.zeros(num_graphs, dtype=torch.float64, device=node_energy64.device)
        energy64 = scatter_sum(node_energy64, data["batch"], dim=-1, dim_size=num_graphs) + unresolved64
""" + _OUTPUTS),
        _RETURN_EDIT),
}


class EnergyReductionError(FastMACEError):
    """The float64 energy reduction does not apply to this model; ``reason`` says why (SPEC §7.3, §6.2)."""

    default_next_step = ("Use the qualified MACE version (0.3.16) and a MACE, ScaleShiftMACE or PolarMACE model; "
                         "float64_energy=False gives 1a's float32 sum, for A/B against the research engine only.")

    def __init__(self, reason, why=None, next_step=None, record=None, environment=None):
        super().__init__(f"the float64 energy reduction is unavailable: {reason}", why, next_step, record,
                         environment)
        self.reason = reason


def node_e0(data, node_e0_model):
    """Per-atom float64 E0: ``data[NODE_E0_KEY]`` when the force field put it there, else the model's upcast."""
    if NODE_E0_KEY in data:
        return data[NODE_E0_KEY]
    return node_e0_model.double()


def node_e0_float64(table, node_attrs, head=0):
    """Per-atom float64 E0 ``(n_atoms,)`` from a float64 table and the one-hot species ``(n_atoms, n_elements)``.

    ``table`` is ``AtomicEnergiesBlock.atomic_energies`` in float64: ``(n_elements,)`` or
    ``(n_heads, n_elements)``. The table entry of each atom's species, looked up by index: bitwise the one-hot
    product ``node_attrs @ table.T`` (one nonzero term per row, times exactly 1), without its [N, n_elements]
    float64 copy of the one-hot matrix (6 GiB at 9M atoms).
    """
    import torch

    table = torch.atleast_2d(torch.as_tensor(table, dtype=torch.float64)).to(node_attrs.device)
    return table[int(head)][node_attrs.argmax(dim=1)].contiguous()


def model_atomic_energies(model):
    """The model's own E0 table as float64 (exact only if the model was not converted from float64)."""
    import torch

    return model.atomic_energies_fn.atomic_energies.detach().to("cpu", torch.float64).clone()


def class_name(model):
    """The MACE class whose forward the model runs: "MACE", "ScaleShiftMACE", "PolarMACE" or None."""
    from mace.modules import models

    try:
        from mace.modules.extensions import PolarMACE
    except ImportError:
        PolarMACE = None  # noqa: N806
    kind = type(model)
    if PolarMACE is not None and kind is PolarMACE:
        return "PolarMACE"
    if kind is models.ScaleShiftMACE:
        return "ScaleShiftMACE"
    if kind is models.MACE:
        return "MACE"
    return None


def terms(model):
    """``{"resolved": (...), "unresolved": (...)}``: the energy terms per-atom energies resolve and omit."""
    name = class_name(model)
    if name is None:
        raise EnergyReductionError(f"{type(model).__name__}: no float64 energy reduction for this model class")
    resolved, unresolved = TERMS[name]
    return {"resolved": resolved, "unresolved": unresolved}


def source_digest(function):
    """sha256 of ``inspect.getsource(function)``."""
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


# Round 5 (5 November 2026). HEAD_SLICE: MACE selects each node's head with ``values[num_atoms_arange, node_heads]``
# (E0, the embedding readout, every readout). For a one-head model that is ``values[:, 0]``: the same values, but the
# advanced index's backward (``IndexBackward0``) zero-fills and radix-sorts the indices, while the slice's is a copy.
# A100 profile at 2a016fd4 (Benchmark_loup, OMAT-0 medium): about 98 us per force call in the sort, scan, histogram,
# memsets, indexing backward and gathers on [5000, 1]. Bitwise: one value per row each way. None: on for
# HEAD_SLICE_GPUS (decision 66, after the MD-step A/Bs); True / False: an A/B side's choice. Models with several
# heads keep the advanced index. MD-step A/Bs at 65bc1ee5 (SrTiO3 10x10x10 / NbCl5, B/A in both orders): RTX 5060 Ti
# 0.996 / 0.998, 0.989 / 0.993; RTX 4000 Ada 0.999 / 0.997, 0.993 / 0.992; A100-SXM4 0.992 / 0.987, 0.975 / 0.979.
HEAD_SLICE = None
HEAD_SLICE_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB"))
_HEAD_INDEX = re.compile(r"\[\n\s*num_atoms_arange, node_heads\n\s*\]")


# GRAPH_SUM: MACE sums node energies per structure with ``scatter_sum(..., data["batch"], dim_size=num_graphs)``.
# For one structure every atomic add hits one address: on the A100 three such scatters of [5000] -> [1] take 37.9 us
# plus 5.5 us of zero fills per force call (profile at 2a016fd4). With one graph this is a plain reduction. Forces and
# stress are bitwise (the gradient of either sum is exactly 1 per atom); the float64 energy's summation order changes
# (the atomics' order is unspecified anyway). None: on for GRAPH_SUM_GPUS (decision 66); True / False: an A/B side.
# MD-step A/Bs (SrTiO3 / NbCl5, b/a in both orders): A100-SXM4 0.997 / 0.998, 0.992 / 0.998 (on); RTX 5060 Ti
# 1.000 / 0.999, 0.998 / 0.998 (SrTiO3 a tie: off); RTX 4000 Ada 1.001 / 0.999, 0.999 / 0.996 (off).
GRAPH_SUM = None
GRAPH_SUM_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB",))
_GRAPH_SCATTER = re.compile(r"scatter_sum\(((?:[^()]|\([^()]*\))*data\[\"batch\"\](?:[^()]|\([^()]*\))*)\)")


def graph_sum(src, index, dim=-1, dim_size=None):
    """``scatter_sum(src, index, dim, dim_size)`` over the batch index; one graph: ``src.sum(dim, keepdim=True)``."""
    if dim_size == 1:
        return src.sum(dim=dim, keepdim=True)
    from mace.tools.scatter import scatter_sum

    return scatter_sum(src=src, index=index, dim=dim, dim_size=dim_size)


def graph_sum_default(model):
    """``GRAPH_SUM`` for ``model``: the constant when set, else on for a ``GRAPH_SUM_GPUS`` device."""
    return bool(GRAPH_SUM) if GRAPH_SUM is not None else _on_gpus(model, GRAPH_SUM_GPUS)


def graph_sum_edit(source):
    """``source`` with every ``scatter_sum`` over ``data["batch"]`` through ``graph_sum``."""
    return _GRAPH_SCATTER.sub(r"_fastmace_graph_sum(\1)", source)


def _on_gpus(model, gpus):
    import torch

    parameter = next(iter(model.parameters()), None)
    if parameter is None or parameter.device.type != "cuda":
        return False
    return torch.cuda.get_device_name(parameter.device) in gpus


def head_index(model, arange, heads):
    """The head-selection index of ``model``: ``(:, 0)`` for one head, else ``(arange, heads)`` (MACE's)."""
    if len(getattr(model, "heads", None) or ()) == 1:
        return (slice(None), 0)
    return (arange, heads)


def head_slice_default(model):
    """``HEAD_SLICE`` for ``model``: the constant when set, else on when its parameters are on a ``HEAD_SLICE_GPUS``
    device."""
    return bool(HEAD_SLICE) if HEAD_SLICE is not None else _on_gpus(model, HEAD_SLICE_GPUS)


def head_slice_edit(source):
    """``source`` with every ``[\n num_atoms_arange, node_heads\n ]`` index through ``head_index``."""
    return _HEAD_INDEX.sub("[_fastmace_head_index(self, num_atoms_arange, node_heads)]", source)


def apply_edits(source, edits, what):
    """Replace each ``(old, new)`` block of ``edits`` in ``source``; each must occur exactly once."""
    for old, new in edits:
        if source.count(old) != 1:
            raise EnergyReductionError(f"{what} is not MACE 0.3.16's; the float64 energy reduction does not apply")
        source = source.replace(old, new)
    return source


def compile_forward(source, namespace, filename):
    """Compile a rewritten ``forward`` source in ``namespace`` (a copy of its module's) and return it."""
    namespace = dict(namespace)
    namespace["_fastmace_node_e0"] = node_e0
    namespace["_fastmace_head_index"] = head_index
    namespace["_fastmace_graph_sum"] = graph_sum
    exec(compile(textwrap.dedent(source), filename, "exec"), namespace)
    return namespace["forward"]


_FORWARDS: dict = {}


def float64_forward(model, head_slice=None, graph_sum=None):
    """MACE's forward of ``model``'s class with the float64 reduction, as a plain function (cached).
    ``head_slice`` (default ``head_slice_default(model)``): the head selection as ``head_index`` (``HEAD_SLICE``);
    ``graph_sum`` (default ``graph_sum_default(model)``): the per-structure sums as ``graph_sum`` (``GRAPH_SUM``).

    ``MACE`` and ``ScaleShiftMACE`` only; PolarMACE takes it through ``fastmace.longrange.polar``.
    Raises ``EnergyReductionError`` for any other class or an unqualified source.
    """
    name = class_name(model)
    if name == "PolarMACE":
        raise EnergyReductionError("PolarMACE: the float64 reduction goes through fastmace.longrange.polar")
    if name is None:
        raise EnergyReductionError(f"{type(model).__name__}: no float64 energy reduction for this model class")
    head_slice = head_slice_default(model) if head_slice is None else bool(head_slice)
    graph_sum = graph_sum_default(model) if graph_sum is None else bool(graph_sum)
    key = (name, head_slice, graph_sum) if head_slice or graph_sum else name
    if key in _FORWARDS:
        return _FORWARDS[key]
    import mace.modules.models as models

    original = getattr(models, name).forward
    qualified = f"mace.modules.models:{name}.forward"
    digest = source_digest(original)
    if digest != SOURCE_HASHES[qualified]:
        raise EnergyReductionError(f"{qualified}: source sha256 {digest[:12]} is not the qualified "
                                   f"{SOURCE_HASHES[qualified][:12]}; the float64 energy reduction does not apply")
    source = apply_edits(inspect.getsource(original), EDITS[name], qualified)
    if head_slice:
        source = head_slice_edit(source)
    if graph_sum:
        source = graph_sum_edit(source)
    _FORWARDS[key] = compile_forward(source, vars(models), f"<fastmace float64-energy {name}.forward>")
    return _FORWARDS[key]


def install(model):
    """Bind the float64 forward to this instance (a private copy: never the caller's model)."""
    import types

    model.forward = types.MethodType(float64_forward(model), model)
    return model


def source_problems():
    """Every forward rewritten here whose source differs from ``SOURCE_HASHES`` (empty list: all match)."""
    import importlib

    problems = []
    for qualified, expected in SOURCE_HASHES.items():
        module_name, attribute = qualified.split(":", 1)
        try:
            target = importlib.import_module(module_name)
            for part in attribute.split("."):
                target = getattr(target, part)
            digest = source_digest(target)
        except (ImportError, AttributeError, OSError, TypeError) as error:
            problems.append(f"{qualified}: {type(error).__name__}: {error}")
            continue
        if digest != expected:
            problems.append(f"{qualified}: source sha256 {digest[:12]} is not the qualified {expected[:12]}")
    return problems


__all__ = [
    "EDITS", "EnergyReductionError", "NODE_E0_KEY", "SOURCE_HASHES", "TERMS", "apply_edits", "class_name",
    "compile_forward", "float64_forward", "install", "model_atomic_energies", "node_e0", "node_e0_float64",
    "source_problems", "terms",
]
