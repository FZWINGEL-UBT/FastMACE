"""PolarMACE's long-range part without host synchronisation, so its force call can be captured in CUDA graphs.

SPEC §11.1. MACE 0.3.16's PolarMACE (MACE-POLAR-1) with graph_longrange 0.4.0 synchronises with
the host on every call: the k-vector list is cut to the reciprocal-space cutoff by boolean
indexing (its length depends on the cell), the long-range blocks branch on ``torch.any(pbc)`` and
on Python booleans of GPU tensors, fill factors by boolean index assignment and copy a CPU index
tensor to the GPU. None of this can run inside a CUDA graph. ``enable()`` replaces these pieces
for one model instance (process-local; the installed packages are not modified):

- k-vectors: a superset of integer coefficients fixed on the host at ``enable()`` (from the cell,
  with optional headroom for a variable cell), in the order graph_longrange builds them, without
  the coefficients that cannot reach the cutoff for cells within the headroom. On the device each
  call computes graph_longrange's own per-cell bounds and cutoff test and marks every coefficient
  outside that set as excluded; excluded k-vectors get the batch index 1 (a graph that does not
  exist), so their cos/sin factors, and with them every density, potential and feature term, are
  exactly zero. Their k-norm is replaced by the cutoff so that every basis value stays finite.
  The included set is exactly graph_longrange's (as long as its bounds stay inside the superset
  and the cell deforms by at most the headroom; ``overflowed()`` reports otherwise), only the
  summation order differs (zeros interleaved).
- ``GTOElectrostaticFeatures.precompute_geometry`` and ``GTOElectrostaticEnergy.forward`` for fully
  periodic single-crystal batches: the same formulas with ``torch.where`` instead of boolean
  indexing; the non-periodic corrections, which are exactly zero for full PBC, are skipped.
- stress (MACE issue 1642, fixed upstream in PR 1718 after 0.3.16): with ``compute_stress`` the
  k-space features and energy use the strained positions, cell, reciprocal cell and volume, so the
  stress includes the long-range energy's cell dependence. 0.3.16 itself (also its ASE
  calculator) passes the unstrained geometry and the batch's constant reciprocal cell and volume.
  graph_longrange's GTO radial basis (``out=`` products) and k-modulus preparation (in-place fill
  of sqrt's output) are rewritten so that autograd can differentiate them by the cell. This is a
  reference correction (SPEC §12.1): there is no switch back to the uncorrected stress. The
  external-field and dipole terms keep the unstrained positions (they are zero without an
  external field).
- MACE PR 1762 (open upstream; opt-in through ``PolarConfig.fukui_regularisation`` and
  ``PolarConfig.dipole_cap``, both off by default): the regularised Fukui normalisation and the
  charge-dipole cap, by a source rewrite of 0.3.16's ``PolarMACE.forward`` (``pr1762_forward``,
  SPEC §6.3 "Source rewrites"). It acts on every model instance whose options ask for it
  (``configure()``), with or without the GPU path, so reference calculators can take the same
  model. The research engine's spin restriction is not carried over (SPEC §6.2 "Never enable").
- float64 energy (SPEC §7.3, plan task 1b-T03; ``set_float64_energy``): a second source rewrite
  of the same forward (``FLOAT64_ENERGY_EDITS``, composable with PR 1762) reduces E0, the
  interaction and local electron energies, the electrostatic energy (per-k and self terms,
  ``coulomb_energy_pbc(with_float64=True)``) and the field term into float64. Forces and stress
  still come from MACE's autograd of its own scalar, so they keep every bit.
- ``mace.modules.extensions._permute_to_e3nn_convention``, graph_longrange's external/internal
  field blocks and MACE's ``compute_total_charge_dipole_permuted`` without their per-call CPU index
  tensors (list indexing) and without a scatter that sizes its output by ``index.max()``.

Only models passed to ``enable()`` take the new path: class-level wrappers (``install()``) check a
per-instance holder, so reference calculators in the same process keep MACE's code. One graph per
batch, all three directions periodic. The PR 1711 GEMM form of ``SparseUvuTensorProduct`` is a
model rewrite and lives in ``fastmace.rewrites.polar``.

**Per-instance state.** The research engine shared one holder between every copy of a model, so
one structure's k-vectors reached every copy. Here a deep copy of an enabled model carries its own
copy of the state, and the force field enables the instance it evaluates. A copy evaluated on a
different cell trips the deformation test and sets its overflow flag (``overflowed()``).

**Process constraint (enforced by the engine and planner, plan tasks 1a-T13 and 1a-T15).** The
research engine calls ``torch._dynamo.reset()`` before it builds every PolarMACE engine: a new
structure changes the k-space state, dynamo recompiles the shared compiled model, and code
compiled for the previous structure kept that call's autograd graph alive, so the next NPT
capture failed (stale AccumulateGrad stream; 16 October 2026). This module does not reset dynamo
itself. Whoever compiles a PolarMACE force field must reset dynamo before compiling it, which
discards every compiled graph in the process (``PROCESS_CONSTRAINTS``).

**Compiled plain cuEquivariance is refused.** Compiled cuEquivariance PolarMACE returns zero forces
(energy correct; with MACE's own long-range code too, so not an effect of this path); the hybrid,
which takes the force derivative outside the compiled graph, is exact. ``route_support`` gives the
verdict and reason for the planner.

Every option comes in as an argument (``PolarConfig``); module-level objects are the call stack of
the class-level wrappers and the record of the installed patch, not switches.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import math
from dataclasses import dataclass

import numpy as np

from fastmace.config import PolarConfig
from fastmace.errors import PolarUnavailableError  # noqa: F401 (re-exported: the 1a names)

# Where the per-instance holder lives on a PolarMACE module.
HOLDER_ATTRIBUTE = "_fastmace_polar"

# The name of the process-wide patch (the class-level wrappers and e3nn's Irrep.__len__).
PATCH_NAME = "polar_longrange"

# Process constraints for the engine and planner (1a-T13 ``engine/constraints.py``, 1a-T15).
PROCESS_CONSTRAINTS = (
    "dynamo_reset_per_polar_engine: torch._dynamo.reset() before compiling each PolarMACE force field",
)

# Why compiled plain cuEquivariance is refused (SPEC §4.3 known issues).
ZERO_FORCE_REASON = ("PolarMACE: compiled plain cuEquivariance returns zero forces (energy correct; "
                     "16 October 2026); use the hybrid backend or compile_mode='none'")

# sha256 of inspect.getsource of every function this module replaces or reproduces, in MACE 0.3.16
# with graph_longrange 0.4.0 (SPEC §6.3). Another version disables the GPU path with a reason.
SOURCE_HASHES = {
    "mace.modules.extensions:PolarMACE.forward":
        "3c623c88b8dabb556cd78c4f0d800f080df5545ec9e66a275727f0891263731f",
    "mace.modules.extensions:_permute_to_e3nn_convention":
        "2d8582e070097f6e53a1c9cf6ea26d4b465ea061ecff3af8b9794858ed169a29",
    "mace.modules.utils:compute_total_charge_dipole_permuted":
        "5814d26d8ccf0f0f3328a8073a8c834f9901f7d652d6da58e1ff0ce9cd253963",
    "graph_longrange.kspace:compute_k_vectors_flat":
        "0ae20ea269d906e316822fe3058beb069f26a102d2412d669c0c92e919f0075c",
    "graph_longrange.features:GTOElectrostaticFeatures.precompute_geometry":
        "800cb31cce6b7409ad272fec88c82a80f2c84aa1b184ba35e7a7a0f505c41674",
    "graph_longrange.features:GTOElectrostaticFeatures._pbc_precompute_geometry":
        "d88aa4542016e95d72dd134497f604e6d5f842e82cfdaf4cba5070d59e798427",
    "graph_longrange.energy:GTOElectrostaticEnergy.forward":
        "cad95d2a9d5d3977a2631e790f450bf3e7b0fd177f68beb7a1b355e5dac6c3a7",
    "graph_longrange.energy:GTOElectrostaticEnergy._pbc_energy_batch":
        "394ca20ec63e1fcf99856c8145cd5bf4d748bbee5f4f2e763151a850fe3552c7",
    "graph_longrange.gto_utils:DisplacedGTOExternalFieldBlock.forward":
        "7c91c14a48cb4c9d72c115633da63b51bd2c6a01aacb025e6da01e019ae3827b",
    "graph_longrange.gto_utils:GTOInternalFieldtoFeaturesBlock.forward":
        "b700e49b1f0103a8def07fab549e831f1c1d41334218a4232b7aa24d1f5418c2",
    "graph_longrange.gto_utils:RadialIntegralDirect.forward":
        "17031f5938af69c516c6baef165fbf93408c4e83424cedf644c1d3a9825da1ef",
    "graph_longrange.gto_utils:GTOBasis._prepare_k_moduli":
        "64f77d42b3c6006cdab01311017dfff2bb72e82589c97af820355f1a794650df",
}


# --------------------------------------------------------------------------- routes and options


def is_polar(model):
    """PolarMACE (MACE-POLAR-1), by class name as the research engine checks it."""
    return type(model).__name__ == "PolarMACE"


def route_support(backend, compile_mode):
    """(supported, reason) of a PolarMACE force field on ``backend`` with ``compile_mode``.

    Compiled plain cuEquivariance returns zero forces and is refused; the hybrid (compiled or not)
    and uncompiled plain cuEquivariance are exact.
    """
    if backend not in ("hybrid", "cueq"):
        return False, f"PolarMACE: backend {backend!r} is not a GPU-engine backend (hybrid or cueq)"
    if backend == "cueq" and compile_mode not in (None, "none"):
        return False, ZERO_FORCE_REASON
    return True, "ok"


def require_route(backend, compile_mode):
    """Raise ``PolarUnavailableError`` for a refused route (``route_support``)."""
    supported, reason = route_support(backend, compile_mode)
    if not supported:
        raise PolarUnavailableError(reason)


def resolve_kspace_headroom(config, variable_cell):
    """``PolarConfig.kspace_headroom`` with ``"auto"`` resolved (switch map ``variable_cell_0_1_else_0``)."""
    if config.kspace_headroom == "auto":
        return 0.1 if variable_cell else 0.0
    return float(config.kspace_headroom)


def stabilised(config):
    """Does ``config`` ask for MACE PR 1762 (so the rewritten forward runs)?"""
    return config is not None and (config.fukui_regularisation > 0.0 or config.dipole_cap is not None)


def validate_charge_spin(numbers, charge=0, spin_multiplicity=1):
    """Check a structure's total charge and spin multiplicity for PolarMACE; returns them as ints.

    The charge is an integer, the multiplicity an integer of at least 1, and the number of
    unpaired electrons (multiplicity - 1) must have the parity of the electron count
    (sum Z - charge).
    """
    for name, value in (("charge", charge), ("spin_multiplicity", spin_multiplicity)):
        if isinstance(value, bool) or float(value) != int(value):
            raise ValueError(f"PolarMACE: {name} must be an integer, got {value!r}")
    charge, multiplicity = int(charge), int(spin_multiplicity)
    if multiplicity < 1:
        raise ValueError(f"PolarMACE: spin_multiplicity must be at least 1, got {multiplicity}")
    electrons = int(np.sum(np.asarray(numbers, dtype=np.int64))) - charge
    if electrons < 0:
        raise ValueError(f"PolarMACE: charge {charge} leaves {electrons} electrons")
    if (multiplicity - 1) % 2 != electrons % 2:
        raise ValueError(f"PolarMACE: spin multiplicity {multiplicity} is impossible with {electrons} electrons")
    return charge, multiplicity


# --------------------------------------------------------------------------- k-vectors


def _bounds(model, cell, headroom):
    """graph_longrange's k bounds (n1, n2, n3) for this cell, computed as the model computes them.

    The cell and reciprocal cell are formed in float64 and cast to the model dtype (as MACE's
    AtomicData does), then graph_longrange's formula runs in that dtype on the model's device, so
    with headroom 0 the bounds are exactly the ones its own call would use for this cell.
    """
    import torch

    dtype, device = model.kspace_cutoff.dtype, model.kspace_cutoff.device
    cell64 = torch.tensor(np.asarray(cell, dtype=np.float64))
    cell_t = cell64.to(dtype).to(device).view(1, 3, 3)
    rcell_t = (2 * torch.pi * torch.linalg.inv(cell64.mT)).to(dtype).to(device).view(1, 3, 3)
    normed = cell_t / torch.norm(cell_t, dim=-1).unsqueeze(-1)
    dots = torch.einsum("bij,bij->bi", rcell_t, normed)
    bounds = torch.ceil(model.kspace_cutoff * (1 + headroom) * torch.pow(dots, -1)).type(torch.int64)[0]
    return tuple(int(n) for n in bounds.cpu())


def _coefficients(bounds):
    """graph_longrange.kspace.compute_k_vectors_flat's integer superset for these bounds (float32 [n, 3])."""
    n1, n2, n3 = bounds

    def grid(a, b, c):
        return np.stack(np.meshgrid(a, b, c, indexing="ij"), -1).reshape(-1, 3)

    parts = (grid([0], [0], [0]), grid([0], [0], np.arange(1, n3)),
             grid([0], np.arange(1, n2), np.arange(-n3, n3)),
             grid(np.arange(1, n1), np.arange(-n2, n2), np.arange(-n3, n3)))
    return np.concatenate(parts).astype(np.float32)


class PolarState:
    """The fixed k-vector superset of one enabled model instance for one structure's cell."""

    def __init__(self, model, cell, headroom):
        import torch

        self.bounds = _bounds(model, cell, headroom)
        coefficients = _coefficients(self.bounds)
        device, dtype = model.kspace_cutoff.device, model.kspace_cutoff.dtype
        # Keep only coefficients whose k-vector can come within the cutoff: for a cell
        # cell0 (I + E) with ||E|| <= headroom, |k| >= |k0| / (1 + ||E||), so |k0| <= cutoff (1 +
        # headroom) suffices (with a margin for float32 round-off). Order kept (origin first).
        # Calls whose cell deforms further set the overflow flag (deformation check below).
        cell64 = np.asarray(cell, dtype=np.float64)
        k0 = np.linalg.norm(coefficients.astype(np.float64) @ (2 * np.pi * np.linalg.inv(cell64.T)), axis=1)
        self.superset_size = len(coefficients)
        coefficients = coefficients[k0 <= float(model.kspace_cutoff) * (1 + headroom) * (1 + 1e-5)]
        self.cell = cell64.copy()
        self.inverse_cell0 = torch.tensor(np.linalg.inv(cell64), dtype=dtype, device=device)
        self.coefficients = torch.tensor(coefficients, device=device)  # float32 as graph_longrange
        integer = self.coefficients.to(torch.int64)
        n1, n2, n3 = integer.unbind(-1)
        # Which part of graph_longrange's superset each coefficient belongs to (its bounds use them).
        self.part = torch.where((n1 == 0) & (n2 == 0) & (n3 == 0), 0,
                                torch.where((n1 == 0) & (n2 == 0), 1, torch.where(n1 == 0, 2, 3)))
        self.integer = integer
        self.k0_mask = torch.zeros(len(coefficients), dtype=dtype, device=device)
        self.k0_mask[0] = 1.0
        self.overflow = torch.zeros((), dtype=torch.bool, device=device)
        self.bounds_t = torch.tensor(self.bounds, dtype=torch.int64, device=device)
        self.headroom = headroom

    def buffers(self):
        """The device tensors a CUDA graph reads (for the force field's buffer list).

        Registered as ``force_field.polar.<name>``: the k-vector set and bounds are reset from the
        cell (``enable``); ``overflow`` is sticky, set with ``|=`` by every call whose cell leaves
        the headroom and cleared only by ``reset_overflow`` (a host decision).
        """
        return {"coefficients": self.coefficients, "integer": self.integer, "part": self.part,
                "k0_mask": self.k0_mask, "inverse_cell0": self.inverse_cell0, "bounds": self.bounds_t,
                "overflow": self.overflow}

    def info(self):
        return {"bounds": self.bounds, "k_vectors": int(len(self.coefficients)), "k_superset": self.superset_size,
                "headroom": self.headroom}


@dataclass
class PolarHolder:
    """Per-instance polar settings of one PolarMACE module: its k-space state and its PR 1762 options.

    ``state`` None means MACE's own long-range code (with the PR 1762 options still applied).
    ``float64_energy``: the forward reduces the energy into float64 (``fastmace.model.energy``).
    A deep copy of the model copies the holder and the state with it.
    """

    state: PolarState | None = None
    config: PolarConfig = PolarConfig()
    float64_energy: bool = False


class _Call:
    """One forward call of an enabled model: its state, data dict, strain flag and strained volume.

    Lives only on the ``_ACTIVE`` stack during the call. Per-call tensors are kept here rather than
    on the long-lived state: under torch.compile, attribute writes on a lasting object are replayed
    as side effects, and a reference kept to one call's autograd graph made a later CUDA-graph
    capture fail (stale AccumulateGrad stream, NVT then NPT engine of one compiled model).
    """

    def __init__(self, state, data=None, strained=False):
        self.state, self.data, self.strained, self.volume = state, data, strained, None


# The calls of enabled models in progress (innermost last): the class-level wrappers dispatch on
# it. Call context, not a switch; empty outside a forward.
_ACTIVE: list = []


def _reciprocal(cell):
    """2 pi inv(cell^T) and det(cell) of a [1, 3, 3] cell by cross products (differentiable, no host sync)."""
    import torch

    a, b, c = cell[0].unbind(0)
    crosses = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a), torch.linalg.cross(a, b)))
    volume = torch.dot(a, crosses[0])
    return (2 * torch.pi * crosses / volume).unsqueeze(0), volume.reshape(1)


def _strained_cell(call, cell_vectors):
    """The cell under the call's symmetric strain, as mace.modules.utils.get_symmetric_displacement."""
    import torch

    displacement = call.data["displacement"].view(-1, 3, 3)
    return cell_vectors + torch.matmul(cell_vectors, 0.5 * (displacement + displacement.transpose(-1, -2)))


def compute_k_vectors_fixed(call, cutoff, cell_vectors, r_cell_vectors):
    """compute_k_vectors_flat with a fixed-size output: (k, |k|^2, batch (1 = excluded), k0 mask)."""
    import torch

    if cell_vectors.size(0) != 1:
        raise ValueError("the GPU-only PolarMACE path supports one graph per batch")
    state = call.state
    call.volume = None
    if call.strained:
        # MACE 0.3.16 passes the unstrained cell and the batch's constant reciprocal cell, so the
        # stress misses the k-space energy's cell dependence (MACE issue 1642, fixed upstream in
        # PR 1718 the same way): use the strained cell, its reciprocal cell and volume.
        cell_vectors = _strained_cell(call, cell_vectors)
        r_cell_vectors, call.volume = _reciprocal(cell_vectors)
    norms = torch.norm(cell_vectors, dim=-1)
    normed = cell_vectors / norms.unsqueeze(-1)
    dots = torch.einsum("bij,bij->bi", r_cell_vectors, normed)
    max_ns = torch.ceil(cutoff * torch.pow(dots, -1)).type(torch.int64)[0]  # [3], as graph_longrange
    n1max, n2max, n3max = max_ns.unbind()
    n1, n2, n3 = state.integer.unbind(-1)
    within_3 = (n3 >= -n3max) & (n3 < n3max)
    in_superset = torch.where(
        state.part == 0, True, torch.where(
            state.part == 1, n3 < n3max, torch.where(
                state.part == 2, (n2 < n2max) & within_3,
                (n1 < n1max) & (n2 >= -n2max) & (n2 < n2max) & within_3)))
    state.overflow |= (max_ns > state.bounds_t).any()
    deformation = torch.matmul(state.inverse_cell0, cell_vectors[0]) - torch.eye(3, dtype=cell_vectors.dtype,
                                                                               device=cell_vectors.device)
    state.overflow |= torch.linalg.vector_norm(deformation) > state.headroom + 1e-5
    k_vectors = torch.einsum("ni,bij->bnj", state.coefficients.to(r_cell_vectors.dtype), r_cell_vectors)[0]
    k_norm2 = torch.einsum("ni,ni->n", k_vectors, k_vectors)
    included = in_superset & k_norm2.le(cutoff * cutoff)
    k_batch = (~included).to(torch.int64)
    k_norm2 = torch.where(included, k_norm2, (cutoff * cutoff).to(k_norm2.dtype))
    return k_vectors, k_norm2, k_batch, state.k0_mask


# --------------------------------------------------------------------------- long-range blocks


def _long_range_geometry(node_positions, volume):
    """Strained positions (as the local graph sees them) and volume when the stress is computed."""
    call = _ACTIVE[-1]
    if not call.strained:
        return node_positions, volume
    return call.data["positions"], call.volume


def _pbc_geometry(k_vectors, k_norm2, k_vector_batch, k0_mask, node_positions, batch, volume):
    import torch

    inner_products = torch.matmul(k_vectors, node_positions.t())
    mask_f = (k_vector_batch[:, None] == batch[None, :]).to(dtype=inner_products.dtype)
    cosines = torch.cos(inner_products) * mask_f
    sines = torch.sin(inner_products) * mask_f
    volume_per_k = volume.reshape(-1)[:1].expand(k_norm2.shape)
    k0 = k0_mask > 0.0
    k_factor_coulomb = torch.where(k0, 0.0, 1.0 / torch.where(k0, 1.0, k_norm2))
    return cosines, sines, volume_per_k, k_factor_coulomb


def precompute_geometry_pbc(self, k_vectors, k_norm2, k_vector_batch, k0_mask, node_positions, batch, volume,
                            pbc, force_pbc_evaluator=False):
    """GTOElectrostaticFeatures._pbc_precompute_geometry without host synchronisation (full PBC)."""
    import torch
    from graph_longrange.features import CORRECTION_MODE_PBC

    node_positions, volume = _long_range_geometry(node_positions, volume)
    cosines, sines, volume_per_k, k_factor_coulomb = _pbc_geometry(
        k_vectors, k_norm2, k_vector_batch, k0_mask, node_positions, batch, volume)
    k_factor_proj = torch.where(k0_mask > 0.0, 0.5, 1.0).to(k_norm2.dtype)
    return {"mode": "pbc", "k_vectors": k_vectors, "k_norm2": k_norm2, "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask, "volume_per_k": volume_per_k, "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj, "volumes": volume.reshape(-1), "batch": batch,
            "node_positions": node_positions, "pbc": pbc, "cosines": cosines, "sines": sines,
            "density_basis_fs": self.density_basis(k_vectors, k_norm2, k0_mask),
            "feature_basis_fs": self.feature_basis(k_vectors, k_norm2, k0_mask),
            "correction_mode": CORRECTION_MODE_PBC}


def coulomb_energy_pbc(self, k_vectors, k_norm2, k_vector_batch, k0_mask, source_feats, node_positions, batch,
                       volume, pbc, force_pbc_evaluator=False, with_float64=False):
    """GTOElectrostaticEnergy._pbc_energy_batch without host synchronisation (one graph, full PBC).

    The monopole/dipole and slab corrections are selected by torch.where only for molecules and
    slabs, so for full PBC they add exactly zero and are skipped. ``with_float64``: return
    ``(energy, energy64)``, the second with the per-k and self terms reduced into float64
    (SPEC §7.3); ``energy`` is unchanged.
    """
    import torch
    from graph_longrange.features import apply_coulomb_kernel_batch, assemble_fourier_series_batch

    node_positions, volume = _long_range_geometry(node_positions, volume)
    cosines, sines, volume_per_k, k_factor_coulomb = _pbc_geometry(
        k_vectors, k_norm2, k_vector_batch, k0_mask, node_positions, batch, volume)
    density = assemble_fourier_series_batch(source_feats=source_feats, cosines=cosines, sines=sines,
                                            density_basis_fs=self.density_basis(k_vectors, k_norm2, k0_mask),
                                            volume_per_k=volume_per_k)
    potential = apply_coulomb_kernel_batch(k_norm2=k_norm2, density=density, k_factor_coulomb=k_factor_coulomb)
    per_k = 2.0 * torch.sum(density * potential, dim=-1)
    energy = 0.5 * volume.reshape(-1) * per_k.sum().reshape(1) / (2 * math.pi) ** 6
    if with_float64:
        energy64 = 0.5 * volume.reshape(-1).double() * per_k.double().sum().reshape(1) / (2 * math.pi) ** 6
    source_feats_lm = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
    if not self.include_self_interaction:
        self_fields = self.self_interaction_terms(source_feats_lm)
        node_energies = torch.einsum("nb,nb->n", source_feats_lm, self_fields)
        self_energy = torch.zeros(1, dtype=node_energies.dtype, device=node_energies.device)
        self_energy.index_add_(0, batch, node_energies)
        energy = energy - self_energy * 0.5
        if with_float64:
            self_energy64 = torch.zeros(1, dtype=torch.float64, device=node_energies.device)
            self_energy64.index_add_(0, batch, node_energies.double())
            energy64 = energy64 - self_energy64 * 0.5
    if with_float64:
        return energy, energy64
    return energy


def coulomb_energy_float64(module, **kwargs):
    """``(energy, energy64)`` of a ``GTOElectrostaticEnergy``: the energy MACE differentiates, and float64.

    On the GPU-only path the per-k and self terms are reduced into float64. MACE's own long-range
    code (an instance without a k-space state, as in a reference calculator) returns its total,
    upcast.
    """
    if _ACTIVE:
        return coulomb_energy_pbc(module, with_float64=True, **kwargs)
    energy = module(**kwargs)
    return energy, energy.double()


def radial_direct_forward(self, k_mods):
    """graph_longrange RadialIntegralDirect.forward without out= (which autograd cannot differentiate)."""
    import torch

    exp_term = torch.exp(-0.5 * (k_mods * k_mods).unsqueeze(-1) * self.sigma2)
    if self.max_l == 0:
        return (self.pref0 * exp_term).unsqueeze(-1)
    return torch.stack((self.pref0 * exp_term, self.pref1 * k_mods.unsqueeze(-1) * exp_term), dim=-1)


def prepare_k_moduli(self, k_norm2, k0_mask):
    """GTOBasis._prepare_k_moduli without the in-place fill of sqrt's output; finite gradient at k = 0."""
    import torch

    k0 = k0_mask > 0.0
    return torch.where(k0, 0.0, torch.sqrt(torch.clamp_min(torch.where(k0, 1.0, k_norm2), 0.0)))


def _reorder_field(node_fields):
    """node_fields[:, [0, 3, 1, 2]] without the per-call CPU index tensor."""
    import torch

    return torch.cat((node_fields[:, 0:1], node_fields[:, 3:4], node_fields[:, 1:3]), dim=1)


def external_field_forward(self, batch, positions, field):
    """graph_longrange DisplacedGTOExternalFieldBlock.forward with _reorder_field."""
    import torch

    assert field.dim() == 2
    node_fields = torch.index_select(field, 0, batch)
    potential_from_displacement = torch.einsum("bi,bi->b", positions, node_fields[:, 1:].clone())
    node_fields[:, 0] += potential_from_displacement
    return torch.einsum("pf,nf->np", self.matrix, _reorder_field(node_fields))


def internal_field_forward(self, batch, positions, node_fields):
    """graph_longrange GTOInternalFieldtoFeaturesBlock.forward with _reorder_field."""
    import torch

    assert node_fields.dim() == 2
    return torch.einsum("pf,nf->np", self.matrix, _reorder_field(node_fields))


def total_charge_dipole(density_coefficients, positions, batch, num_graphs):
    """mace.modules.utils.compute_total_charge_dipole_permuted without host synchronisation.

    The original reorders with a Python list (CPU index tensor) and sums the charge without
    dim_size (index.max() on the host).
    """
    import torch
    from mace.tools.scatter import scatter_sum

    dipole = scatter_sum(src=positions * density_coefficients[:, :1], index=batch.unsqueeze(-1), dim=0,
                         dim_size=num_graphs)
    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(src=density_coefficients[..., 1:4], index=batch, dim=-2, dim_size=num_graphs)
        dipole = dipole + torch.cat((dipole_p[..., 2:3], dipole_p[..., 0:2]), dim=-1)  # [2, 0, 1]
    total_charge = scatter_sum(src=density_coefficients[:, 0], index=batch, dim=-1, dim_size=num_graphs)
    return total_charge, dipole


def _permute_to_e3nn_convention(x):
    import torch

    return torch.cat((x[..., 1:3], x[..., 0:1]), dim=-1)


# --------------------------------------------------------------------------- MACE PR 1762


def normalise_fukui(fukui_sources, batch, num_graphs, regularisation=0.0):
    """PolarMACE._normalise_fukui of MACE PR 1762 with r = ``regularisation``.

    r = 0: MACE 0.3.16's normalisation (division by the signed per-graph sum, in float64, with
    sum 0 replaced by 1). r > 0: s / (s^2 + (r sum|f|)^2) with the withheld weight spread evenly;
    the weights sum to one, are smooth through s = 0 and bounded by about 1/(2r).
    """
    import torch
    from mace.tools.scatter import scatter_sum

    r = float(regularisation)
    if r > 0.0:
        f = fukui_sources.double()
        count = scatter_sum(src=torch.ones_like(f[:, :1]), index=batch, dim=0, dim_size=num_graphs)[batch]
        mean_sq = scatter_sum(src=f * f, index=batch, dim=0, dim_size=num_graphs)[batch] / count
        empty = mean_sq == 0
        f = torch.where(empty, torch.ones_like(f), f)
        mean_sq = torch.where(empty, torch.ones_like(mean_sq), mean_sq)
        norm = scatter_sum(src=f, index=batch, dim=0, dim_size=num_graphs)[batch]
        smooth_abs = torch.sqrt(f * f + 1e-6 * mean_sq)
        total_abs = scatter_sum(src=smooth_abs, index=batch, dim=0, dim_size=num_graphs)[batch]
        eps2 = (r * total_abs) ** 2
        denom = norm * norm + eps2
        weights = f * norm / denom + eps2 / denom / count
        return torch.where(empty, 1.0 / count, weights).to(fukui_sources.dtype)
    norm = scatter_sum(src=fukui_sources.double(), index=batch, dim=0, dim_size=num_graphs)[batch]
    norm = norm.to(fukui_sources.dtype)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return fukui_sources / norm


def cap_charge_dipole(spin_charge_density, cap=None):
    """PolarMACE._cap_charge_dipole of MACE PR 1762 with ``cap`` = (onset, ceiling) in e Angstrom; None: identity.

    Each atom's l = 1 charge dipole: identity up to the onset, then a tanh to the ceiling; monopoles
    and the spin density unchanged; atoms below the onset keep their input bit for bit.
    """
    import torch

    if cap is None:
        return spin_charge_density
    onset, ceiling = cap
    density = spin_charge_density.double()
    charge = density[:, 0, :] + density[:, 1, :]
    spin = density[:, 0, :] - density[:, 1, :]
    dipole = charge[:, 1:4]
    sq = (dipole * dipole).sum(dim=-1, keepdim=True)
    safe = torch.sqrt(torch.clamp(sq, min=onset ** 2))
    width = ceiling - onset
    capped = onset + width * torch.tanh((safe - onset) / width)
    charge = torch.cat((charge[:, :1], dipole * capped / safe, charge[:, 4:]), dim=-1)
    rebuilt = 0.5 * torch.stack((charge + spin, charge - spin), dim=1)
    return torch.where((sq > onset ** 2).unsqueeze(1), rebuilt.to(spin_charge_density.dtype),
                       spin_charge_density)


def _instance_config(model):
    holder = model.__dict__.get(HOLDER_ATTRIBUTE)
    return PolarConfig() if holder is None else holder.config


def _model_normalise_fukui(model, fukui_sources, batch, num_graphs):
    return normalise_fukui(fukui_sources, batch, num_graphs, _instance_config(model).fukui_regularisation)


def _model_cap_charge_dipole(model, spin_charge_density):
    return cap_charge_dipole(spin_charge_density, _instance_config(model).dipole_cap)


# MACE 0.3.16's PolarMACE.forward blocks that PR 1762 changes (each must occur exactly once). The
# replacements pass the model, whose holder carries the options.
PR1762_EDITS = (
    ("""        fukui_sources = self.fukui_source_map(fukui_input)
        fukui_norm = scatter_sum(
            src=fukui_sources.double(),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )[data["batch"]].to(vectors.dtype)
        fukui_norm = torch.where(
            fukui_norm == 0, torch.ones_like(fukui_norm), fukui_norm
        )
        fukui_sources = fukui_sources / fukui_norm
""", """        fukui_sources = _fastmace_normalise_fukui(
            self, self.fukui_source_map(fukui_input), data["batch"], num_graphs
        )
"""),
    ("""        ] * ((Q_m_S / 2) - pred_total_charges_0[:, 1])
""", """        ] * ((Q_m_S / 2) - pred_total_charges_0[:, 1])
        spin_charge_density = _fastmace_cap_charge_dipole(self, spin_charge_density)
"""),
    ("""            fukui_norm2 = scatter_sum(
                src=current_fukui_sources.double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            fukui_norm2 = torch.where(
                fukui_norm2 == 0, torch.ones_like(fukui_norm2), fukui_norm2
            )
            current_fukui_sources = current_fukui_sources / fukui_norm2
""", """            current_fukui_sources = _fastmace_normalise_fukui(
                self, current_fukui_sources, data["batch"], num_graphs
            )
"""),
    ("""            ] + current_fukui_sources[:, 1] * ((Q_m_S / 2) - pred_total_charges[:, 1])
""", """            ] + current_fukui_sources[:, 1] * ((Q_m_S / 2) - pred_total_charges[:, 1])
            spin_charge_density = _fastmace_cap_charge_dipole(self, spin_charge_density)
"""),
)


def pr1762_forward(original_forward):
    """MACE 0.3.16's PolarMACE.forward with MACE PR 1762's two changes, built from its source.

    The four blocks of ``PR1762_EDITS`` are replaced in the installed source (refused unless each
    occurs exactly once, i.e. unless this is the 0.3.16 code the port was written for) and the
    result is compiled in mace.modules.extensions' namespace, so it calls the same (wrapped)
    module functions as the original.
    """
    import textwrap

    import mace.modules.extensions as extensions

    source = inspect.getsource(original_forward)
    for old, new in PR1762_EDITS:
        if source.count(old) != 1:
            raise PolarUnavailableError("PolarMACE.forward is not MACE 0.3.16's; the PR 1762 port does not apply")
        source = source.replace(old, new)
    namespace = dict(vars(extensions))
    namespace.update(_fastmace_normalise_fukui=_model_normalise_fukui,
                     _fastmace_cap_charge_dipole=_model_cap_charge_dipole)
    exec(compile(textwrap.dedent(source), "<fastmace PR 1762 PolarMACE.forward>", "exec"), namespace)
    return namespace["forward"]


# MACE 0.3.16's PolarMACE.forward blocks that the float64 energy reduction (SPEC §7.3,
# fastmace.model.energy) changes. Disjoint from PR1762_EDITS, so both apply to one source. The
# electrostatic energy MACE differentiates is unchanged; its float64 twin goes into the energy.
FLOAT64_ENERGY_EDITS = (
    ("""        electro_energy = self.coulomb_energy(
""", """        electro_energy, electro_energy64 = _fastmace_coulomb_energy_float64(
            self.coulomb_energy,
"""),
    ("""        forces, virials, stress, hessian, edge_forces = get_outputs(
""", """        node_energy64 = _fastmace_node_e0(data, node_e0) + node_inter_es.double()
        if getattr(self, "add_local_electron_energy", False):
            node_energy64 = node_energy64 + local_q_e.double()
        unresolved64 = electro_energy64 + torch.sum(
            external_potential[:, 1:].double() * total_dipole.double(), dim=-1
        )
        energy64 = scatter_sum(node_energy64, data["batch"], dim=-1, dim_size=num_graphs) + unresolved64
        forces, virials, stress, hessian, edge_forces = get_outputs(
"""),
    ("""            "energy": total_energy,
""", """            "energy": energy64,
            "node_energy_float64": node_energy64,
            "energy_unresolved_float64": unresolved64,
"""),
)


def polar_forward(original_forward, stabilise=False, float64_energy=False):
    """MACE 0.3.16's PolarMACE.forward with PR 1762 (``stabilise``) and/or the float64 energy reduction.

    Neither: ``original_forward`` itself. Otherwise built from its source like ``pr1762_forward``.
    An unqualified source raises ``PolarUnavailableError`` (PR 1762) or
    ``fastmace.model.energy.EnergyReductionError``.
    """
    if not float64_energy:
        return pr1762_forward(original_forward) if stabilise else original_forward
    import mace.modules.extensions as extensions

    from fastmace.model import energy

    source = inspect.getsource(original_forward)
    if stabilise:
        for old, new in PR1762_EDITS:
            if source.count(old) != 1:
                raise PolarUnavailableError("PolarMACE.forward is not MACE 0.3.16's; the PR 1762 port does not apply")
            source = source.replace(old, new)
    source = energy.apply_edits(source, FLOAT64_ENERGY_EDITS, "PolarMACE.forward")
    namespace = dict(vars(extensions))
    namespace.update(_fastmace_normalise_fukui=_model_normalise_fukui,
                     _fastmace_cap_charge_dipole=_model_cap_charge_dipole,
                     _fastmace_coulomb_energy_float64=coulomb_energy_float64)
    return energy.compile_forward(source, namespace, "<fastmace PolarMACE.forward, float64 energy>")


# --------------------------------------------------------------------------- process-wide install


def source_digest(function):
    """sha256 of ``inspect.getsource(function)`` (the hash SPEC §6.1 records)."""
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _resolve(qualified):
    import importlib

    module_name, attribute = qualified.split(":", 1)
    target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


# What install() replaced: "module:attribute" -> MACE's or graph_longrange's object.
_ORIGINALS: dict = {}


def _original(qualified):
    """The unwrapped function behind ``qualified`` (the one in the package before install())."""
    return _ORIGINALS.get(qualified) or _resolve(qualified)


def source_problems():
    """Every replaced function whose source differs from ``SOURCE_HASHES`` (empty list: all match)."""
    problems = []
    for qualified, expected in SOURCE_HASHES.items():
        try:
            digest = source_digest(_original(qualified))
        except (ImportError, AttributeError, OSError, TypeError) as error:
            problems.append(f"{qualified}: {type(error).__name__}: {error}")
            continue
        if digest != expected:
            problems.append(f"{qualified}: source sha256 {digest[:12]} is not the qualified {expected[:12]}")
    return problems


def allow_irrep_guards():
    """Let torch.compile guard e3nn Irrep values: Irrep (an (l, p) tuple) raises in __len__.

    Dynamo's sequence-length guards call len() on the Irrep entries of the Irreps lists that
    graph_longrange's and MACE's field blocks read in forward (e.g. ``features_irreps.dim``), so a
    compiled PolarMACE fails while building guards. Return the tuple's real length, 2
    (process-local; nothing in e3nn relies on the exception). Part of ``install()``.
    """
    from e3nn.o3 import _irreps

    if getattr(_irreps.Irrep.__len__, "_fastmace_polar", False):
        return
    if "e3nn.o3._irreps:Irrep.__len__" not in _ORIGINALS:
        _ORIGINALS["e3nn.o3._irreps:Irrep.__len__"] = _irreps.Irrep.__len__

    def length(self):
        return 2

    length._fastmace_polar = True
    _irreps.Irrep.__len__ = length


# Set by install(): the patch record (name, value, targets, detail), or None before.
_INSTALLED: dict = {}


def installed():
    """The record of the process-wide patch (as ``fastmace._patches.PatchRecord`` fields), or None."""
    return dict(_INSTALLED) if _INSTALLED else None


def install():
    """Class-level wrappers that take the GPU-only path only for enabled model instances (idempotent).

    Refuses (``PolarUnavailableError``) when graph_longrange is missing or when a replaced function's
    source differs from ``SOURCE_HASHES``; nothing is patched then.
    """
    if _INSTALLED:
        return installed()
    try:
        import graph_longrange  # noqa: F401
    except ImportError as error:
        raise PolarUnavailableError(f"graph_longrange is not installed ({error})") from error
    problems = source_problems()
    if problems:
        raise PolarUnavailableError("unqualified MACE/graph_longrange source: " + "; ".join(problems))
    import mace.modules.extensions as extensions
    from graph_longrange import energy, features, gto_utils

    allow_irrep_guards()
    original_forward = extensions.PolarMACE.forward
    original_kvectors = extensions.compute_k_vectors_flat
    original_permute = extensions._permute_to_e3nn_convention
    original_charge_dipole = extensions.compute_total_charge_dipole_permuted
    original_geometry = features.GTOElectrostaticFeatures.precompute_geometry
    original_energy = energy.GTOElectrostaticEnergy.forward
    original_external = gto_utils.DisplacedGTOExternalFieldBlock.forward
    original_internal = gto_utils.GTOInternalFieldtoFeaturesBlock.forward
    original_radial = gto_utils.RadialIntegralDirect.forward
    original_moduli = gto_utils.GTOBasis._prepare_k_moduli
    rewritten = {}  # (PR 1762, float64 energy) -> forward, built on first use

    def model_forward(holder):
        key = (stabilised(holder.config), bool(holder.float64_energy))
        if key == (False, False):
            return original_forward
        if key not in rewritten:
            rewritten[key] = polar_forward(original_forward, *key)
        return rewritten[key]

    def forward(self, data, *args, **kwargs):
        holder = self.__dict__.get(HOLDER_ATTRIBUTE)
        if holder is None:
            return original_forward(self, data, *args, **kwargs)
        if holder.state is None:
            return model_forward(holder)(self, data, *args, **kwargs)
        strained = bool(kwargs.get("compute_stress") or kwargs.get("compute_virials"))
        if strained and data.get("displacement") is None:
            # The strain leaf the stress differentiates by (prepare_graph uses it); the long-range
            # part needs it too.
            import torch
            data["displacement"] = torch.zeros((data["ptr"].numel() - 1, 3, 3), dtype=data["positions"].dtype,
                                               device=data["positions"].device, requires_grad=True)
        _ACTIVE.append(_Call(holder.state, data, strained))
        try:
            return model_forward(holder)(self, data, *args, **kwargs)
        finally:
            _ACTIVE.pop()

    def k_vectors(cutoff, cell_vectors, r_cell_vectors):
        if not _ACTIVE:
            return original_kvectors(cutoff, cell_vectors, r_cell_vectors)
        return compute_k_vectors_fixed(_ACTIVE[-1], cutoff, cell_vectors, r_cell_vectors)

    def permute(x):
        return _permute_to_e3nn_convention(x) if _ACTIVE else original_permute(x)

    def geometry(self, *args, **kwargs):
        return (precompute_geometry_pbc if _ACTIVE else original_geometry)(self, *args, **kwargs)

    def coulomb(self, *args, **kwargs):
        return (coulomb_energy_pbc if _ACTIVE else original_energy)(self, *args, **kwargs)

    def external(self, *args, **kwargs):
        return (external_field_forward if _ACTIVE else original_external)(self, *args, **kwargs)

    def internal(self, *args, **kwargs):
        return (internal_field_forward if _ACTIVE else original_internal)(self, *args, **kwargs)

    def charge_dipole(*args, **kwargs):
        return (total_charge_dipole if _ACTIVE else original_charge_dipole)(*args, **kwargs)

    def radial(self, *args, **kwargs):
        return (radial_direct_forward if _ACTIVE else original_radial)(self, *args, **kwargs)

    def moduli(self, *args, **kwargs):
        return (prepare_k_moduli if _ACTIVE else original_moduli)(self, *args, **kwargs)

    replacements = (
        (gto_utils.RadialIntegralDirect, "forward", radial),
        (gto_utils.GTOBasis, "_prepare_k_moduli", moduli),
        (extensions, "compute_total_charge_dipole_permuted", charge_dipole),
        (gto_utils.DisplacedGTOExternalFieldBlock, "forward", external),
        (gto_utils.GTOInternalFieldtoFeaturesBlock, "forward", internal),
        (extensions.PolarMACE, "forward", forward),
        (extensions, "compute_k_vectors_flat", k_vectors),
        (extensions, "_permute_to_e3nn_convention", permute),
        (features.GTOElectrostaticFeatures, "precompute_geometry", geometry),
        (energy.GTOElectrostaticEnergy, "forward", coulomb),
    )
    targets = ["e3nn.o3._irreps:Irrep.__len__"]
    for owner, attribute, replacement in replacements:
        module = owner.__name__ if isinstance(owner, type(extensions)) else owner.__module__
        name = attribute if isinstance(owner, type(extensions)) else f"{owner.__name__}.{attribute}"
        key = f"{module}:{name}"
        _ORIGINALS.setdefault(key, getattr(owner, attribute))
        setattr(owner, attribute, replacement)
        targets.append(key)
    _INSTALLED.update(name=PATCH_NAME, value=True, targets=tuple(targets),
                      detail="class-level PolarMACE wrappers; GPU-only path for enabled instances only")
    return installed()


# --------------------------------------------------------------------------- per-instance API


def _holder(model, create=False):
    holder = model.__dict__.get(HOLDER_ATTRIBUTE)
    if holder is None and create:
        holder = PolarHolder()
        object.__setattr__(model, HOLDER_ATTRIBUTE, holder)
    return holder


def configure(model, config=None):
    """Set the PR 1762 options of one PolarMACE instance (``PolarConfig``; None: the released model).

    Works with or without the GPU-only path, so a reference calculator's model can take the same
    options. Installs the class-level wrappers.
    """
    if not is_polar(model):
        raise TypeError(f"{type(model).__name__} is not PolarMACE")
    config = PolarConfig() if config is None else config
    if not isinstance(config, PolarConfig):
        raise TypeError(f"config must be a PolarConfig, got {type(config).__name__}")
    install()
    _holder(model, create=True).config = config
    return config


def enable(model, cell, config=None, *, variable_cell=False, pbc=(True, True, True)):
    """Let this PolarMACE instance run its long-range part without host synchronisation.

    ``cell``: the structure's cell (Angstrom). ``config.kspace_headroom`` widens the k-vector
    superset for a variable cell (``"auto"``: 0.1 with ``variable_cell``, else 0). Returns the
    superset's bounds, sizes and headroom (for provenance).

    Process constraint: compile this model only after ``torch._dynamo.reset()`` (module
    docstring, ``PROCESS_CONSTRAINTS``); the engine (1a-T13) and planner (1a-T15) enforce it.
    """
    if not is_polar(model):
        raise TypeError(f"{type(model).__name__} is not PolarMACE")
    if not all(bool(p) for p in pbc):
        raise ValueError("the GPU-only PolarMACE path needs all three directions periodic")
    config = configure(model, config)
    headroom = resolve_kspace_headroom(config, variable_cell)
    state = _holder(model).state = PolarState(model, cell, headroom)
    return state.info()


def disable(model):
    """Back to MACE's own long-range code for this instance (the PR 1762 options stay)."""
    holder = _holder(model)
    if holder is not None:
        holder.state = None


def get_state(model):
    """The instance's ``PolarState``, or None."""
    holder = _holder(model)
    return None if holder is None else holder.state


def set_state(model, state):
    """Switch an instance between a state from ``enable()`` and None (MACE's long-range code)."""
    install()
    _holder(model, create=True).state = state


def set_float64_energy(model, enabled=True):
    """Let this PolarMACE instance reduce its energy into float64 (SPEC §7.3); installs the wrappers.

    Works with or without the GPU-only path. Call it after the rewrites' self-checks, which
    compare against MACE's own energy.
    """
    if not is_polar(model):
        raise TypeError(f"{type(model).__name__} is not PolarMACE")
    install()
    _holder(model, create=True).float64_energy = bool(enabled)


def get_config(model):
    """The instance's ``PolarConfig`` (the default when none was set)."""
    return _instance_config(model)


def overflowed(model):
    """True if graph_longrange's own k bounds left the superset during any call (host read)."""
    state = get_state(model)
    return bool(state is not None and state.overflow.item())


def reset_overflow(model):
    """Clear the overflow flag (after a re-plan)."""
    state = get_state(model)
    if state is not None:
        state.overflow.zero_()


@contextlib.contextmanager
def no_sync_check():
    """Raise on any host synchronisation inside the block (torch sync debug mode)."""
    import torch

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


__all__ = [
    "FLOAT64_ENERGY_EDITS", "HOLDER_ATTRIBUTE", "PATCH_NAME", "PR1762_EDITS", "PROCESS_CONSTRAINTS", "SOURCE_HASHES",
    "ZERO_FORCE_REASON", "PolarHolder", "PolarState", "PolarUnavailableError", "cap_charge_dipole",
    "compute_k_vectors_fixed", "configure", "coulomb_energy_float64", "coulomb_energy_pbc", "disable", "enable",
    "get_config", "get_state", "install", "installed", "is_polar", "no_sync_check", "normalise_fukui",
    "overflowed", "polar_forward", "pr1762_forward", "require_route", "reset_overflow", "resolve_kspace_headroom",
    "route_support", "set_float64_energy", "set_state", "source_digest", "source_problems", "stabilised",
    "validate_charge_spin",
]
