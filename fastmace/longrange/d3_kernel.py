"""D3(BJ) dispersion on the GPU in three passes with analytic derivatives (SPEC §11.2, plan task 3b-T03).

The input is a half list: undirected pairs ``(i, j, n)`` with ``i < j``, plus each self-image pair
``(i, i, n)`` once. The pair vector is ``x_j + n @ cell - x_i``. Unused slots of a padded list hold
``i = j = N``. ``prepare`` sorts the list by ``i`` and builds a permutation sorted by ``j``, with row
offsets for both. It runs at list builds, is capturable, and keeps the shapes static. Each pair
then belongs to the i-segment of one atom and the j-segment of another.

One Triton program per atom walks both of its segments and sums in float64 registers. There are no
atomics, so the result is deterministic and a captured replay equals the eager call bit for bit.
Every pair term is computed twice, once for each atom, from the same expressions in the same
order (pair vector ``x_j + shift - x_i``, C6 from the ``(i, j)`` table), so both atoms get exactly
opposite pair forces. The energy and the virial are summed only on the i side, so each pair counts
once. A pair over the cutoff stays in the list and is masked in registers (``r² ≤ r_c²``,
``where``), with no compaction.

- Pass 1: coordination numbers ``CN_a = Σ_b 1 / (1 + exp(-k1 (R_cov,ab / r_ab - 1)))`` over
  ``r ≤ cn_cutoff``. A self-image pair adds to its atom twice, once from each segment, as the full
  sum over ``+n`` and ``-n`` does.
- Pass 2: the energy ``E = -Σ (s6 C6 / (r⁶ + R0⁶) + s8 C8 / (r⁸ + R0⁸))`` over
  ``r ≤ pair_cutoff``, with ``R0 = a1 sqrt(C8/C6) + a2`` and ``C8 = 3 C6 r2r4_a r2r4_b``. The
  direct forces and virial at fixed CN. And ``dE/dCN_a`` over every pair of atom a.
- Pass 3, after the kernel boundary that completes ``dE/dCN``: the chain term
  ``(dE/dCN_a + dE/dCN_b) ∂CN/∂r_ab`` into the forces and virial of both atoms. ``torch-dftd``
  gets the same term from autograd. Leaving it out keeps the energy but changes the forces.

C6 is interpolated as ``torch-dftd`` 0.5.1 does it: a softmax over the reference pairs with
``c6 > 0``, weighted by ``k3 ((CN_a - CN_A)² + (CN_b - CN_B)²)``.

Precision (plan phase 3b, conductor's choice 3): the pair terms are computed in ``dtype``, the
force precision. CN, ``dE/dCN``, energy, forces and virial are summed in float64. Forces come back
in ``dtype``, energy and virial in float64. Units are ``torch-dftd``'s: lengths are divided by the
record's ``autoang``, and energies multiplied by its ``autoev``.

The reference tables (C6 references, covalent radii, ``r2r4``) are ``torch-dftd``'s
``dftd3_params.npz`` (the ``[d3]`` extra), indexed by Z and read through ``numbers``, so species
events need no refresh. On the CPU, or without Triton, the same three passes run in PyTorch with
``index_add``.
"""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path

import numpy as np
import torch

K1 = 16.0  # torch-dftd d3_k1: steepness of the coordination-number counting function
K3 = -4.0  # torch-dftd d3_k3: Gaussian weight of the C6 reference interpolation
REFERENCES = 32  # the 5 x 5 C6 reference pairs, padded to a power of two (padding has c6 = 0)
BLOCK = 128  # pairs per loop iteration of a program (fixed for every crystal and GPU)

# The Becke-Johnson parameters of torch-dftd 0.5.1 (``get_dftd3_default_params("bj", xc)``), used
# while the record does not carry them itself. r2scan: doi 10.1063/5.0041008.
_BJ_PARAMETERS = {
    "r2scan": {"s6": 1.0, "s8": 0.7898, "a1": 0.4948, "a2": 5.7308},
    "pbe": {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
}


@cache
def reference_tables() -> dict[str, np.ndarray]:
    """``torch-dftd``'s D3 tables: ``c6ab`` (95, 95, 5, 5, 3), ``rcov`` (95,), ``r2r4`` (95,), by Z."""
    spec = importlib.util.find_spec("torch_dftd")
    if spec is None or spec.origin is None:
        raise ImportError("D3 needs torch-dftd's parameter tables: pip install 'fastmace[d3]' (torch-dftd 0.5.1)")
    with np.load(Path(spec.origin).parent / "nn" / "params" / "dftd3_params.npz") as data:
        return {name: np.array(data[name], dtype=np.float64) for name in ("c6ab", "rcov", "r2r4")}


def bj_parameters(record) -> dict[str, float]:
    """``s6``, ``s8``, ``a1``, ``a2`` of the record: its own fields, else ``torch-dftd``'s set for its functional."""
    names = ("s6", "s8", "a1", "a2")
    if all(getattr(record, name, None) is not None for name in names):
        return {name: float(getattr(record, name)) for name in names}
    if record.functional not in _BJ_PARAMETERS:
        raise ValueError(f"no D3(BJ) parameters for functional {record.functional!r}")
    return dict(_BJ_PARAMETERS[record.functional])


class D3Kernel:
    """D3(BJ) energy, forces and virial of one structure, on a half pair list.

    - ``record``: an ``fm.D3`` (functional or explicit BJ parameters, cutoffs, unit constants);
    - ``numbers``: atomic numbers (N,), an integer tensor on ``device`` (kept by reference, so species
      events may change it in place) or anything ``numpy`` converts;
    - ``dtype``: the force precision of the pair terms (float32 or float64);
    - ``chain``: include pass 3. Only for the test that shows the chain term matters.

    ``prepare(pair_i, pair_j, images)`` after each list build, then ``kernel(positions, cell)`` returns
    ``(energy 0-d float64 eV, forces (N, 3) dtype eV/Å, virial (3, 3) float64 eV)``. The virial is
    ``-dE/dε`` (MACE's convention). The ASE stress is ``-virial / volume``.
    """

    def __init__(self, record, numbers, *, device, dtype=torch.float64, chain=True):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"D3 pair terms run in float32 or float64, got {dtype}")
        if record.damping != "bj" or record.three_body or record.smoothing != "none":
            raise ValueError("the D3 kernel implements Becke-Johnson damping without ATM or smoothing")
        self.device = torch.device(device)
        self.dtype = dtype
        self.chain = bool(chain)
        self.autoang = float(record.autoang)
        self.autoev = float(record.autoev)
        self.pair_cutoff2 = float(record.pair_cutoff_bohr) ** 2
        self.cn_cutoff2 = float(record.cn_cutoff_bohr) ** 2
        parameters = bj_parameters(record)
        self.s6, self.s8 = parameters["s6"], parameters["s8"]

        tables = reference_tables()
        count = tables["rcov"].shape[0]
        if not isinstance(numbers, torch.Tensor):
            numbers = torch.as_tensor(np.asarray(numbers, dtype=np.int64), device=self.device)
        if numbers.device != self.device or numbers.dtype not in (torch.int32, torch.int64) or numbers.ndim != 1:
            raise ValueError(f"numbers must be a 1-d integer tensor on {self.device}")
        if numbers.numel() and (int(numbers.min()) < 1 or int(numbers.max()) >= count):
            raise ValueError(f"D3 has reference data for Z = 1 to {count - 1}")
        # Tables by Z for every element, read through ``numbers`` in the kernels: a species event that
        # changes ``numbers`` in place needs no refresh (conductor's choice 5), and no address moves.
        references = np.zeros((count, count, REFERENCES, 3))
        references[:, :, :25, :] = tables["c6ab"].reshape(count, count, 25, 3)
        r2r4 = tables["r2r4"]
        c8_factor = 3.0 * r2r4[:, None] * r2r4[None, :]
        r0 = parameters["a1"] * np.sqrt(c8_factor) + parameters["a2"]

        def put(array, dtype=dtype):
            return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=self.device)

        self.n_atoms = numbers.shape[0]
        self.species = numbers  # by reference, at its fixed address
        self.rcov = put(tables["rcov"])
        self.references = put(references)
        self.c8_factor = put(c8_factor)
        self.r0_6 = put(r0 ** 6)
        self.r0_8 = put(r0 ** 8)
        # float64 scalars for the kernels (Triton would pass Python floats as float32)
        self.scalars = put([self.autoang, self.pair_cutoff2, self.cn_cutoff2, self.s6, self.s8], torch.float64)
        self._prepared = False

    # ----------------------------------------------------------------------------------- list
    def prepare(self, pair_i, pair_j, images):
        """Take a half list: ``pair_i``, ``pair_j`` (P,), integer ``images`` (P, 3); unused slots ``i = j = N``."""
        n = self.n_atoms
        pair_i = torch.as_tensor(pair_i, device=self.device).to(torch.int64).reshape(-1)
        pair_j = torch.as_tensor(pair_j, device=self.device).to(torch.int64).reshape(-1)
        images = torch.as_tensor(images, device=self.device).to(torch.int32).reshape(-1, 3)
        if pair_i.shape != pair_j.shape or images.shape[0] != pair_i.shape[0]:
            raise ValueError("pair_i, pair_j and images must have one row per pair")
        sentinel = torch.full((1,), n, dtype=torch.int64, device=self.device)
        pair_i = torch.cat([pair_i, sentinel])  # never empty, so the kernels always get real buffers
        pair_j = torch.cat([pair_j, sentinel])
        images = torch.cat([images, images.new_zeros((1, 3))])
        order = torch.argsort(pair_i, stable=True)
        self.pair_i = pair_i[order].to(torch.int32).contiguous()
        self.pair_j = pair_j[order].to(torch.int32).contiguous()
        self.images = images[order].contiguous()
        by_j = torch.argsort(self.pair_j.to(torch.int64), stable=True)
        atoms = torch.arange(n + 1, device=self.device, dtype=torch.int32)
        self.row = torch.searchsorted(self.pair_i, atoms).to(torch.int32).contiguous()
        self.by_j = by_j.to(torch.int32).contiguous()
        self.col = torch.searchsorted(self.pair_j[by_j], atoms).to(torch.int32).contiguous()
        self._prepared = True

    # ----------------------------------------------------------------------------- evaluation
    def __call__(self, positions, cell):
        if not self._prepared:
            raise RuntimeError("D3Kernel.prepare(pair_i, pair_j, images) must run before an evaluation")
        if positions.is_cuda and _triton_available():
            return self._triton(positions, cell)
        return self._torch(positions, cell)

    def _finish(self, energy_atoms, forces, virial_atoms):
        force_unit = self.autoev / self.autoang
        energy = energy_atoms.sum() * self.autoev
        virial = virial_atoms.sum(0).reshape(3, 3) * self.autoev
        return energy, (forces * force_unit).to(self.dtype), virial

    def _torch(self, positions, cell):
        n, dt, f64 = self.n_atoms, self.dtype, torch.float64
        valid = self.pair_i < n
        i = torch.where(valid, self.pair_i, 0).to(torch.int64)
        j = torch.where(valid, self.pair_j, 0).to(torch.int64)
        shift = self.images.to(cell.dtype) @ cell
        vector = ((positions[j] + shift - positions[i]) / self.autoang).to(dt)
        r2 = (vector * vector).sum(-1)
        in_cn = valid & (r2 <= self.cn_cutoff2)
        in_pair = valid & (r2 <= self.pair_cutoff2)
        r = torch.sqrt(torch.where(valid, r2, 1.0))
        si, sj = self.species[i].to(torch.int64), self.species[j].to(torch.int64)

        # pass 1
        rco = self.rcov[si] + self.rcov[sj]
        damp = 1.0 / (1.0 + torch.exp(-K1 * (rco / r - 1.0)))
        term = torch.where(in_cn, damp, 0.0).to(f64)
        cn = torch.zeros(n, dtype=f64, device=self.device).index_add_(0, i, term).index_add_(0, j, term)

        # pass 2
        c6, dc6_i, dc6_j = _c6_torch(self.references[si, sj], cn[i].to(dt), cn[j].to(dt))
        r6 = r2 * r2 * r2
        d6 = 1.0 / (r6 + self.r0_6[si, sj])
        d8 = 1.0 / (r6 * r2 + self.r0_8[si, sj])
        c8_factor = self.c8_factor[si, sj]
        de_dc6 = -(self.s6 * d6 + self.s8 * c8_factor * d8)
        energy = torch.where(in_pair, c6 * de_dc6, 0.0)
        g = torch.where(in_pair, c6 * (6.0 * self.s6 * r2 * r2 * d6 * d6
                                       + 8.0 * self.s8 * c8_factor * r6 * d8 * d8), 0.0)
        de_dcn = (torch.zeros(n, dtype=f64, device=self.device)
                  .index_add_(0, i, torch.where(in_pair, de_dc6 * dc6_i, 0.0).to(f64))
                  .index_add_(0, j, torch.where(in_pair, de_dc6 * dc6_j, 0.0).to(f64)))

        # pass 3
        if self.chain:
            ddamp = -damp * (1.0 - damp) * K1 * rco / (r * r)
            g = g + torch.where(in_cn, (de_dcn[i] + de_dcn[j]).to(dt) * ddamp / r, 0.0)

        pair_force = (g[:, None] * vector).to(f64)  # -dE/dx_i of the pair; x_j gets the opposite
        forces = (torch.zeros((n, 3), dtype=f64, device=self.device)
                  .index_add_(0, i, pair_force).index_add_(0, j, -pair_force))
        virial = -(vector.to(f64)[:, :, None] * pair_force[:, None, :]).reshape(-1, 9)
        return self._finish(energy.to(f64), forces, virial)

    def _triton(self, positions, cell):
        n, f64 = self.n_atoms, torch.float64
        positions = positions.contiguous()
        cell = cell.to(positions.dtype).contiguous()
        cn = torch.empty(n, dtype=f64, device=self.device)
        de_dcn = torch.empty(n, dtype=f64, device=self.device)
        energy = torch.empty(n, dtype=f64, device=self.device)
        forces = torch.empty((n, 3), dtype=f64, device=self.device)
        virial = torch.empty((n, 9), dtype=f64, device=self.device)
        kernels = _triton_kernels()
        common = (positions, cell, self.scalars, self.species, self.pair_i, self.pair_j, self.images, self.row,
                  self.col, self.by_j)
        constants = {"F64": self.dtype == torch.float64, "BLOCK": BLOCK}
        kernels["cn"][(n,)](*common, self.rcov, cn, num_warps=4, **constants)
        kernels["energy"][(n,)](*common, cn, self.references, self.c8_factor, self.r0_6, self.r0_8,
                                energy, de_dcn, forces, virial, len(self.rcov), num_warps=4, **constants)
        if self.chain:
            kernels["chain"][(n,)](*common, self.rcov, de_dcn, forces, virial, num_warps=4, **constants)
        return self._finish(energy, forces, virial)


def _c6_torch(references, cn_i, cn_j):
    """C6 and its derivatives by CN_i and CN_j: torch-dftd's softmax interpolation, per pair."""
    c6_ref, cn_a, cn_b = references.unbind(-1)
    da = cn_i[:, None] - cn_a
    db = cn_j[:, None] - cn_b
    log_weight = torch.where(c6_ref > 0.0, K3 * (da * da + db * db), -1.0e20)
    weight = torch.exp(log_weight - log_weight.max(dim=1, keepdim=True).values)
    total = weight.sum(1)
    c6 = (weight * c6_ref).sum(1) / total
    excess = weight * (c6_ref - c6[:, None])
    return c6, (excess * (2.0 * K3) * da).sum(1) / total, (excess * (2.0 * K3) * db).sum(1) / total


@cache
def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


@cache
def _triton_kernels():
    from fastmace.longrange import _d3_triton

    return {"cn": _d3_triton.cn_kernel, "energy": _d3_triton.energy_kernel, "chain": _d3_triton.chain_kernel}
