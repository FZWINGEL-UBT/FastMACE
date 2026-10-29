"""D3(BJ) dispersion on the GPU in three passes with analytic derivatives (SPEC §11.2, plan task 3b-T03).

The input is a half list (docs/design/d3.md): undirected pairs ``(i, j, s)`` in canonical order,
``i < j``, plus each self-image pair ``(i, i, s)`` once. ``s`` is an int32 index into an integer
image table, multiplied by the current cell at every evaluation (NPT), and the pair vector is
``x_j + image_s @ cell - x_i`` on unwrapped positions. Unused slots of the padded list hold
``i = j = N``. ``prepare`` runs at list builds: it sorts by ``i`` and builds the permutation by
``j``, with row offsets for both. It is capturable, and the shapes stay static. Each pair then
belongs to the i-segment of one atom and the j-segment of another.

One Triton program per atom walks both of its segments and sums in float64 registers. There are no
atomics. The final sums over atoms are fixed-order reductions of per-atom float64 buffers. So the
result is deterministic, and a captured replay equals the eager call bit for bit. Every pair term
is computed twice, once for each atom, from the same expressions in the same order (pair vector
``x_j + shift - x_i``, C6 from the ``(Z_i, Z_j)`` table), so both atoms get exactly opposite pair
forces. The energy and the virial are summed only on the i side, so each pair counts once. A pair
over the cutoff stays in the list and is masked in registers (``r² ≤ r_c²``, ``where``), with no
compaction.

- Pass 1: coordination numbers ``CN_a = Σ_b 1 / (1 + exp(-k1 (R_cov,ab / r_ab - 1)))`` over
  ``r ≤ cn_cutoff``. A self-image pair adds to its atom twice, once from each segment, as the full
  sum over ``+s`` and ``-s`` does.
- Pass 2: the energy ``E = -Σ (s6 C6 / (r⁶ + R0⁶) + s8 C8 / (r⁸ + R0⁸))`` over
  ``r ≤ pair_cutoff``, with ``R0 = a1 sqrt(C8/C6) + a2`` and ``C8 = 3 C6 r2r4_a r2r4_b``. The
  direct forces and virial at fixed CN. And ``dE/dCN_a`` over every pair of atom a.
- Pass 3, after the kernel boundary that completes ``dE/dCN``: the chain term
  ``(dE/dCN_a + dE/dCN_b) ∂CN/∂r_ab`` into the forces and virial of both atoms. ``torch-dftd``
  gets the same term from autograd. Leaving it out keeps the energy but changes the forces.

``evaluate_energy`` (MC energy evaluations) runs pass 1 and pass 2's energy alone, on the i
segments, with no force, virial or dE/dCN writes. Its energy operations are those of the full
pass 2, so the energy is the same bit for bit.

C6 is interpolated as ``torch-dftd`` 0.5.1 does it: a softmax over the reference pairs with
``c6 > 0``, weighted by ``k3 ((CN_a - CN_A)² + (CN_b - CN_B)²)``.

Precision (plan phase 3b, conductor's choice 3): the pair terms are computed in ``dtype``, the
force precision. CN, ``dE/dCN``, energy, forces and virial are summed in float64. Forces come back
in ``dtype``, energy and virial in float64. Units are ``torch-dftd``'s: lengths are divided by the
record's ``autoang``, and energies multiplied by its ``autoev``.

The reference tables (C6 references, covalent radii, ``r2r4``) are ``torch-dftd`` 0.5.1's
``dftd3_params.npz``, shipped byte for byte as ``data/dftd3_params.npz`` so that no ``[d3]`` extra
is needed (SPEC §4.1, §14), and checked against the SHA-256 in ``d3.REFERENCE_TABLES``. They are
indexed by Z and read through ``numbers``, so a species event that writes ``numbers`` in place needs
no refresh, and no address moves (conductor's choice 5).

On CPU devices, or without Triton, the same three passes run in plain PyTorch with ``index_add``
(the torch tier, also the GPU kernels' oracle).
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import cache
from pathlib import Path

import numpy as np
import torch

from fastmace.longrange import d3
from fastmace.neighbors.skin import times_3x3

K1 = 16.0  # torch-dftd d3_k1: steepness of the coordination-number counting function
K3 = -4.0  # torch-dftd d3_k3: Gaussian weight of the C6 reference interpolation
REFERENCES = 32  # the 5 x 5 C6 reference pairs, padded to a power of two (padding has c6 = 0)
BLOCK = 128  # pairs per loop iteration of a program (fixed for every crystal and GPU)


TABLES = Path(__file__).resolve().parent / "data" / "dftd3_params.npz"


@cache
def reference_tables() -> dict[str, np.ndarray]:
    """The D3 tables by Z: ``c6ab`` (95, 95, 5, 5, 3), ``rcov`` (95,), ``r2r4`` (95,).

    ``TABLES``, shipped with FastMACE (``data/NOTICE.md``), refused unless its SHA-256 is the one
    ``d3.REFERENCE_TABLES`` names.
    """
    expected = d3.REFERENCE_TABLES.rsplit("sha256=", 1)[1]
    found = hashlib.sha256(TABLES.read_bytes()).hexdigest()
    if found != expected:
        raise RuntimeError(f"D3: {TABLES} has sha256 {found}, not {expected} ({d3.REFERENCE_TABLES}); "
                           "reinstall FastMACE")
    with np.load(TABLES) as data:
        return {name: np.array(data[name], dtype=np.float64) for name in ("c6ab", "rcov", "r2r4")}


class D3Kernel:
    """D3(BJ) energy, forces and virial of one structure, on a half pair list.

    - ``record``: a complete ``fm.D3``;
    - ``numbers``: atomic numbers (N,), an integer tensor on ``device`` kept by reference, so species
      events may change it in place (or anything ``numpy`` converts);
    - ``dtype``: the force precision of the pair terms (float32 or float64);
    - ``chain``: include pass 3. Only for the test that shows the chain term matters.

    ``prepare(pair_i, pair_j, image_index, image_table)`` after each list build, then
    ``evaluate(positions, cell, *, virial=True)`` gives a ``d3.DispersionResult`` and
    ``evaluate_energy(positions, cell)`` the energy alone. 3b-T04's ``DispersionTerm`` combines
    this kernel with 3b-T02's list.
    """

    def __init__(self, record: d3.D3, numbers, *, device, dtype=torch.float64, chain=True):
        if not isinstance(record, d3.D3):
            raise TypeError(f"record must be an fm.D3, got {type(record).__name__}")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"D3 pair terms run in float32 or float64, got {dtype}")
        if record.damping != "bj" or record.three_body or record.smoothing != "none":
            raise ValueError("the D3 kernel implements Becke-Johnson damping without ATM or smoothing")
        self.record = record
        self.device = torch.device(device)
        self.dtype = dtype
        self.chain = bool(chain)
        self.autoang = float(record.autoang)
        self.autoev = float(record.autoev)
        self.pair_cutoff2 = float(record.pair_cutoff_bohr) ** 2
        self.cn_cutoff2 = float(record.cn_cutoff_bohr) ** 2
        self.s6, self.s8 = float(record.s6), float(record.s8)

        tables = reference_tables()
        count = tables["rcov"].shape[0]
        if not isinstance(numbers, torch.Tensor):
            numbers = torch.as_tensor(np.asarray(numbers, dtype=np.int64), device=self.device)
        if numbers.device.type != self.device.type or numbers.dtype not in (torch.int32, torch.int64) \
                or numbers.ndim != 1:
            raise ValueError(f"numbers must be a 1-d integer tensor on {self.device}")
        self.device = numbers.device  # "cuda" resolved to the tensor's own index
        if numbers.numel() and (int(numbers.min()) < 1 or int(numbers.max()) >= count):
            raise ValueError(f"D3 has reference data for Z = 1 to {count - 1}")
        references = np.zeros((count, count, REFERENCES, 3))
        references[:, :, :25, :] = tables["c6ab"].reshape(count, count, 25, 3)
        r2r4 = tables["r2r4"]
        c8_factor = 3.0 * r2r4[:, None] * r2r4[None, :]
        r0 = record.a1 * np.sqrt(c8_factor) + record.a2

        def put(array, dtype=dtype):
            return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=self.device)

        self.n_atoms = numbers.shape[0]
        self.numbers = numbers  # by reference, at its fixed address
        self.count = count
        self.rcov = put(tables["rcov"])
        self.references = put(references)
        self.c8_factor = put(c8_factor)
        self.r0_6 = put(r0 ** 6)
        self.r0_8 = put(r0 ** 8)
        # float64 scalars for the kernels (Triton would pass Python floats as float32)
        self.scalars = put([self.autoang, self.pair_cutoff2, self.cn_cutoff2, self.s6, self.s8], torch.float64)
        self._prepared = False

    # ----------------------------------------------------------------------------------- list
    def prepare(self, pair_i, pair_j, image_index, image_table):
        """Take a half list: ``pair_i``, ``pair_j``, ``image_index`` (P,) and the ``image_table`` (M, 3).

        The image table holds integer image vectors, in an integer or float dtype (3b-T02's
        ``D3PairList.shifts`` is float64). Unused slots have ``i = j = N`` (any image index).
        Capturable: sorts and searches, no host synchronisation, shapes fixed by P and M. Returns True when
        a buffer address changed (a new P, or another image table tensor): captured graphs are then stale.
        With the shapes of the last call the buffers are refilled in place.
        """
        n = self.n_atoms
        pair_i = torch.as_tensor(pair_i, device=self.device).to(torch.int64).reshape(-1)
        pair_j = torch.as_tensor(pair_j, device=self.device).to(torch.int64).reshape(-1)
        image_index = torch.as_tensor(image_index, device=self.device).to(torch.int32).reshape(-1)
        image_table = torch.as_tensor(image_table, device=self.device).reshape(-1, 3)
        if pair_i.shape != pair_j.shape or image_index.shape != pair_i.shape:
            raise ValueError("pair_i, pair_j and image_index must have one entry per pair")
        if image_table.shape[0] == 0:
            raise ValueError("image_table must hold at least one image vector")
        sentinel = torch.full((1,), n, dtype=torch.int64, device=self.device)
        pair_i = torch.cat([pair_i, sentinel])  # never empty, so the kernels always get real buffers
        pair_j = torch.cat([pair_j, sentinel])
        image_index = torch.cat([image_index, image_index.new_zeros(1)])
        order = torch.argsort(pair_i, stable=True)
        sorted_i = pair_i[order].to(torch.int32).contiguous()
        sorted_j = pair_j[order].to(torch.int32).contiguous()
        by_j = torch.argsort(sorted_j.to(torch.int64), stable=True)
        atoms = torch.arange(n + 1, device=self.device, dtype=torch.int32)
        fresh = {"pair_i": sorted_i, "pair_j": sorted_j, "image_index": image_index[order].contiguous(),
                 "row": torch.searchsorted(sorted_i, atoms).to(torch.int32).contiguous(),
                 "by_j": by_j.to(torch.int32).contiguous(),
                 "col": torch.searchsorted(sorted_j[by_j], atoms).to(torch.int32).contiguous()}
        # Same shapes as the last prepare: copy in place, so captured graphs that read these addresses stay
        # valid after a list rebuild (3b-T04); new shapes (a grown list) move them.
        moved = False
        for name, tensor in fresh.items():
            old = getattr(self, name, None)
            if old is not None and old.shape == tensor.shape and old.dtype == tensor.dtype:
                old.copy_(tensor)
            else:
                setattr(self, name, tensor)
                moved = True
        table = image_table.contiguous()
        old = getattr(self, "image_table", None)
        if old is None or (old.data_ptr(), old.shape, old.dtype) != (table.data_ptr(), table.shape, table.dtype):
            moved = True
        self.image_table = table
        self._prepared = True
        return moved

    def buffers(self):
        """The prepared list's device buffers by name (the runtime-state registry and trial saves, 3b-T04)."""
        if not self._prepared:
            return {}
        return {name: getattr(self, name) for name in ("pair_i", "pair_j", "image_index", "row", "by_j", "col")}

    # ----------------------------------------------------------------------------- evaluation
    def evaluate(self, positions, cell, *, virial: bool = True) -> d3.DispersionResult:
        """Energy (0-d float64 eV), forces ((N, 3) ``dtype`` eV/Å) and, if asked, the virial (-dE/dε, eV)."""
        energy, forces, virial_ = self._run(positions, cell, energy_only=False, virial=virial)
        return d3.DispersionResult(energy=energy, forces=forces, virial=virial_)

    def evaluate_energy(self, positions, cell):
        """The energy alone (0-d float64 eV): pass 1 and pass 2's energy, no force or virial writes."""
        return self._run(positions, cell, energy_only=True, virial=False)[0]

    def _run(self, positions, cell, *, energy_only, virial):
        if not self._prepared:
            raise RuntimeError("D3Kernel.prepare(...) must run before an evaluation")
        if positions.is_cuda and _triton_available():
            return self._triton(positions, cell, energy_only, virial)
        return self._torch(positions, cell, energy_only, virial)

    def _finish(self, energy_atoms, forces, virial_atoms):
        energy = energy_atoms.sum() * self.autoev
        if forces is not None:
            forces = (forces * (self.autoev / self.autoang)).to(self.dtype)
        if virial_atoms is not None:
            virial_atoms = virial_atoms.sum(0).reshape(3, 3) * self.autoev
        return energy, forces, virial_atoms

    def _torch(self, positions, cell, energy_only, virial):
        n, dt, f64 = self.n_atoms, self.dtype, torch.float64
        valid = self.pair_i < n
        i = torch.where(valid, self.pair_i, 0).to(torch.int64)
        j = torch.where(valid, self.pair_j, 0).to(torch.int64)
        shifts = times_3x3(self.image_table.to(cell.dtype), cell)  # no FP64 GEMM (neighbors.skin.matmul_small)
        vector = ((positions[j] + shifts[self.image_index.to(torch.int64)] - positions[i]) / self.autoang).to(dt)
        r2 = (vector * vector).sum(-1)
        in_cn = valid & (r2 <= self.cn_cutoff2)
        in_pair = valid & (r2 <= self.pair_cutoff2)
        r = torch.sqrt(torch.where(valid, r2, 1.0))
        zi, zj = self.numbers[i].to(torch.int64), self.numbers[j].to(torch.int64)

        # pass 1
        rco = self.rcov[zi] + self.rcov[zj]
        damp = 1.0 / (1.0 + torch.exp(-K1 * (rco / r - 1.0)))
        term = torch.where(in_cn, damp, 0.0).to(f64)
        cn = torch.zeros(n, dtype=f64, device=self.device).index_add_(0, i, term).index_add_(0, j, term)

        # pass 2
        c6, dc6_i, dc6_j = _c6_torch(self.references[zi, zj], cn[i].to(dt), cn[j].to(dt))
        r6 = r2 * r2 * r2
        d6 = 1.0 / (r6 + self.r0_6[zi, zj])
        d8 = 1.0 / (r6 * r2 + self.r0_8[zi, zj])
        c8_factor = self.c8_factor[zi, zj]
        de_dc6 = -(self.s6 * d6 + self.s8 * c8_factor * d8)
        energy = torch.zeros(n, dtype=f64, device=self.device).index_add_(
            0, i, torch.where(in_pair, c6 * de_dc6, 0.0).to(f64))
        if energy_only:
            return self._finish(energy, None, None)
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
        virial_atoms = None
        if virial:
            virial_atoms = torch.zeros((n, 9), dtype=f64, device=self.device).index_add_(
                0, i, -(vector.to(f64)[:, :, None] * pair_force[:, None, :]).reshape(-1, 9))
        return self._finish(energy, forces, virial_atoms)

    def _triton(self, positions, cell, energy_only, virial):
        n, f64 = self.n_atoms, torch.float64
        positions = positions.contiguous()
        shifts = times_3x3(self.image_table.to(positions.dtype), cell.to(positions.dtype)).contiguous()
        cn = torch.empty(n, dtype=f64, device=self.device)
        energy = torch.empty(n, dtype=f64, device=self.device)
        if energy_only:
            de_dcn = forces = virial_atoms = energy  # not written
        else:
            de_dcn = torch.empty(n, dtype=f64, device=self.device)
            forces = torch.empty((n, 3), dtype=f64, device=self.device)
            virial_atoms = torch.empty((n, 9), dtype=f64, device=self.device) if virial else energy
        kernels = _triton_kernels()
        common = (positions, shifts, self.scalars, self.numbers, self.pair_i, self.pair_j, self.image_index,
                  self.row, self.col, self.by_j)
        f64_pairs = self.dtype == torch.float64
        kernels.cn_kernel[(n,)](*common, self.rcov, cn, F64=f64_pairs, BLOCK=BLOCK, num_warps=4)
        kernels.energy_kernel[(n,)](*common, cn, self.references, self.c8_factor, self.r0_6, self.r0_8, energy,
                                    de_dcn, forces, virial_atoms, self.count, F64=f64_pairs, VIRIAL=bool(virial),
                                    ENERGY_ONLY=energy_only, BLOCK=BLOCK, num_warps=4)
        if energy_only:
            return self._finish(energy, None, None)
        if self.chain:
            kernels.chain_kernel[(n,)](*common, self.rcov, de_dcn, forces, virial_atoms, F64=f64_pairs,
                                       VIRIAL=bool(virial), BLOCK=BLOCK, num_warps=4)
        return self._finish(energy, forces, virial_atoms if virial else None)


def image_table_of(images):
    """``(image_index (P,) int32, image_table (M, 3) int32)`` of per-pair integer images (P, 3)."""
    images = torch.as_tensor(images)
    table, index = torch.unique(images.reshape(-1, 3), dim=0, return_inverse=True)
    return index.to(torch.int32), table.to(torch.int32)


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


def _triton_kernels():
    from fastmace.longrange import _d3_triton

    return _d3_triton
