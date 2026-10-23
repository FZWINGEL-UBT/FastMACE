"""D3 references on exactly an ``fm.D3`` record's numbers (SPEC §11.2, §12.1; plan task 3b-T01).

- ``torch_dftd3``: ``torch-dftd`` 0.5.1, the §12.1 reference (MACE plus ``torch-dftd``), with
  Becke–Johnson damping passed explicitly (its default is zero damping) and the record's s6, s8,
  a1, a2 set on the module, so any record works, not only ``torch-dftd``'s named sets;
- ``simple_dftd3``: Grimme's ``simple-dftd3`` through PyPI's ``dftd3`` 1.6.0 (the "``dftd3``
  program" of §11.2), with the same numbers, positions and cell converted with the record's
  ``autoang`` and energies with its ``autoev``, so the two compare implementations, not constants.

Both return ``(energy eV, forces (N, 3) eV/Å, stress (3, 3) eV/Å³ in ASE's sign)`` as float64
numpy. ``tad-dftd3`` 0.7.0 has no periodic boundaries, so it is no reference for the periodic
cells FastMACE runs (docs/design/d3.md). Both packages are optional: ``pip install
'fastmace[d3-reference]'``.

Cutoffs: ``torch-dftd`` takes them in Å, selects the pairs in Å, then masks the coordination number
with ``cnthr / ase.units.Bohr`` against distances divided by its own ``d3_autoang``. Passing
``pair_cutoff_bohr * autoang`` Å puts both at exactly the record's bohr in its units (``mace_mp``'s
``40 * ase.units.Bohr`` gives pairs to 39.9999963 bohr).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fastmace.longrange import d3


def _check(record):
    if not isinstance(record, d3.D3):
        raise TypeError(f"record must be an fm.D3, got {type(record).__name__}")
    if record.three_body:
        raise ValueError("the references here are two-body only: three_body=True is not supported")


def torch_dftd3_tables_sha256() -> str:
    """SHA-256 of the installed ``torch-dftd``'s D3 tables (``d3.REFERENCE_TABLES`` names the expected one)."""
    import torch_dftd.nn.dftd3_module as module

    path = Path(module.__file__).parent / "params" / "dftd3_params.npz"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torch_dftd3_calculator(record: d3.D3, *, dtype: str = "float64", device: str = "cpu"):
    """A ``TorchDFTD3Calculator`` on ``record``'s numbers: BJ damping, the record's cutoffs in its own
    units, no ATM, no smoothing, ``dtype`` (``"float64"``: the reference, ``mace_mp``'s choice)."""
    import torch
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    _check(record)
    calculator = TorchDFTD3Calculator(damping="bj", xc="pbe", device=device, cutoff=d3.pair_cutoff_A(record),
                                      cnthr=d3.cn_cutoff_A(record), abc=False, dtype=getattr(torch, dtype),
                                      cutoff_smoothing="none")
    # torch-dftd's names: rs6 = a1, s18 = s8, rs18 = a2; alp is zero damping's and unused by bj.
    calculator.dftd_module.params = {"s6": record.s6, "rs6": record.a1, "s18": record.s8, "rs18": record.a2,
                                     "alp": 14.0}
    return calculator


def torch_dftd3(atoms, record: d3.D3, *, dtype: str = "float64", device: str = "cpu"):
    """``(energy, forces, stress)`` of ``atoms`` from ``torch-dftd`` on ``record``'s numbers; every atom's forces,
    whatever ``atoms``' constraints."""
    atoms = atoms.copy()
    atoms.set_constraint()
    atoms.calc = torch_dftd3_calculator(record, dtype=dtype, device=device)
    return (float(atoms.get_potential_energy()), np.array(atoms.get_forces(), dtype=float),
            np.array(atoms.get_stress(voigt=False), dtype=float))


def simple_dftd3(atoms, record: d3.D3):
    """``(energy, forces, stress)`` of periodic ``atoms`` from ``simple-dftd3`` on ``record``'s numbers."""
    from dftd3.interface import DispersionModel, RationalDampingParam

    _check(record)
    if not all(atoms.pbc):
        raise ValueError(f"simple_dftd3 here is for 3D periodic cells, got pbc={tuple(atoms.pbc)}")
    model = DispersionModel(np.asarray(atoms.numbers), np.asarray(atoms.positions) / record.autoang,
                            np.asarray(atoms.cell) / record.autoang, np.array([True, True, True]))
    # disp3 is ATM's cutoff; s9 = 0 switches ATM off. Widths 0: no smoothing.
    model.set_realspace_cutoff(record.pair_cutoff_bohr, record.cn_cutoff_bohr, record.cn_cutoff_bohr)
    result = model.get_dispersion(RationalDampingParam(s6=record.s6, s8=record.s8, s9=0.0, a1=record.a1,
                                                       a2=record.a2), grad=True)
    energy = float(result["energy"]) * record.autoev
    forces = -np.asarray(result["gradient"]) * record.autoev / record.autoang
    stress = np.asarray(result["virial"]) * record.autoev / atoms.get_volume()  # its virial is dE/d(strain)
    return energy, forces, stress
