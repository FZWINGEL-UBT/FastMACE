"""The step's small float64 products are element-wise sums, never FP64 GEMMs (fp64-D1 relaxation fix, 28 Oct 2026).

Inside a captured CUDA graph, cuBLAS's float64 emulation with a fixed mantissa bit count (owner decision 65 D1, 56
bits on the RTX 5060 Ti and RTX 4000 Ada) returned small products about 8 bits accurate: a variable-cell LBFGS
relaxation of NaCl 3x3x3 from a non-minimal start set the cell to 16.75000 instead of 16.76443 Angstrom after one
step and diverged (ResearchModule Experiments/Energy_Hull, H0-diag). The model's own products were exact. The
products of the optimisers, the cell filter, the barostats, the variable-cell neighbour tests, the D3 image shifts
and the leader-shift copy are therefore written as element-wise sums (``neighbors.skin.matmul_small`` and friends),
which no cuBLAS setting reaches (LBFGS's triangular solves excepted, see ``ALLOWED``). These tests hold the helpers
to torch's products and keep GEMMs out of the step code.
"""

import ast
import inspect
import textwrap

import pytest
import torch

from fastmace.neighbors import skin

RNG = torch.Generator().manual_seed(0)


def rand(*shape, scale=1.0):
    return torch.randn(*shape, generator=RNG, dtype=torch.float64) * scale


@pytest.mark.parametrize(("left", "right"), [((3, 3), (3, 3)), ((6, 6), (6, 6)), ((7, 3), (3, 3)), ((2, 3, 3), (3, 3))])
def test_matmul_small_equals_matmul(left, right):
    a, b = rand(*left, scale=16.0), rand(*right)
    torch.testing.assert_close(skin.matmul_small(a, b), a @ b, rtol=1e-15, atol=1e-13)


def test_matvec_and_vecmat_small_equal_matmul():
    m, v, w = rand(100, 1620, scale=0.01), rand(1620), rand(100)
    torch.testing.assert_close(skin.matvec_small(m, v), m @ v, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(skin.vecmat_small(w, m), w @ m, rtol=1e-14, atol=1e-15)


def _step_functions():
    from fastmace.integrators import baoab, berendsen, mtk
    from fastmace.longrange import d3_kernel
    from fastmace.model import forces
    from fastmace.neighbors import d3_list
    from fastmace.optimize import cell, lbfgs

    return {
        "lbfgs.DeviceLBFGS.update": lbfgs.DeviceLBFGS.update,
        "lbfgs.DeviceLBFGS.two_loop": lbfgs.DeviceLBFGS.two_loop,
        "cell._sqrtm_3x3": cell._sqrtm_3x3, "cell.logm_3x3": cell.logm_3x3,
        "cell.FrechetCellFilter.set_x": cell.FrechetCellFilter.set_x,
        "cell.FrechetCellFilter.refresh": cell.FrechetCellFilter.refresh,
        "cell.FrechetCellFilter._atom_rows": cell.FrechetCellFilter._atom_rows,
        "baoab.expm_3x3": baoab.expm_3x3,
        **{f"baoab.{name}": getattr(baoab.LangevinBAOAB, name) for name in vars(baoab.LangevinBAOAB)
           if inspect.isfunction(getattr(baoab.LangevinBAOAB, name))},
        **{f"mtk.{name}": getattr(cls, name) for cls in (mtk.MTK,) for name in vars(cls)
           if inspect.isfunction(getattr(cls, name))},
        **{f"berendsen.{name}": getattr(cls, name) for cls in (berendsen.Berendsen,) for name in vars(cls)
           if inspect.isfunction(getattr(cls, name))},
        "skin.SkinNeighbourList.needs_rebuild": skin.SkinNeighbourList.needs_rebuild,
        "d3_list.needs_rebuild": d3_list.D3PairList.needs_rebuild,
        "d3_kernel._torch": d3_kernel.D3Kernel._torch, "d3_kernel._triton": d3_kernel.D3Kernel._triton,
        "forces._read_leader": forces.MACEForceField._read_leader,
    }


def _products(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.append(f"line {node.lineno}: @")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                "matmul", "mm", "bmm", "einsum", "solve_triangular", "addmm", "mv"):
            found.append(f"line {node.lineno}: {node.func.attr}")
    return found


# LBFGS's two memory x memory triangular solves stay TRSM: a captured fixed-mantissa replay leaves TRSM exact (the
# fp64-D1 graph probe), and a TRSM-free doubling solve lost ASE's step-by-step agreement on the cell rows.
ALLOWED = {"lbfgs.DeviceLBFGS.two_loop": ["solve_triangular", "solve_triangular"]}


def test_step_code_has_no_matrix_products():
    """Every function that runs inside a captured step: no ``@``, ``matmul``, ``mm``, ``einsum`` or
    ``solve_triangular`` beyond ``ALLOWED`` (host-side NumPy products live elsewhere, in ``build`` and the
    constructors)."""
    offenders = {}
    for name, function in _step_functions().items():
        found = [entry.split(": ")[1] for entry in _products(function)]
        if found != ALLOWED.get(name, []):
            offenders[name] = _products(function)
    assert not offenders, offenders
