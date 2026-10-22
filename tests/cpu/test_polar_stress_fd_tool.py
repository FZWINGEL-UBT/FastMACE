"""tools/polar_stress_fd.py without a GPU: its central differences and its argument checks (plan task 3b-T08)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("_tool_polar_stress_fd", TOOLS / "polar_stress_fd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def _cell_energy(sigma0, volume0, step=None):
    """E = V0 sigma0 : eps + V0 |eps|^2 for the strain eps of the cell against the reference cell; with
    ``step`` = (threshold, jump) a step of ``jump`` eV where eps_xx crosses ``threshold``."""
    def energy(atoms, cell0):
        eps = np.linalg.solve(cell0, atoms.cell.array) - np.identity(3)
        eps = 0.5 * (eps + eps.T)
        value = volume0 * (np.sum(sigma0 * eps) + np.sum(eps * eps))
        if step is not None and eps[0, 0] > step[0]:
            value += step[1]
        return value
    return energy


def test_central_differences_recover_a_smooth_stress_and_show_a_step_as_1_over_h():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([3.0, 4.0, 5.0]) + 0.1, pbc=True)
    cell0, volume0 = atoms.cell.array.copy(), atoms.get_volume()
    sigma0 = np.array([[1.0, 0.2, 0.3], [0.2, -0.5, 0.1], [0.3, 0.1, 0.7]]) * 1e-2
    smooth = _cell_energy(sigma0, volume0)
    stress, detail = tool.fd_stress(lambda a: (smooth(a, cell0), None), atoms, 1e-4)
    # The quadratic term cancels in central differences; V(eps) = V0 to first order.
    np.testing.assert_allclose(stress, sigma0, atol=1e-10)
    assert [d["component"] for d in detail] == ["00", "11", "22", "12", "02", "01"]
    jump = 1e-3
    stepped = _cell_energy(sigma0, volume0, step=(0.5e-4, jump))
    for h in (1e-4, 3e-4):
        stress, _ = tool.fd_stress(lambda a: (stepped(a, cell0), None), atoms, h)
        assert stress[0, 0] - sigma0[0, 0] == pytest.approx(jump / (2 * h * volume0), rel=1e-6)
        np.testing.assert_allclose(np.delete(stress.ravel(), 0), np.delete(sigma0.ravel(), 0), atol=1e-10)
    stress, _ = tool.fd_stress(lambda a: (stepped(a, cell0), None), atoms, 1e-5)  # the step is outside +-h
    np.testing.assert_allclose(stress, sigma0, atol=1e-9)


def test_voigt_strain_halves_the_shear():
    assert tool.voigt_strain(0, 0, 1e-3)[0, 0] == 1e-3
    shear = tool.voigt_strain(0, 1, 1e-3)
    assert shear[0, 1] == shear[1, 0] == 5e-4 and np.count_nonzero(shear) == 2


def test_arguments():
    args = tool.parse_args(["--path", "reference", "--output", "o.json"])
    assert args.dtype == "float64" and args.label == "polar-1-s" and list(args.repeat) == [3, 3, 3]
    assert args.steps == list(tool.DEFAULT_STEPS) and args.configs == ["ideal", "gate"]
    with pytest.raises(SystemExit):
        tool.parse_args(["--path", "hybrid", "--dtype", "float64", "--output", "o.json"])
    with pytest.raises(SystemExit):
        tool.parse_args(["--path", "stock", "--frozen-kset", "--output", "o.json"])
