"""ASE's ``FrechetCellFilter`` on the device (plan task 4-T07; SPEC §10.3), against ASE 3.27 on the CPU.

The matrix functions first: ``expm_3x3`` (now any square size) bitwise unchanged on 3 x 3, the device ``logm``
against SciPy's, and the one-call Fréchet adjoint against ASE's nine ``expm_frechet`` directions. Then the
filter: the torch Lennard-Jones force field of ``test_engine`` with stress, relaxed by ``DeviceLBFGS`` and
``FIRE`` over ``FrechetCellFilter`` on ``GraphEngine``, against ASE's optimisers on ASE's ``FrechetCellFilter``
and ``LennardJones`` from the same strained, rattled argon crystal. Then refusals, the range guard and the
buffer registry.

The step-by-step reference is ``ExactCellRows``: ASE's filter with its cell rows evaluated without the
``ln(N)`` offset's rounding. ASE computes ``expm(X / ecf) / cf`` and ``ecf * logm(cf * F)`` on matrices of
norm ~ln N; against 40-digit mpmath that costs 2e-14 relative in F and 3e-13 in the cell rows, where the
shifted forms ``expm(X / ecf - ln(cf) I)`` and ``ecf * (logm(F) + ln(cf) I)`` (FastMACE's) are exact to
1e-17. LBFGS on the cell rows (|x| ~ 500, steps ~ 0.1) amplifies that: ASE against itself with one
position moved by 1.6e-14 Angstrom drifts 4e-12 by step 2 and 9e-11 by step 50 (4-T07, 23 October 2026).
Against ``ExactCellRows`` FIRE agrees within 1e-12 Angstrom after every step. LBFGS keeps one rounding
floor: the cell rows sit at |x| ~ 500 (the offset is ASE's definition of x), where one ulp is 6e-14 against
history differences ``s`` of ~1e-4, so any two implementations that round x differently by one ulp part.
``ExactCellRows`` against itself with +-1 ulp of noise in its cell rows drifts 5e-12 to 2.5e-11 Angstrom
over 50 steps on the five filters below (4-T07); FastMACE against it, 4e-12 to 4e-11. LBFGS is held at
``LBFGS_CELL_ATOL`` = 1e-10 after every step. Against stock ASE: the first step within 1e-12, and the
relaxed structure within 1e-8 at the same step count.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import units
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter as AseFrechet
from ase.optimize import FIRE as AseFIRE
from ase.optimize import LBFGS as AseLBFGS
from test_engine import EPSILON, RC, SIGMA, ReplayEngine, TorchLJ, argon

from fastmace import optimize
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.engine.registry import unregistered_mutations
from fastmace.integrators.baoab import EXPM_DEGREE, EXPM_SCALING, expm_3x3
from fastmace.integrators.registry import build_dof, build_optimizer, optimizer_refusals, registry
from fastmace.optimize import cell, driver
from fastmace.state import SimulationState

POSITION_ATOL = 1e-12
LBFGS_CELL_ATOL = 1e-10  # a few times ASE's own one-ulp drift on the cell rows (module docstring)


# --------------------------------------------------------------------------- matrix functions


def _expm_3x3_before(a):
    """``expm_3x3`` as it was before 4-T07 generalised its size (``torch.eye(3)``)."""
    a = a / 2 ** EXPM_SCALING
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    result, term = eye.clone(), eye
    for k in range(1, EXPM_DEGREE + 1):
        term = term @ a / k
        result = result + term
    for _ in range(EXPM_SCALING):
        result = result @ result
    return result


def test_expm_is_unchanged_on_3x3_to_rounding():
    """Bitwise until the fp64-D1 fix (28 October 2026): ``expm_3x3`` now multiplies with element-wise sums
    (``neighbors.skin.matmul_small``), which round each product where the GEMM of ``_expm_3x3_before`` fuses them,
    so the two agree to a few ulps (and bit for bit only where the BLAS happens to sum the same way)."""
    generator = np.random.default_rng(1)
    for scale in (1e-6, 1e-3, 0.1, 1.0):
        a = torch.tensor(generator.standard_normal((3, 3)) * scale)
        # A few ulps: 6.7e-16 relative on leto's CPU (29 October 2026), so 4e-16 was too tight.
        torch.testing.assert_close(expm_3x3(a), _expm_3x3_before(a), rtol=1e-15, atol=1e-16)


def test_expm_takes_larger_squares():
    from scipy.linalg import expm

    a = np.random.default_rng(2).standard_normal((6, 6)) * 0.2
    np.testing.assert_allclose(expm_3x3(torch.tensor(a)).numpy(), expm(a), rtol=1e-13, atol=1e-15)


def deformations(count=40, seed=4):
    """Deformation gradients near I: strains up to 30 % and shears, and the identity."""
    generator = np.random.default_rng(seed)
    out = [np.eye(3)]
    for k in range(count):
        scale = (1e-6, 1e-3, 0.05, 0.3)[k % 4]
        out.append(np.eye(3) + generator.uniform(-scale, scale, (3, 3)))
    return out


def test_logm_matches_scipy():
    from scipy.linalg import logm

    for f in deformations():
        ours = cell.logm_3x3(torch.tensor(f)).numpy()
        # 2e-14 since the fp64-D1 fix: element-wise sums round each product (1e-14 with the fused GEMM before)
        np.testing.assert_allclose(ours, logm(f).real, rtol=0, atol=2e-14)


def test_frechet_adjoint_matches_ases_nine_directions():
    from scipy.linalg import expm_frechet, logm

    generator = np.random.default_rng(5)
    for f in deformations(12):
        a = logm(f).real
        g = generator.normal(size=(3, 3)) * 30.0  # virial-sized, eV
        expected = np.zeros((3, 3))
        for mu in range(3):
            for nu in range(3):
                direction = np.zeros((3, 3))
                direction[mu, nu] = 1.0
                expected[mu, nu] = np.sum(expm_frechet(a, direction, compute_expm=False) * g)
        ours = cell.expm_frechet_adjoint(torch.tensor(a), torch.tensor(g)).numpy()
        np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-12 * np.abs(g).max())


# --------------------------------------------------------------------------- the filter against ASE


def crystal(fixed=None, strain=0.03):
    atoms = argon(seed=11, displacement=0.1)
    shear = np.eye(3) + np.array([[strain, 0.01, 0.0], [0.0, -strain / 2, 0.02], [0.0, 0.0, strain]])
    atoms.set_cell(atoms.cell.array @ shear, scale_atoms=True)
    if fixed is not None:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def build(record=None, *, atoms=None, replay=False, skin=0.5):
    """A test helper standing in for the planner: force field with stress, state, filter, optimiser, engine."""
    record = record or optimize.LBFGS(cell_filter=optimize.FrechetCell())
    atoms = crystal() if atoms is None else atoms
    neighbours = NeighbourConfig(skin=skin, sort_receivers="off", gpu_neighbour_list="off")
    engine = EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu")
    field = TorchLJ(atoms, neighbours, compute_stress=True)
    state = SimulationState.from_atoms(atoms, "cpu", 0, compute_stress=True)
    config = record.to_config()
    optimizer = build_optimizer(build_dof(state, config), config)
    cls = ReplayEngine if replay else GraphEngine
    out = cls(engine, field, driver.OptimizerStepper(optimizer), state, neighbours=neighbours, capture=replay,
              constraints=constraints.ProcessRecord())
    if replay:
        out.captures = []
    return out


class ExactCellRows(AseFrechet):
    """ASE's ``FrechetCellFilter`` with ``get_positions`` and ``set_positions`` free of the ``ln(N)`` offset's
    rounding (module docstring): the same arithmetic otherwise, ``get_forces`` unchanged."""

    def get_positions(self):
        from scipy.linalg import logm

        deform = np.linalg.solve(self.orig_cell, self.atoms.cell.array).T
        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        pos[:natoms] = np.linalg.solve(deform, self.atoms.positions.T).T
        pos[natoms:] = self.exp_cell_factor * (logm(deform).real + np.log(self.cell_factor) * np.eye(3))
        return pos

    def set_positions(self, new, **kwargs):
        from scipy.linalg import expm

        natoms = len(self.atoms)
        deform = (expm(new[natoms:] / self.exp_cell_factor - np.log(self.cell_factor) * np.eye(3))
                  - np.eye(3)).T * self.mask
        self.atoms.set_cell(self.orig_cell @ (np.eye(3) + deform), scale_atoms=True)
        self.atoms.set_positions(new[:natoms] @ (np.eye(3) + deform), **kwargs)


def ase_filter(atoms, filter_record, exact=False):
    mask = filter_record.mask
    if mask is not None:
        mask = np.array(mask, dtype=float)
    return (ExactCellRows if exact else AseFrechet)(
        atoms, mask=mask, exp_cell_factor=filter_record.exp_cell_factor,
        hydrostatic_strain=filter_record.hydrostatic_strain, constant_volume=filter_record.constant_volume,
        scalar_pressure=filter_record.scalar_pressure_GPa * units.GPa)


def ase_optimizer(atoms, record, exact=True):
    atoms.calc = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)
    target = ase_filter(atoms, record.cell_filter, exact)
    if isinstance(record, optimize.LBFGS):
        out = AseLBFGS(target, logfile=None, maxstep=record.maxstep_A, memory=record.memory,
                       damping=record.damping, alpha=record.alpha)
    else:
        out = AseFIRE(target, logfile=None, dt=record.dt, maxstep=record.maxstep_A, dtmax=record.dtmax,
                      Nmin=record.Nmin, finc=record.finc, fdec=record.fdec, astart=record.astart, fa=record.fa,
                      a=record.a, downhill_check=record.downhill_check)
    out.cell_filter_object = target
    return out


def lockstep(record, *, steps=50, fmax=1e-6, replay=False, exact=True):
    """ASE's ``irun`` and the engine side by side; positions and cell compared after every step."""
    reference = crystal()
    ase_opt = ase_optimizer(reference, record, exact)
    engine = build(record, replay=replay)
    engine.integrator.optimizer.set_fmax(fmax)
    run = ase_opt.irun(fmax=fmax, steps=steps)
    next(run)  # the initial convergence check
    worst = 0.0
    for taken in range(1, steps + 1):
        try:
            next(run)
        except StopIteration:
            break
        engine.step()
        assert not engine.halted
        assert engine.state.step == taken == ase_opt.nsteps
        worst = max(worst, float(np.abs(engine.state.positions.numpy() - reference.positions).max()),
                    float(np.abs(engine.state.cell.numpy() - reference.cell.array).max()))
    return reference, engine, worst


FILTERS = {
    "default": optimize.FrechetCell(),
    "pressure": optimize.FrechetCell(scalar_pressure_GPa=0.05),
    "hydrostatic": optimize.FrechetCell(hydrostatic_strain=True),
    "mask": optimize.FrechetCell(mask=(1, 1, 0, 0, 0, 1)),
    "factor": optimize.FrechetCell(exp_cell_factor=10.0),
}


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_lbfgs_matches_ase_at_every_step(name):
    reference, engine, worst = lockstep(optimize.LBFGS(cell_filter=FILTERS[name]))
    assert worst < LBFGS_CELL_ATOL, worst
    assert not np.allclose(engine.state.cell.numpy(), crystal().cell.array)  # the cell moved


@pytest.mark.parametrize("name", ["default", "constant_volume", "mask"])
def test_fire_matches_ase_at_every_step(name):
    cell_filter = optimize.FrechetCell(constant_volume=True) if name == "constant_volume" else FILTERS[name]
    reference, engine, worst = lockstep(optimize.FIRE(cell_filter=cell_filter))
    assert worst < POSITION_ATOL, worst


def test_fire_downhill_check_with_the_enthalpy():
    record = optimize.FIRE(dt=0.4, dtmax=2.0, downhill_check=True,
                           cell_filter=optimize.FrechetCell(scalar_pressure_GPa=0.05))
    reference, engine, worst = lockstep(record, steps=40)
    assert worst < POSITION_ATOL, worst


def test_the_first_step_matches_stock_ase():
    for record in (optimize.LBFGS(cell_filter=optimize.FrechetCell()), optimize.FIRE(cell_filter=FILTERS["mask"])):
        reference, engine, worst = lockstep(record, steps=1, exact=False)
        assert worst < POSITION_ATOL, worst


def test_replayed_stages_match_ase():
    reference, engine, worst = lockstep(optimize.LBFGS(cell_filter=optimize.FrechetCell()), steps=30, replay=True)
    assert worst < LBFGS_CELL_ATOL, worst
    reference, engine, worst = lockstep(optimize.FIRE(cell_filter=optimize.FrechetCell()), steps=30, replay=True)
    assert worst < POSITION_ATOL, worst


@pytest.mark.parametrize("exact", [True, False], ids=["exact-rows", "stock-ase"])
def test_same_step_count_to_fmax(exact):
    record = optimize.LBFGS(cell_filter=optimize.FrechetCell())
    reference = crystal()
    ase_opt = ase_optimizer(reference, record, exact)
    converged = bool(ase_opt.run(fmax=1e-3, steps=1000))
    engine = build(record)
    summary = driver.relax(engine, fmax=1e-3, steps=1000)
    assert converged and summary["converged"] and not summary["fixed_cell"]
    assert summary["steps"] == ase_opt.nsteps
    atol = 1e-10 if exact else 1e-8
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=atol)
    np.testing.assert_allclose(engine.state.cell.numpy(), reference.cell.array, atol=atol)
    assert summary["volume_A3"] == pytest.approx(reference.get_volume(), rel=atol)
    np.testing.assert_allclose(summary["cell_A"], reference.cell.array, atol=atol)
    gradient = ase_opt.cell_filter_object.get_forces()
    assert summary["fmax_eV_A"] == pytest.approx(float(np.linalg.norm(gradient, axis=1).max()), abs=atol)


def test_the_filter_view_matches_ases_at_the_start():
    """``x`` (with the ln N offset of ASE's cell rows), the gradient and the enthalpy before any step."""
    record = optimize.FrechetCell(scalar_pressure_GPa=0.05, mask=(1, 1, 1, 0, 1, 1))
    reference = crystal()
    reference.calc = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)
    target = ase_filter(reference, record)
    engine = build(optimize.LBFGS(cell_filter=record))
    dof = engine.integrator.dof
    assert dof.rows == len(reference) + 3 and dof.variable_cell and dof.mask is None
    np.testing.assert_allclose(dof.x.numpy(), target.get_positions().ravel(), rtol=1e-15, atol=1e-12)
    np.testing.assert_allclose(dof.gradient().numpy(), -target.get_forces().ravel(), rtol=0, atol=1e-12)
    assert float(dof.energy()) == pytest.approx(target.get_potential_energy(), abs=1e-12)


# --------------------------------------------------------------------------- refusals, guard, registry


def test_the_filter_is_registered():
    record = registry().cell_filter("frechet")
    assert record.load() is cell.FrechetCellFilter
    assert record.ase_class == "ase.filters.FrechetCellFilter"
    assert {name for name, _, _ in record.buffers} == set(cell.FrechetCellFilter.BUFFERS)


def test_fix_atoms_with_a_cell_filter_is_refused():
    config = optimize.LBFGS(cell_filter=optimize.FrechetCell()).to_config()
    reasons = optimizer_refusals(config, fix_atoms=True)
    assert [reason.code for reason in reasons] == ["constraints"]
    assert "owner decision D1" in str(reasons[0])
    assert optimizer_refusals(config) == []
    with pytest.raises(ValueError, match="FixAtoms"):
        build(atoms=crystal(fixed=[0, 3]))


def test_a_non_periodic_structure_is_refused():
    atoms = crystal()
    atoms.pbc = (True, True, False)
    with pytest.raises(ValueError, match="periodic"):
        build(atoms=atoms)


def test_the_range_guard_stops_the_relaxation():
    engine = build()
    dof = engine.integrator.dof
    x = dof.x.clone()
    x.view(-1, 3)[-3:] += torch.eye(3, dtype=torch.float64) * dof.exp_cell_factor * 0.8  # log strain 0.8
    before = engine.state.cell.clone()
    dof.set_x(x)
    assert torch.equal(engine.state.cell, before)  # an out-of-range move is not applied
    assert not bool(dof.cell_ok)
    with pytest.raises(FloatingPointError, match="range"):
        driver.relax(engine, fmax=1e-3, steps=5)


def test_buffers_are_registered_and_restored_around_warm_ups():
    engine = build()
    assert {f"dof.{name}" for name in cell.FrechetCellFilter.BUFFERS} <= set(engine.registry.entries())
    assert unregistered_mutations(engine, lambda: [engine.step() for _ in range(5)]) == []


def test_a_converged_step_leaves_the_committed_state_bitwise():
    engine = build()
    driver.relax(engine, fmax=1e-3, steps=1000)
    positions, cell_now = engine.state.positions.clone(), engine.state.cell.clone()
    engine.step()  # propose finds converged set and moves to the same x
    assert engine.halted
    assert torch.equal(engine.state.positions, positions) and torch.equal(engine.state.cell, cell_now)
