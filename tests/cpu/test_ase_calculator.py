"""``FastMACECalculator`` and ``fm.calc``: the ASE front end, compiled tier (plan task 1b-T18).

SPEC §7.4, §7.6 "ASE front end", §7.7 level 0. On the CPU with tiny models:

- results equal stock MACE in float64 (energy, free energy, forces, Voigt stress). TODO(1b-T13):
  compare against the corrected reference calculator within the gate rule once T13 is merged;
- Voigt order and sign as ASE (against ASE's numerical stress);
- non-3D-PBC raises ``PlanRefused`` (rule 1), also after a periodic calculation;
- the compatibility-tier note is printed once and kept in ``calc.notes``;
- a new species set or atom count gets a fresh plan, the same structure keeps its plan;
- ``energies`` only for models whose terms are all atom-resolved (not PolarMACE).
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress

from _support.structures import nacl
from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.planner.minimal import MinimalPlanner

NA_CL = (11, 17)
NOTE = "compatibility tier: compiled, no CUDA graphs; use `fm.md` or `fastmace.ase` integrators for full speed"

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"),
              pytest.mark.filterwarnings("ignore:Please use `ase.calculators.fd:FutureWarning")]


@pytest.fixture
def quiet_licences(monkeypatch):
    from fastmace.model import loading

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())


@pytest.fixture
def checkpoint(tmp_path, quiet_licences):
    """A tiny checkpoint file of ``family`` for species ``z``; returns its path."""

    def make(family="mh1", z=NA_CL):
        model = tiny_mace_model(list(z), **FAMILIES[family])
        path = tmp_path / f"tiny-{family}-{'-'.join(map(str, z))}.model"
        torch.save(model, path)
        return str(path)

    return make


@pytest.fixture
def pristine(checkpoint):
    from fastmace.model import loading

    def make(family="mh1", z=NA_CL):
        return loading.load_pristine(checkpoint(family, z), trusted=True)

    return make


@pytest.fixture
def planners(monkeypatch):
    """CPU planners (no compilation) sharing one process record; the list of those made."""
    pytest.importorskip("cuequivariance_torch")
    from fastmace.planner import minimal

    record = constraints.ProcessRecord()
    made = []

    def make(force_field):
        planner = MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={}, process=record)
        made.append(planner)
        return planner

    monkeypatch.setattr(minimal, "default_planner", make)
    return made


def _stock(loaded, atoms):
    probe = atoms.copy()
    probe.calc = loaded.calculator  # MACE's own calculator on the pristine e3nn float64 model
    return probe.get_potential_energy(), probe.get_forces(), probe.get_stress()


@pytest.mark.parametrize("family", ["mh1", "residual", "density", "off"])
def test_results_equal_stock_mace_in_float64(pristine, planners, family):
    from fastmace.ase import FastMACECalculator

    loaded = pristine(family)
    atoms = nacl(rattle=0.05, seed=3)
    atoms.cell *= 1.01  # a cell other than the one of the first plan's structure is fine (stress path)
    atoms.calc = FastMACECalculator(loaded, precision="float64")
    energy, forces, stress = _stock(loaded, atoms)
    assert isinstance(atoms.get_potential_energy(), float)
    np.testing.assert_allclose(atoms.get_potential_energy(), energy, rtol=1e-12, atol=1e-10)
    assert atoms.get_potential_energy(force_consistent=True) == atoms.get_potential_energy()
    np.testing.assert_allclose(atoms.get_forces(), forces, rtol=0, atol=1e-10)
    assert atoms.get_stress().shape == (6,)
    np.testing.assert_allclose(atoms.get_stress(), stress, rtol=0, atol=1e-10)


def test_float32_results_are_close_to_stock_float64(pristine, planners):
    from fastmace.ase import FastMACECalculator

    loaded = pristine()
    atoms = nacl(rattle=0.05, seed=4)
    atoms.calc = FastMACECalculator(loaded)
    energy, forces, stress = _stock(loaded, atoms)
    np.testing.assert_allclose(atoms.get_potential_energy(), energy, rtol=1e-6, atol=1e-5)
    assert atoms.calc.plan.precision.dtype == "float32" and atoms.calc.plan.precision.float64_energy
    np.testing.assert_allclose(atoms.get_forces(), forces, rtol=0, atol=1e-4)
    np.testing.assert_allclose(atoms.get_stress(), stress, rtol=0, atol=1e-5)


def test_voigt_order_and_sign_as_ase(pristine, planners):
    from fastmace.ase import FastMACECalculator

    atoms = nacl(rattle=0.05, seed=5)
    atoms.set_cell(atoms.cell.array @ np.array([[1.0, 0.02, -0.01], [0.0, 0.99, 0.03], [0.0, 0.0, 1.02]]),
                   scale_atoms=True)
    calculator = FastMACECalculator(pristine(), precision="float64")
    atoms.calc = calculator
    analytic = atoms.get_stress()
    numeric = calculator.calculate_numerical_stress(atoms, d=1e-5, voigt=True)
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-7)
    np.testing.assert_allclose(analytic, full_3x3_to_voigt_6_stress(atoms.get_stress(voigt=False)), rtol=0, atol=1e-15)


@pytest.mark.parametrize("pbc", [False, (True, True, False)])
def test_non_3d_pbc_is_refused(pristine, planners, pbc):
    from fastmace.ase import FastMACECalculator

    calculator = FastMACECalculator(pristine(), precision="float64")
    molecule = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2.4]], cell=[10, 10, 10], pbc=pbc)
    molecule.calc = calculator
    with pytest.raises(PlanRefused, match="3D PBC"):
        molecule.get_potential_energy()

    periodic = nacl(rattle=0.05, seed=6)
    periodic.calc = calculator
    periodic.get_forces()
    periodic.pbc = pbc
    with pytest.raises(PlanRefused, match="3D PBC"):
        periodic.get_forces()


def test_note_is_printed_once_and_recorded(pristine, planners, capsys):
    from fastmace.ase import FastMACECalculator

    calculator = FastMACECalculator(pristine(), precision="float64")
    assert calculator.plan is None
    small, large = nacl(repeat=1, rattle=0.05, seed=7), nacl(repeat=2, rattle=0.05, seed=8)
    for atoms in (small, large, small):
        atoms.calc = calculator
        atoms.get_forces()
    printed = capsys.readouterr()
    assert (printed.out + printed.err).count(NOTE) == 1
    assert calculator.notes.count(NOTE) == 1


def test_replans_when_species_or_atom_count_change(pristine, planners, monkeypatch):
    from fastmace.ase import FastMACECalculator
    from fastmace.planner import minimal

    caches = []
    build = MinimalPlanner.build_evaluation

    def recording(self, plan, force_field, atoms, *, compile_cache=None, check=True):
        caches.append(compile_cache)
        return build(self, plan, force_field, atoms, compile_cache=compile_cache, check=check)

    monkeypatch.setattr(minimal.MinimalPlanner, "build_evaluation", recording)
    loaded = pristine(z=(8, 11, 17))
    calculator = FastMACECalculator(loaded, precision="float64")
    atoms = nacl(repeat=1, rattle=0.05, seed=9)
    atoms.calc = calculator
    atoms.get_forces()
    first = calculator.plan
    assert first.superset == NA_CL

    atoms.rattle(0.02, seed=10)  # positions only: same plan
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # cell too: the plan has stress for any cell
    atoms.get_forces()
    assert calculator.plan is first and len(caches) == 1

    bigger = nacl(repeat=2, rattle=0.05, seed=11)  # atom count
    bigger.calc = calculator
    bigger.get_forces()
    assert calculator.plan is not first and len(caches) == 2

    oxide = nacl(repeat=1, rattle=0.05, seed=12)  # species: a fresh plan whose superset is the new structure's
    oxide.numbers[0] = 8
    oxide.calc = calculator
    np.testing.assert_allclose(oxide.get_forces(), _stock(loaded, oxide)[1], rtol=0, atol=1e-10)
    assert calculator.plan.superset == (8, 11, 17) and len(caches) == 3
    assert caches[0] is not None and all(cache is caches[0] for cache in caches)


def test_energies_for_atom_resolved_models(pristine, planners):
    from fastmace.ase import FastMACECalculator

    atoms = nacl(rattle=0.05, seed=13)
    atoms.calc = FastMACECalculator(pristine(), precision="float64")
    assert "energies" in atoms.calc.implemented_properties
    energies = atoms.get_potential_energies()
    assert energies.shape == (len(atoms),) and energies.dtype == np.float64
    np.testing.assert_allclose(energies.sum(), atoms.get_potential_energy(), rtol=1e-12, atol=1e-10)


def test_no_energies_for_polar_models(pristine):
    if not polar_available():
        pytest.skip("graph_longrange is not installed")
    from fastmace.ase import FastMACECalculator

    calculator = FastMACECalculator(pristine("polar"))
    assert set(calculator.implemented_properties) == {"energy", "free_energy", "forces", "stress"}
    atoms = nacl()
    atoms.calc = calculator
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energies()


def test_no_energies_when_a_term_is_not_atom_resolved(pristine, monkeypatch):
    """The rule without graph_longrange: PolarMACE's terms (energy.TERMS) leave ``energies`` out."""
    from fastmace.ase import FastMACECalculator
    from fastmace.model import energy

    resolved, unresolved = energy.TERMS["PolarMACE"]
    monkeypatch.setattr(energy, "terms", lambda model: {"resolved": resolved, "unresolved": unresolved})
    assert "energies" not in FastMACECalculator(pristine()).implemented_properties


def test_implemented_properties():
    from fastmace.ase.calculator import FastMACECalculator

    assert {"energy", "free_energy", "forces", "stress"} <= set(FastMACECalculator.implemented_properties)
    assert "energies" not in FastMACECalculator.implemented_properties  # per instance, by the model's terms


def test_lbfgs_with_a_cell_filter_runs(pristine, planners):
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    from fastmace.ase import FastMACECalculator

    atoms = nacl(repeat=1, rattle=0.05, seed=14)
    atoms.calc = FastMACECalculator(pristine(), precision="float64")
    start = atoms.get_potential_energy()
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=5)
    assert atoms.get_potential_energy() <= start


def test_calc_loads_and_passes_options(checkpoint, planners, capsys):
    import fastmace as fm
    from fastmace.ase import FastMACECalculator
    from fastmace.errors import CheckpointTrustError

    path = checkpoint()
    with pytest.raises(CheckpointTrustError):
        fm.calc(path)
    calculator = fm.calc(path, trusted=True, precision="float64", rewrites="auto")
    assert isinstance(calculator, FastMACECalculator) and calculator.precision == "float64"
    atoms = nacl(rattle=0.05, seed=15)
    atoms.calc = calculator
    atoms.get_forces()
    assert NOTE in capsys.readouterr().out
    with pytest.raises(TypeError, match="no_such_option"):
        fm.calc(path, trusted=True, no_such_option=1)


def test_calculator_takes_a_loaded_model_or_a_label(pristine):
    from fastmace.ase import FastMACECalculator

    loaded = pristine()
    with pytest.raises(TypeError, match="fm.load"):
        FastMACECalculator("/no/such/file.model")
    with pytest.raises(ValueError, match="head"):
        FastMACECalculator(loaded, head="no_such_head")
    with pytest.raises(ValueError, match="precision"):
        FastMACECalculator(loaded, precision="float16")
