"""Phase 1b contracts (plan task 1b-T01; SPEC §5, §7.1, §7.2, §7.6, §9.2, §10.6, §11.2).

The public records and the stubs that later 1b tasks fill: config fields, ``Workload``,
``Output``, the public integrator records, ``D3``, ``RunResult``, the registered-buffer record,
and the signatures of ``Simulation``, ``ForceField``, ``md`` and ``calc``. Everything here is
torch-free; ``docs/design/contracts-1b.md`` is the page that explains it.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest

from fastmace import config


def _frozen(instance, name, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, value)


# --------------------------------------------------------------------------- config fields


def test_new_config_defaults():
    assert config.PrecisionConfig().float64_energy is True
    integrator = config.IntegratorConfig()
    assert integrator.temperature_K is None
    assert integrator.external_stress_GPa is None
    assert integrator.disable_cell_langevin is False


def test_new_config_fields_are_frozen():
    _frozen(config.PrecisionConfig(), "float64_energy", False)
    _frozen(config.IntegratorConfig(), "temperature_K", 300.0)
    _frozen(config.IntegratorConfig(), "external_stress_GPa", 1.0)
    _frozen(config.IntegratorConfig(), "disable_cell_langevin", True)


def test_external_stress_scalar_and_voigt():
    assert config.IntegratorConfig(external_stress_GPa=2).external_stress_GPa == 2.0
    voigt = config.IntegratorConfig(external_stress_GPa=[1, 2, 3, 0, 0, 0.5]).external_stress_GPa
    assert voigt == (1.0, 2.0, 3.0, 0.0, 0.0, 0.5)
    assert isinstance(voigt, tuple)
    hash(config.IntegratorConfig(external_stress_GPa=[1, 2, 3, 0, 0, 0]))  # still hashable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"external_stress_GPa": (1.0, 2.0, 3.0)},  # neither scalar nor six components
        {"external_stress_GPa": math.nan},
        {"external_stress_GPa": (1, 2, 3, 4, 5, math.inf)},
        {"external_stress_GPa": True},
        {"external_stress_GPa": 1.0, "pressure_GPa": 1.0},  # two owners of one fact
        {"temperature_K": -1.0},
        {"temperature_K": math.inf},
        {"temperature_K": True},
        {"disable_cell_langevin": 1},
    ],
)
def test_invalid_new_integrator_fields(kwargs):
    with pytest.raises(ValueError):
        config.IntegratorConfig(**kwargs)


def test_float64_energy_must_be_bool():
    with pytest.raises(ValueError):
        config.PrecisionConfig(float64_energy=1)


def test_effective_external_stress():
    assert config.IntegratorConfig(pressure_GPa=2.0).effective_external_stress_GPa() == -2.0
    assert config.IntegratorConfig(external_stress_GPa=0.5).effective_external_stress_GPa() == 0.5
    voigt = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    assert config.IntegratorConfig(external_stress_GPa=voigt).effective_external_stress_GPa() == voigt


# --------------------------------------------------------------------------- Workload (§9.2)


def test_workload_defaults_and_frozen():
    from fastmace.planner.workload import Workload

    workload = Workload(max_temperature_K=1000)
    assert workload.max_temperature_K == 1000
    assert workload.species_changes is False
    assert workload.atom_count_changes is False
    assert workload.min_mc_temperature_K is None
    assert workload.host_callbacks_every is None
    assert workload.needs_per_atom == frozenset()
    assert workload.mc_moves == ()
    assert workload.reproducible is False
    assert workload.mc_energy_precision == "auto"
    _frozen(workload, "max_temperature_K", 2000)
    assert hash(workload) == hash(Workload(max_temperature_K=1000.0))


def test_workload_requires_max_temperature():
    from fastmace.planner.workload import Workload

    with pytest.raises(TypeError):
        Workload()


def test_workload_spec_example():
    """The §7.2 example constructs."""
    from fastmace.planner.workload import Workload

    workload = Workload(species_changes=True, mc_moves=("swap",), max_temperature_K=1000,
                        min_mc_temperature_K=250, host_callbacks_every=1000)
    assert workload.mc_moves == ("swap",)


def test_workload_normalises_collections():
    from fastmace.planner.workload import Workload

    workload = Workload(max_temperature_K=300, needs_per_atom=["energies"], mc_moves=["displace"],
                        min_mc_temperature_K=300)
    assert workload.needs_per_atom == frozenset({"energies"})
    assert workload.mc_moves == ("displace",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mc_moves": ("displace",)}, "min_mc_temperature_K"),  # MC without a temperature floor
        ({"mc_moves": ("displace",), "min_mc_temperature_K": 0.0}, "min_mc_temperature_K"),
        ({"mc_moves": ("displace",), "min_mc_temperature_K": -5.0}, "min_mc_temperature_K"),
        ({"min_mc_temperature_K": 0.0}, "min_mc_temperature_K"),  # <= 0 is refused even without moves
        ({"mc_moves": ("swap",), "min_mc_temperature_K": 300}, "species_changes"),
        ({"mc_moves": ("transmute",), "min_mc_temperature_K": 300}, "species_changes"),
        ({"mc_moves": ("insert",), "min_mc_temperature_K": 300}, "mc_moves"),
        ({"mc_moves": ("displace", "displace"), "min_mc_temperature_K": 300}, "mc_moves"),
        ({"mc_moves": "swap", "species_changes": True, "min_mc_temperature_K": 300}, "mc_moves"),
        ({"max_temperature_K": -1}, "max_temperature_K"),
        ({"max_temperature_K": math.nan}, "max_temperature_K"),
        ({"host_callbacks_every": 0}, "host_callbacks_every"),
        ({"host_callbacks_every": 2.5}, "host_callbacks_every"),
        ({"mc_energy_precision": "float16"}, "mc_energy_precision"),
        ({"reproducible": "yes"}, "reproducible"),
        ({"needs_per_atom": "energies"}, "needs_per_atom"),
    ],
)
def test_workload_validation(kwargs, message):
    from fastmace.planner.workload import Workload

    kwargs = {"max_temperature_K": 1000, **kwargs}
    with pytest.raises(ValueError, match=message):
        Workload(**kwargs)


def test_workload_accepts_atom_count_changes():
    """Reserved: the dataclass accepts it, the planner refuses it (rule 7, 1b-T12)."""
    from fastmace.planner.workload import Workload

    assert Workload(max_temperature_K=300, atom_count_changes=True).atom_count_changes is True


def test_workload_has_no_copies_of_other_owners_facts():
    """§9.2: ensemble, cell mode, stress, precision and the superset live elsewhere."""
    from fastmace.planner.workload import Workload

    names = {field.name for field in dataclasses.fields(Workload)}
    assert names == {"max_temperature_K", "species_changes", "atom_count_changes", "min_mc_temperature_K",
                     "host_callbacks_every", "needs_per_atom", "mc_moves", "reproducible",
                     "mc_energy_precision"}


# --------------------------------------------------------------------------- Output (§7.2, §10.6)


def test_output_spec_example_and_defaults():
    from fastmace.io.output import Output

    output = Output("run/", trajectory_every=20, thermo_every=20, restart_every=10_000)
    assert output.directory == "run/"
    assert Output("run/") == output
    assert Output("run/", trajectory_every=None).trajectory_every is None
    _frozen(output, "thermo_every", 10)


@pytest.mark.parametrize("kwargs", [{"thermo_every": 0}, {"trajectory_every": -1}, {"restart_every": 2.5},
                                    {"thermo_every": True}, {"directory": ""}])
def test_output_validation(kwargs):
    from fastmace.io.output import Output

    kwargs = {"directory": "run", **kwargs}
    with pytest.raises(ValueError):
        Output(**kwargs)


# --------------------------------------------------------------------------- integrators (§7.2, §10)


def test_langevin_record_maps_onto_config():
    from ase import units

    from fastmace import integrators

    record = integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.02)
    cfg = record.to_config(seed=7)
    assert cfg.ensemble == "nvt"
    assert cfg.timestep_fs == 0.5
    assert cfg.temperature_K == 300.0
    assert cfg.friction == pytest.approx(0.02 / units.fs, rel=1e-15)
    assert cfg.seed == 7
    assert cfg.fixcm is True
    _frozen(record, "temperature_K", 10)


def test_integrator_records_are_keyword_only():
    from fastmace import integrators

    for cls in (integrators.Langevin, integrators.LangevinBAOAB):
        with pytest.raises(TypeError):
            cls(0.5, 300, 0.02)


def test_baoab_record_maps_onto_config():
    from fastmace import integrators

    record = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=600, friction_per_fs=0.01,
                                       external_stress_GPa=(1, 1, 1, 0, 0, 0), cell_mode="full",
                                       barostat_tau_fs=500.0, barostat_mass_factor=2.0,
                                       disable_cell_langevin=True)
    cfg = record.to_config(seed=1)
    assert cfg.ensemble == "npt"
    assert cfg.cell_mode == "full"
    assert cfg.external_stress_GPa == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert cfg.pressure_GPa == 0.0
    assert cfg.barostat_tau_fs == 500.0
    assert cfg.barostat_mass_factor == 2.0
    assert cfg.disable_cell_langevin is True
    assert cfg.temperature_K == 600.0
    assert integrators.LangevinBAOAB(temperature_K=300, friction_per_fs=0.01,
                                     pressure_GPa=1.0).to_config(seed=0).pressure_GPa == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"friction_per_fs": 0.0},
        {"timestep_fs": -0.5},
        {"temperature_K": -1.0},
        {"cell_mode": "isotropic"},
        {"external_stress_GPa": 1.0, "pressure_GPa": 1.0},
    ],
)
def test_baoab_record_validates_at_construction(kwargs):
    from fastmace import integrators

    kwargs = {"temperature_K": 300, "friction_per_fs": 0.02, **kwargs}
    with pytest.raises(ValueError):
        integrators.LangevinBAOAB(**kwargs)


def test_langevin_record_requires_temperature_and_friction():
    from fastmace import integrators

    with pytest.raises(TypeError):
        integrators.Langevin(timestep_fs=0.5)


# --------------------------------------------------------------------------- D3 (§11.2)


def test_d3_record_defaults_are_the_spec_table():
    from fastmace.longrange.d3 import D3

    record = D3(functional="r2scan")
    assert record.damping == "bj"
    assert record.pair_cutoff_bohr == 40.0
    assert record.cn_cutoff_bohr == 40.0
    assert record.three_body is False
    assert record.smoothing == "none"
    assert record.precision == "float64"
    assert record.autoang == 0.52917726
    assert record.autoev == 27.21138505
    _frozen(record, "functional", "pbe")


@pytest.mark.parametrize("kwargs", [{"functional": ""}, {"functional": "pbe", "damping": "zero"},
                                    {"functional": "pbe", "pair_cutoff_bohr": 0.0},
                                    {"functional": "pbe", "precision": "float16"}])
def test_d3_record_validation(kwargs):
    from fastmace.longrange.d3 import D3

    with pytest.raises(ValueError):
        D3(**kwargs)


def test_d3_requires_functional():
    from fastmace.longrange.d3 import D3

    with pytest.raises(TypeError):
        D3()


# --------------------------------------------------------------------------- registry record (§8.1, §10.6)


def test_registered_buffer_policies():
    from fastmace.engine import registry

    assert registry.RESTART_POLICIES == ("save", "reset", "derived")
    buffer = registry.RegisteredBuffer(name="force_field.polar.overflow", owner="force_field", restart="reset",
                                       recompute="force_field.reset_polar", sticky=True)
    assert buffer.capture is True
    _frozen(buffer, "restart", "save")
    assert registry.RegisteredBuffer(name="state.positions", owner="state", restart="save").recompute is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"restart": "keep"},
        {"restart": "reset"},  # a reset buffer names what recomputes it
        {"restart": "save", "recompute": "something"},  # a saved buffer is not recomputed
        {"name": ""},
        {"owner": ""},
        {"sticky": "yes"},
        {"capture": 1},
    ],
)
def test_registered_buffer_validation(kwargs):
    from fastmace.engine import registry

    kwargs = {"name": "engine.flags", "owner": "engine", "restart": "save", **kwargs}
    with pytest.raises(ValueError):
        registry.RegisteredBuffer(**kwargs)


def test_registry_record_is_torch_free():
    import ast
    from pathlib import Path

    import fastmace.engine.registry as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in top_level for alias in node.names] + [
        node.module for node in top_level if isinstance(node, ast.ImportFrom) and node.module]
    assert not any(name.split(".")[0] == "torch" for name in names)


# --------------------------------------------------------------------------- Simulation, ForceField, md, calc


def _keyword_only(signature):
    return [name for name, p in signature.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY]


def test_simulation_signature():
    from fastmace.simulation import CHECK_MODES, Simulation

    signature = inspect.signature(Simulation)
    assert list(signature.parameters)[:2] == ["atoms", "force_field"]
    assert _keyword_only(signature) == ["species", "integrator", "workload", "output", "seed", "check",
                                        "allow_naive_cueq", "mode", "allow_reference", "fallback", "gate",
                                        "runner", "engine", "neighbours"]  # the planner inputs: 2-T01
    assert signature.parameters["check"].default == "skip"  # owner decision D1: opt-in gate
    assert signature.parameters["workload"].default is None  # owner decision D7
    assert signature.parameters["integrator"].default is inspect.Parameter.empty
    assert signature.parameters["seed"].default is inspect.Parameter.empty
    assert CHECK_MODES == ("skip", "gate", "allow_unavailable")
    for name in ("run", "save_restart", "resume", "set_positions", "set_momenta"):
        assert callable(getattr(Simulation, name))
    for name in ("plan", "gate"):
        assert isinstance(inspect.getattr_static(Simulation, name), property)
    run = inspect.signature(Simulation.run)
    assert list(run.parameters) == ["self", "steps", "schedule", "mc"]
    assert run.parameters["mc"].default is Ellipsis  # "keep the attached move set" (§7.6)
    resume = inspect.signature(Simulation.resume)
    assert _keyword_only(resume) == ["allow_plan_change", "check"]
    assert isinstance(inspect.getattr_static(Simulation, "resume"), classmethod)


def test_simulation_stub_refuses_an_unknown_check_mode():
    from fastmace.simulation import Simulation

    with pytest.raises(ValueError, match="check"):
        Simulation(None, None, integrator=None, seed=1, check="maybe")


def test_force_field_signature():
    from fastmace.model.forcefield import ForceField

    signature = inspect.signature(ForceField)
    assert list(signature.parameters)[:2] == ["model", "atoms"]
    assert signature.parameters["atoms"].default is None
    assert _keyword_only(signature) == ["precision", "rewrites", "species"]
    assert signature.parameters["precision"].default == "float32"
    assert signature.parameters["rewrites"].default == "auto"
    evaluate = inspect.signature(ForceField.evaluate)
    assert list(evaluate.parameters) == ["self", "atoms", "stress"]
    assert evaluate.parameters["stress"].default is False


def test_md_signature():
    from fastmace.quick import RunResult, md

    signature = inspect.signature(md)
    assert list(signature.parameters)[0] == "structure"
    expected = {"model", "head", "charge", "spin_multiplicity", "sha256", "trusted", "ensemble", "temperature_K",
                "schedule", "pressure_GPa", "cell_mode", "timestep_fs", "steps", "mc", "min_mc_temperature_K",
                "dispersion", "repeat", "output", "seed", "check", "integrator",
                "mode", "allow_reference", "fallback", "gate", "runner", "engine", "neighbours"}  # 2-T01
    assert set(_keyword_only(signature)) == expected
    assert signature.parameters["repeat"].default == (1, 1, 1)
    assert signature.parameters["check"].default == "skip"
    assert signature.parameters["ensemble"].default == "NVT"
    assert {field.name for field in dataclasses.fields(RunResult)} == {
        "summary", "plan", "gate", "simulation", "metadata", "output"}


def test_calc_signature():
    from fastmace.ase.calculator import FastMACECalculator, calc

    assert list(inspect.signature(calc).parameters)[0] == "model"
    assert "head" in _keyword_only(inspect.signature(calc))
    assert list(inspect.signature(FastMACECalculator).parameters)[0] == "model"
