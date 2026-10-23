"""Phase 4 contracts (plan task 4-T01): the scheme registry and rule 1 as data, the phase 4 records, the
optimiser and DOF-view contract, FixAtoms in the state, restart format 4 and the ASE argument lists."""

import dataclasses
import importlib
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest
from ase.constraints import FixAtoms, FixBondLength

from _support.structures import nacl
from fastmace import integrators, optimize
from fastmace.config import DEFAULT_SCHEMES, IntegratorConfig, OptimizerConfig, scheme_parameters
from fastmace.integrators import registry as reg

# --------------------------------------------------------------------------- the registry


def test_discovery_finds_phase_1a_schemes_and_imports_no_torch():
    code = ("import sys; from fastmace.integrators import registry as r; s = r.registry(); "
            "print('torch' in sys.modules); print(*s.schemes('integrator'), sep='\\n')")
    torch_loaded, *schemes = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                            check=True).stdout.split()
    assert {"baoab", "langevin"} <= set(schemes) and torch_loaded == "False"  # later tasks add schemes


def test_the_public_namespaces():
    import fastmace as fm

    assert {"integrators", "optimize"} <= set(fm.__all__)
    assert fm.optimize is optimize and fm.integrators is integrators


def test_the_phase_1a_records():
    langevin, baoab = reg.registry().integrator("langevin"), reg.registry().integrator("baoab")
    assert (langevin.ensembles, langevin.cell_modes, langevin.constraints) == (("nvt",), ("fixed",), ("fixcm",))
    assert (baoab.ensembles, baoab.cell_modes, baoab.variable_cell) == (("npt",), ("hydrostatic", "full"), True)
    assert langevin.load().__name__ == "Langevin" and baoab.load().__name__ == "LangevinBAOAB"
    assert "externalstress" in baoab.constructor_keywords(IntegratorConfig(ensemble="npt"))
    assert langevin.constructor_keywords(IntegratorConfig()) == {}
    hash(langevin)


def _record(**fields):
    base = dict(scheme="toy", implementation=f"{__name__}:ToyIntegrator", ensembles=("nvt",),
                constraints=("fixcm", "fix_atoms"))
    return reg.IntegratorRecord(**{**base, **fields})


def test_records_are_checked():
    with pytest.raises(ValueError, match="constraints"):
        _record(constraints=("rattle",))
    with pytest.raises(ValueError, match="npt"):
        _record(ensembles=("npt",), cell_modes=("fixed",))
    with pytest.raises(ValueError, match="buffer policy"):
        _record(buffers=(("eta", "keep", None),))
    with pytest.raises(ValueError, match="refusals"):
        _record(refusals={"rattle": "no"})


def test_a_scheme_registers_once():
    registry = reg.Registry([_record()])
    registry.register(_record())  # the same record again is fine
    with pytest.raises(ValueError, match="twice"):
        registry.register(_record(non_canonical=True))
    with pytest.raises(KeyError, match="registered: toy"):
        registry.integrator("nhc")


class ToyIntegrator:
    def __init__(self, state, config, *, temperature_K=0.0, extra=None):
        self.state, self.config, self.temperature_K, self.extra = state, config, temperature_K, extra


def toy_keywords(config):
    return {"extra": config.parameter("tau_fs")}


def test_build_integrator_dispatches_from_the_record():
    registry = reg.Registry([_record(keywords=f"{__name__}:toy_keywords")])
    config = IntegratorConfig(scheme="toy", parameters={"tau_fs": 50.0})
    built = reg.build_integrator("state", config, temperature_K=300.0, records=registry)
    assert (built.state, built.config, built.temperature_K, built.extra) == ("state", config, 300.0, 50.0)
    assert reg.build_integrator("state", config, records=registry).temperature_K == 0.0  # None: the class default


# --------------------------------------------------------------------------- rule 1


def _codes(reasons):
    assert all(reason.rule == 1 for reason in reasons)
    return sorted(reason.code for reason in reasons)


def test_phase_1a_configs_pass():
    assert reg.integrator_refusals(IntegratorConfig()) == []
    assert reg.integrator_refusals(IntegratorConfig(ensemble="npt", cell_mode="full"), variable_cell=True) == []


def test_fix_atoms_is_refused_by_langevin_until_4_t04():
    reasons = reg.integrator_refusals(IntegratorConfig(), fix_atoms=True)
    assert _codes(reasons) == ["constraints"]
    assert "FixAtoms" in reasons[0] and "4-T04" in reasons[0]


@pytest.mark.parametrize(("kwargs", "config", "codes"), [
    (dict(fix_atoms=True), dict(fixcm=False), []),
    (dict(fix_atoms=True), dict(fixcm=True), ["fix-atoms-with-fixcm"]),
    (dict(fix_atoms=True, mc=True), dict(fixcm=False), ["mc-with-constraints"]),
    (dict(other_constraints=("FixBondLength",)), dict(), ["constraints"]),
    (dict(mc=True), dict(), []),
    (dict(variable_cell=True), dict(), ["variable-cell-without-barostat"]),
    (dict(), dict(ensemble="npt"), ["scheme-ensemble"]),
    (dict(), dict(dof="ase"), ["scheme-dof"]),
    (dict(), dict(dof="auto"), []),  # "auto" is DOF_AUTO, "fastmace"
])
def test_rule_1_from_the_record(kwargs, config, codes):
    registry = reg.Registry([_record(dof_modes=("fastmace",))])
    reasons = reg.integrator_refusals(IntegratorConfig(scheme="toy", **config), records=registry, **kwargs)
    assert _codes(reasons) == codes


def test_cell_mode_and_fixcm_rows():
    registry = reg.Registry([_record(ensembles=("npt",), cell_modes=("hydrostatic",), constraints=())])
    reasons = reg.integrator_refusals(IntegratorConfig(scheme="toy", ensemble="npt", cell_mode="full"),
                                      records=registry)
    assert _codes(reasons) == ["constraints", "scheme-cell-mode"]  # full cell; fixcm=True not declared


def test_an_unregistered_scheme_is_refused_with_the_registered_ones():
    reasons = reg.integrator_refusals(integrators.NoseHooverChain(timestep_fs=1, temperature_K=300,
                                                                  tdamp_fs=100).to_config(seed=1),
                                      records=reg.Registry([_record()]))
    assert _codes(reasons) == ["scheme-unknown"] and "registered: toy" in reasons[0]


def test_optimizer_rule_1():
    records = reg.Registry([
        reg.OptimizerRecord(scheme="lbfgs", implementation="x:Y", cell_filters=(None, "frechet"),
                            constraints=("fix_atoms",)),
        reg.CellFilterRecord(scheme="frechet", implementation="x:Z",
                             refusals={"fix_atoms": "a variable cell with fixed atoms is refused (D1)"})])
    fixed = optimize.LBFGS().to_config()
    cell = optimize.LBFGS(cell_filter=optimize.FrechetCell()).to_config()
    assert reg.optimizer_refusals(fixed, fix_atoms=True, records=records) == []
    reasons = reg.optimizer_refusals(cell, fix_atoms=True, records=records)
    assert _codes(reasons) == ["constraints"] and "D1" in reasons[0]
    assert _codes(reg.optimizer_refusals(optimize.FIRE().to_config(), records=records)) == ["scheme-unknown"]
    no_filter = reg.Registry([reg.OptimizerRecord(scheme="lbfgs", implementation="x:Y")])
    assert _codes(reg.optimizer_refusals(cell, records=no_filter)) == ["scheme-cell-filter", "scheme-unknown"]


def test_the_planner_takes_rule_1_from_the_registry():
    from test_minimal_planner import codes, plan_label

    from fastmace.errors import PlanRefused

    with pytest.raises(PlanRefused) as caught:
        plan_label("mh1-r2scan", integrator=IntegratorConfig(scheme="no-such-scheme"))
    assert codes(caught.value) == ["scheme-unknown"]
    atoms = nacl()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(PlanRefused) as caught:
        plan_label("mh1-r2scan", atoms=atoms, integrator=IntegratorConfig(fixcm=False))
    assert codes(caught.value) == ["constraints"] and "4-T04" in caught.value.reasons[0]
    plan = plan_label("mh1-r2scan")
    assert plan.integrator.dof == "fastmace" and plan.integrator.scheme == "langevin"
    assert dict((item, choice) for item, choice, _ in plan.decisions)["IntegratorConfig.dof"] == "fastmace"


def test_constraint_names():
    atoms = nacl(repeat=1)
    atoms.set_constraint([FixAtoms(indices=[3, 1]), FixAtoms(mask=[True] + [False] * (len(atoms) - 1)),
                          FixBondLength(0, 2)])
    assert reg.constraint_names(atoms) == ((0, 1, 3), ("FixBondLengths",))  # ASE: FixBondLength is FixBondLengths


# --------------------------------------------------------------------------- degrees of freedom (D2)


@pytest.mark.parametrize(("kwargs", "expected"), [
    (dict(fixcm=True), 3 * 8 - 3), (dict(fixcm=False), 24), (dict(fixcm=True, n_fixed=2), 18),
    (dict(fixcm=True, dof="ase"), 24), (dict(fixcm=False, n_fixed=2, dof="ase"), 18),
    (dict(fixcm=False, n_fixed=2, dof="ase", ase_dof="3n"), 24),
])
def test_d2_degrees_of_freedom(kwargs, expected):
    assert reg.degrees_of_freedom(8, **kwargs) == expected


def test_langevin_keeps_its_count():
    from fastmace.io.trajectory import degrees_of_freedom

    assert degrees_of_freedom(8) == 24
    assert degrees_of_freedom(8, IntegratorConfig()) == 24  # "auto" outside a plan reads as ASE's projection
    assert degrees_of_freedom(8, IntegratorConfig(fixcm_projection="mass_weighted")) == 21
    assert degrees_of_freedom(8, IntegratorConfig(ensemble="npt", fixcm_projection="mass_weighted")) == 24
    assert degrees_of_freedom(8, IntegratorConfig(scheme="no-such-scheme")) == 24  # unregistered: 3N


# --------------------------------------------------------------------------- configs and records


def test_default_schemes_keep_phase_1a():
    assert IntegratorConfig().scheme == "langevin" and IntegratorConfig(ensemble="npt").scheme == "baoab"
    assert IntegratorConfig(ensemble="nve").scheme == DEFAULT_SCHEMES["nve"] == "nve"
    assert integrators.Langevin(temperature_K=300, friction_per_fs=0.01).to_config(seed=1).scheme == "langevin"
    with pytest.raises(ValueError, match="scheme"):
        IntegratorConfig(scheme="")


def test_parameters_are_sorted_hashable_and_survive_json():
    config = integrators.Berendsen(timestep_fs=1, temperature_K=300, taut_fs=100, pressure_GPa=1.0, taup_fs=1000,
                                   compressibility_per_GPa=0.01, mask=[1, 0, 1]).to_config(seed=3)
    assert [name for name, _ in config.parameters] == sorted(name for name, _ in config.parameters)
    assert config.parameter("mask") == (1, 0, 1) and config.parameter("missing", 7) == 7
    hash(config)
    again = IntegratorConfig(**json.loads(json.dumps(dataclasses.asdict(config))))
    assert again == config
    with pytest.raises(ValueError, match="distinct"):
        scheme_parameters([("a", 1), ("a", 2)])
    with pytest.raises(ValueError, match="finite"):
        scheme_parameters({"a": float("nan")})
    with pytest.raises(ValueError, match="tuple of them"):
        scheme_parameters({"a": {"b": 1}})


@pytest.mark.parametrize(("record", "scheme", "ensemble", "cell_mode"), [
    (integrators.NVE(timestep_fs=1.0), "nve", "nve", None),
    (integrators.CSVR(timestep_fs=1.0, temperature_K=300, tau_fs=100), "csvr", "nvt", None),
    (integrators.NoseHooverChain(timestep_fs=1.0, temperature_K=300, tdamp_fs=100), "nhc", "nvt", None),
    (integrators.MTK(timestep_fs=1.0, temperature_K=300, pressure_GPa=0, tdamp_fs=100, pdamp_fs=1000,
                     cell_mode="full"), "mtk", "npt", "full"),
    (integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100), "berendsen", "nvt", None),
    (integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100, pressure_GPa=0, taup_fs=1000,
                           compressibility_per_GPa=0.01), "berendsen", "npt", "hydrostatic"),
])
def test_phase_4_records_map_to_configs(record, scheme, ensemble, cell_mode):
    config = record.to_config(seed=11)
    assert (config.scheme, config.ensemble, config.seed, config.timestep_fs) == (scheme, ensemble, 11, 1.0)
    if cell_mode is not None:
        assert config.cell_mode == cell_mode and config.variable_cell
    assert integrators.RECORDS[type(record).__name__] is type(record)


@pytest.mark.parametrize(("cls", "kwargs"), [
    (integrators.CSVR, dict(timestep_fs=1, temperature_K=300)),  # no tau_fs: no default, as in ASE
    (integrators.MTK, dict(timestep_fs=1, temperature_K=300, pressure_GPa=0, tdamp_fs=100)),
])
def test_time_constants_have_no_defaults(cls, kwargs):
    with pytest.raises(TypeError):
        cls(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(pressure_GPa=1.0, taup_fs=1000),  # no compressibility
    dict(mask=(1, 1, 0)),  # a mask without a pressure
    dict(pressure_GPa=1.0, taup_fs=1000, compressibility_per_GPa=0.01, mask=(1, 2, 0)),
])
def test_berendsen_is_checked(kwargs):
    with pytest.raises(ValueError):
        integrators.Berendsen(timestep_fs=1, temperature_K=300, taut_fs=100, **kwargs)


def test_optimizer_records():
    lbfgs = optimize.LBFGS().to_config(fmax=0.01, steps=5000)
    assert (lbfgs.scheme, lbfgs.fmax, lbfgs.steps, lbfgs.cell_filter, lbfgs.variable_cell) == (
        "lbfgs", 0.01, 5000, None, False)
    assert dict(lbfgs.parameters) == {"maxstep_A": 0.2, "memory": 100, "damping": 1.0, "alpha": 70.0}
    fire = optimize.FIRE(cell_filter=optimize.FrechetCell(scalar_pressure_GPa=1.0)).to_config()
    assert fire.cell_filter == "frechet" and fire.cell_filter_parameter("scalar_pressure_GPa") == 1.0
    assert fire.parameter("Nmin") == 5 and fire.variable_cell
    hash(fire)
    with pytest.raises(TypeError, match="FrechetCell"):
        optimize.LBFGS(cell_filter="frechet")
    with pytest.raises(ValueError, match="cell_filter"):
        OptimizerConfig(scheme="lbfgs", cell_filter_parameters={"mask": None})
    with pytest.raises(ValueError, match="mask"):
        optimize.FrechetCell(mask=(1, 1, 1))


def test_ase_defaults_of_the_optimizer_records():
    from ase.optimize import FIRE, LBFGS

    lbfgs = inspect.signature(LBFGS.__init__).parameters
    assert (optimize.LBFGS().memory, optimize.LBFGS().damping, optimize.LBFGS().alpha) == (
        lbfgs["memory"].default, lbfgs["damping"].default, lbfgs["alpha"].default)
    fire = inspect.signature(FIRE.__init__).parameters
    for name in ("dt", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a", "downhill_check"):
        assert getattr(optimize.FIRE(), name) == fire[name].default, name


# --------------------------------------------------------------------------- ASE argument lists


def _ase_arguments(cls):
    names = []
    for klass in cls.__mro__:
        if klass is object or "__init__" not in vars(klass):
            continue
        for name, parameter in inspect.signature(klass.__init__).parameters.items():
            if name != "self" and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) \
                    and name not in names:
                names.append(name)
    return names


@pytest.mark.parametrize("name", [*__import__("fastmace.ase.md", fromlist=["x"]).ASE_ARGUMENTS,
                                  *__import__("fastmace.ase.md", fromlist=["x"]).ASE_OPTIMIZER_ARGUMENTS])
def test_ase_argument_lists_match_ase(name):
    from fastmace.ase import md

    expected = {**md.ASE_ARGUMENTS, **md.ASE_OPTIMIZER_ARGUMENTS}[name]
    ase_class = getattr(importlib.import_module(md.ASE_MODULES[name]), name)
    assert tuple(_ase_arguments(ase_class)) == expected
    assert name in md.LATER or name in md.ASE_OPTIMIZER_ARGUMENTS


# --------------------------------------------------------------------------- state, DOF view, buffers


torch = pytest.importorskip("torch")


def test_fix_atoms_becomes_a_mask_and_indices():
    from fastmace.state import SimulationState

    atoms = nacl(repeat=1)
    atoms.set_constraint(FixAtoms(indices=[2, 0]))
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    assert state.fixed_indices == (0, 2) and state.n_fixed == 2 and state.constraints == ()
    assert state.free_mask.dtype == torch.float64 and state.free_mask.shape == (len(atoms), 1)
    assert state.free_mask[:, 0].tolist() == [0.0 if i in (0, 2) else 1.0 for i in range(len(atoms))]
    plain = SimulationState.from_atoms(nacl(repeat=1), "cpu", 0)
    assert plain.fixed_indices == () and plain.free_mask is None and plain.n_fixed == 0
    atoms.set_constraint(FixBondLength(0, 1))
    with pytest.raises(ValueError, match="FixBondLengths"):
        SimulationState.from_atoms(atoms, "cpu", 0)


def test_positions_dof_view():
    from fastmace.optimize.base import DOFView, PositionsDOF
    from fastmace.state import SimulationState

    atoms = nacl(repeat=1)
    atoms.set_constraint(FixAtoms(indices=[1]))
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    state.forces.copy_(torch.arange(state.natoms * 3, dtype=torch.float64).view(-1, 3))
    dof = PositionsDOF(state)
    assert isinstance(dof, DOFView) and dof.rows == state.natoms and not dof.variable_cell
    assert dof.x.data_ptr() == state.positions.data_ptr() and dof.x.shape == (state.natoms * 3,)
    gradient = dof.gradient().view(-1, 3)
    assert gradient[1].abs().sum() == 0 and torch.equal(gradient[0], -state.forces[0])
    assert dof.mask.view(-1, 3)[1].tolist() == [False] * 3
    expected = torch.linalg.vector_norm(state.forces[[0, 2, 3, 4, 5, 6, 7]], dim=1).max()
    assert torch.equal(dof.max_force(), expected)
    old = state.positions.clone()
    dof.set_x(dof.x + 1.0)
    assert torch.equal(state.positions[1], old[1]) and torch.equal(state.positions[0], old[0] + 1.0)
    assert dof.energy() is state.energy


def test_scheme_records_declare_their_buffer_policies(monkeypatch):
    from fastmace.engine import registry as runtime

    records = reg.Registry([*reg.registry().records(), _record(buffers=(("chain_q", "save", None),))])
    monkeypatch.setattr(reg, "registry", lambda: records)
    assert runtime.policy_for("integrator.chain_q").restart == "save"
    assert runtime.policy_for("state.free_mask").restart == "derived"
    with pytest.raises(runtime.RegistryError):
        runtime.policy_for("integrator.p_eta")


def test_two_schemes_may_not_give_one_buffer_different_policies():  # 4-T01b
    from fastmace.engine.registry import RegistryError

    derived = _record(scheme="toy_a", buffers=(("pressure", "derived", "Toy.set_pressure"),))
    saved = _record(scheme="toy_b", buffers=(("pressure", "save", None),))
    with pytest.raises(RegistryError, match=r"integrator\.pressure.*toy_a.*toy_b"):
        reg.Registry([derived, saved])
    other_owner = reg.OptimizerRecord(scheme="toy_o", implementation="x:Y", buffers=(("pressure", "save", None),))
    assert len(reg.Registry([derived, other_owner]).records()) == 2  # optimizer.pressure is another buffer


def test_one_buffer_of_two_schemes_lists_both_recompute_functions(monkeypatch):  # 4-T01b
    from fastmace.engine import registry as runtime

    a = _record(scheme="toy_a", buffers=(("chain_r", "reset", "ToyA.start"),))
    b = _record(scheme="toy_b", buffers=(("chain_r", "reset", "ToyB.start"),))
    records = reg.Registry([*reg.registry().records(), b, a])
    monkeypatch.setattr(reg, "registry", lambda: records)
    policy = runtime.policy_for("integrator.chain_r")
    assert (policy.restart, policy.recompute) == ("reset", "ToyA.start | ToyB.start")


def test_a_scheme_may_not_shadow_a_policy_table_buffer(monkeypatch):  # 4-T01b
    from fastmace.engine import registry as runtime

    shadow = _record(scheme="toy_s", buffers=(("eta", "save", None),))  # Langevin's integrator.eta is "derived"
    records = reg.Registry([*reg.registry().records(), shadow])
    monkeypatch.setattr(reg, "registry", lambda: records)
    with pytest.raises(runtime.RegistryError, match=r"integrator\.eta.*toy_s.*derived"):
        runtime.policy_for("integrator.eta")
    same = _record(scheme="toy_s", buffers=(("eta", "derived", None),))  # the same policy is kept
    records = reg.Registry([*reg.discover().records(), same])
    monkeypatch.setattr(reg, "registry", lambda: records)
    assert runtime.policy_for("integrator.eta").restart == "derived"


def test_the_real_records_shadow_no_policy_table_buffer():  # 4-T01b
    from fastmace.engine import registry as runtime

    runtime.scheme_policies()  # raises RegistryError on a shadowing record


def test_the_registered_schemes_have_no_buffer_clash():  # 4-T01b: the real records, every phase 4 task's
    records = reg.discover()  # raises RegistryError on a clash
    seen = {}
    for record in records.records():
        for name, restart, _ in record.buffers:
            seen.setdefault((reg.BUFFER_OWNERS[record.kind], name), set()).add(restart)
    assert {key: policies for key, policies in seen.items() if len(policies) > 1} == {}


# --------------------------------------------------------------------------- restart format 4


def test_format_4_records_fixed_atoms_and_the_scheme(tmp_path):
    from test_restart import make_state, write_full

    from fastmace.io import restart as restart_io

    state, integrator = make_state(npt=False)
    state.fixed_indices = (0, 3)  # as SimulationState.from_atoms sets it with FixAtoms
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    assert restart_io.format_version(stored) == 4
    assert restart_io.fixed_atoms(stored) == (0, 3) and restart_io.scheme(stored) == "langevin"
    fresh, _ = make_state(npt=False, seed=5, steps=1)
    with pytest.raises(ValueError, match="fixed_atoms"):
        restart_io.load_state(stored, fresh)  # another set of fixed atoms is another structure
    fresh.fixed_indices = (0, 3)
    restart_io.load_state(stored, fresh)


def test_format_3_files_load_without_fixed_atoms(tmp_path):
    from test_restart import make_state, write_full

    from fastmace.io import restart as restart_io

    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = dict(restart_io.read_restart(path))
    physical = json.loads(str(stored["physical"]))
    del physical["fixed_atoms"]
    stored.update(physical=np.asarray(json.dumps(physical)), fastmace_restart=np.asarray(3))
    record = json.loads(str(stored["integrator"]))
    del record["config"]["scheme"]
    stored["integrator"] = np.asarray(json.dumps(record))
    assert restart_io.fixed_atoms(stored) == () and restart_io.scheme(stored) == "baoab"
    fresh, fresh_integrator = make_state(seed=5, steps=1)
    restart_io.load_state(stored, fresh, fresh_integrator)
    assert torch.equal(fresh.positions, state.positions)


def test_format_4_checks_the_scheme_state_keys(tmp_path):
    from test_restart import make_state

    from fastmace.io import restart as restart_io

    state, integrator = make_state(npt=False)
    with pytest.raises(ValueError, match="declares"):
        restart_io.state_arrays(state, "fastmace", {"eta": np.zeros(3)}, integrator=integrator)
