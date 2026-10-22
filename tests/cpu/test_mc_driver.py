"""The MC driver: swaps and transmutations (plan task 3-T09; SPEC §10.4, §8.5, §12.2 "Mutation events").

On the CPU in float64 with a tiny MH-1 model on NaCl (superset Na, Cl, K), through a ``Simulation``'s
engine:

- an accepted swap and an accepted transmutation leave a state equal to a fresh construction from the
  committed state (the derived-state registry check and the first MD step after it, bit for bit); a
  rejected one restores the state bit for bit;
- impossible attempts change nothing and consume their fixed counts;
- stream positions never depend on decisions;
- the device counters match the decisions;
- the momentum treatments, the restart record and its resume, the block record;
- the move checks: the charge/spin parity refusal (and "not applicable" without charge inputs), the
  ``fixcm`` refusals and rule 2; a tiny OMOL-0-style model with charge and spin inputs moves its total
  charge by Δq inside the attempt.
"""

from __future__ import annotations

import pytest
import torch
from test_canonical_restart import assert_same_registered_state, registered
from test_events import take_dynamics

from _support.mc_fixtures import cpu_planner, mc_simulation, tiny_loaded
from _support.structures import nacl
from fastmace import events, integrators
from fastmace.config import MC_STREAMS, mc_stream_seeds
from fastmace.errors import CHARGE_PARITY_CODE, ChargeParityRefused, PlanRefused, SpeciesNotInSuperset
from fastmace.mc import MCDriver, Swap, Transmute, momenta, moves, streams
from fastmace.planner.minimal import check_mc_moves

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA, CL, K, MG = 11, 17, 19, 12
SUPERSET = (NA, CL, K)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.01)
LANGEVIN_FREE = integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.01, fixcm=False)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    return tiny_loaded(tmp_path_factory.mktemp("models"), SUPERSET)


@pytest.fixture(autouse=True)
def planner(monkeypatch):
    return cpu_planner(monkeypatch)


def structure(numbers=None):
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    if numbers is not None:
        atoms.set_atomic_numbers(numbers)
    return atoms


def simulation(loaded, atoms=None, *, integrator=LANGEVIN, superset=SUPERSET, seed=1):
    return mc_simulation(loaded, structure() if atoms is None else atoms, superset, integrator=integrator, seed=seed)


def force(monkeypatch, accept: bool):
    """Fault injection (tests only): every Metropolis decision accepts, or rejects."""
    monkeypatch.setattr(moves, "metropolis", lambda u, excess, kT: torch.full_like(u, accept, dtype=torch.bool))


def fresh(loaded, sim, *, integrator=LANGEVIN, superset=SUPERSET):
    """A plain construction from ``sim``'s committed state, with its velocities and noise stream."""
    engine = sim._engine
    state = engine.state
    atoms = structure(state.species.numpy())
    atoms.set_cell(state.cell.numpy())
    atoms.set_positions(state.positions.numpy())
    atoms.set_masses(state.masses[:, 0].numpy())
    other = simulation(loaded, atoms, integrator=integrator, superset=superset)
    take_dynamics(other._engine, engine)
    return other


def assert_equals_fresh(loaded, sim, *, integrator=LANGEVIN, superset=SUPERSET):
    other = fresh(loaded, sim, integrator=integrator, superset=superset)
    engine = sim._engine
    assert events.check_against(engine, other._engine) == []
    torch.testing.assert_close(engine.state.forces, other._engine.state.forces, rtol=0, atol=0)
    engine.step()
    other._engine.step()
    for name in ("positions", "velocities", "forces", "energy"):
        torch.testing.assert_close(getattr(engine.state, name), getattr(other._engine.state, name), rtol=0, atol=0,
                                   msg=name)


def first_proposal(sim, seed, move):
    """The first attempt's atoms and possibility, decoded from the streams directly (the documented layout)."""
    state = sim._engine.state
    n = state.natoms
    seeds = mc_stream_seeds(seed)
    if isinstance(move, Swap):
        u = streams.uniforms(seeds["proposal"], torch.arange(2 * n + 1))
        pairs = torch.tensor([[moves.species_number(a), moves.species_number(b)] for a, b in move.pairs])
        return streams.decode_swap(u, state.species, pairs)
    u = streams.uniforms(seeds["proposal"], torch.arange(n + 1))
    numbers = [moves.species_number(s) for s in move.species]
    set_ = torch.tensor(numbers)
    members = torch.zeros(119, dtype=torch.bool)
    members[set_] = True
    position = torch.zeros(119, dtype=torch.int64)
    position[set_] = torch.arange(len(numbers))
    return streams.decode_transmute(u, state.species, set_, members, position)


# --------------------------------------------------------------------------- swaps


def test_an_accepted_swap_equals_a_fresh_construction(loaded, monkeypatch):
    force(monkeypatch, True)
    sim = simulation(loaded)
    sim.run(steps=3)
    engine = sim._engine
    move = Swap([("Na", "Cl")], every=3, attempts=1)
    driver = MCDriver([move], seed=4)
    driver.attach(engine)
    (i, j), possible = first_proposal(sim, 4, move)
    assert bool(possible)
    species, masses = engine.state.species.clone(), engine.state.masses.clone()
    velocities = engine.state.velocities.clone()
    record = driver.run_block(engine, 3)
    assert record["moves"] == [{"move": "swap", "attempts": 1, "attempted": 1, "impossible": 0, "accepted": 1}]
    state = engine.state
    assert (int(state.species[i]), int(state.species[j])) == (int(species[j]), int(species[i]))
    assert (float(state.masses[i]), float(state.masses[j])) == (float(masses[j]), float(masses[i]))
    # exchange: the two atoms swap velocities (then the canonical point keeps them).
    torch.testing.assert_close(state.velocities[i], velocities[j], rtol=0, atol=0)
    torch.testing.assert_close(state.velocities[j], velocities[i], rtol=0, atol=0)
    assert int((state.species != species).sum()) == 2
    assert_equals_fresh(loaded, sim)


def test_a_rejected_swap_restores_the_state_exactly(loaded, monkeypatch):
    force(monkeypatch, False)
    sim = simulation(loaded)
    sim.run(steps=2)
    engine = sim._engine
    engine.canonicalise()
    before = registered(engine)
    driver = MCDriver([Swap([("Na", "Cl")], every=2, attempts=7)], seed=4)
    driver.attach(engine)
    record = driver.run_block(engine, 2)
    assert record["moves"][0]["accepted"] == 0 and record["moves"][0]["attempted"] == 7
    assert record["energy_eV"] == record["baseline_eV"]
    assert_same_registered_state(registered(engine), before)


# --------------------------------------------------------------------------- transmutations


@pytest.mark.parametrize(("integrator", "velocities"), [(LANGEVIN, None), (LANGEVIN_FREE, "rescale"),
                                                        (LANGEVIN_FREE, "redraw")])
def test_an_accepted_transmutation_equals_a_fresh_construction(loaded, monkeypatch, integrator, velocities):
    force(monkeypatch, True)
    sim = simulation(loaded, integrator=integrator)
    sim.run(steps=2)
    engine = sim._engine
    move = Transmute(["Na", "K"], every=2, attempts=1, velocities=velocities)
    driver = MCDriver([move], seed=9)
    driver.attach(engine)
    atom, new, possible = first_proposal(sim, 9, move)
    assert bool(possible) and int(new) == K
    v_old = engine.state.velocities.clone()
    m_old = engine.state.masses[atom].clone()
    driver.run_block(engine, 2)
    state = engine.state
    assert int(state.species[atom]) == K
    from ase.data import atomic_masses

    assert float(state.masses[atom]) == float(atomic_masses[K])
    if velocities == "rescale":
        torch.testing.assert_close(state.velocities[atom], v_old[atom] * torch.sqrt(m_old / state.masses[atom]),
                                   rtol=0, atol=0)
    elif velocities == "redraw":
        seed = mc_stream_seeds(9)["velocity"]
        z = streams.normals(seed, torch.arange(3))
        torch.testing.assert_close(state.velocities[atom], torch.sqrt(driver.streams.kT / state.masses[atom]) * z,
                                   rtol=0, atol=0)
    else:  # fixcm: the default redraw_all, zero total momentum, numbers after the attempt's (none)
        assert driver.attempts[0].velocities == "redraw_all"
        momentum = (state.masses * state.velocities).sum(dim=0)
        torch.testing.assert_close(momentum, torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-12)
        z = streams.normals(mc_stream_seeds(9)["velocity"], torch.arange(3 * state.natoms)).view(-1, 3)
        torch.testing.assert_close(state.velocities, momenta.redraw_all(state.masses, z, driver.streams.kT),
                                   rtol=0, atol=0)
        assert driver.offsets["velocity"] == 3 * state.natoms
    assert_equals_fresh(loaded, sim, integrator=integrator)


def test_a_rejected_transmutation_restores_the_state_exactly(loaded, monkeypatch):
    force(monkeypatch, False)
    sim = simulation(loaded, integrator=LANGEVIN_FREE)
    sim.run(steps=2)
    engine = sim._engine
    engine.canonicalise()
    before = registered(engine)
    driver = MCDriver([Transmute(["Na", "K"], every=2, attempts=5, velocities="redraw")], seed=9)
    driver.attach(engine)
    driver.run_block(engine, 2)
    assert driver.counters["transmute"] == {"attempted": 5, "impossible": 0, "accepted": 0}
    assert_same_registered_state(registered(engine), before)
    assert driver.offsets == {"proposal": 5 * (8 + 1), "acceptance": 5, "velocity": 5 * 3}


# --------------------------------------------------------------------------- streams and counters


def test_impossible_attempts_change_nothing_and_consume_their_counts(loaded, monkeypatch):
    force(monkeypatch, True)  # even an accepting rule cannot accept an impossible attempt
    sim = simulation(loaded, integrator=LANGEVIN_FREE)
    sim.run(steps=2)
    engine = sim._engine
    engine.canonicalise()
    before = registered(engine)
    driver = MCDriver([Swap([("Na", "K")], every=2, attempts=6, velocities="redraw")],
                      seed=2)  # no K in the structure
    driver.attach(engine)
    record = driver.run_block(engine, 2)
    assert record["moves"][0] == {"move": "swap", "attempts": 6, "attempted": 6, "impossible": 6, "accepted": 0}
    assert_same_registered_state(registered(engine), before)
    assert driver.offsets == {"proposal": 6 * 17, "acceptance": 6, "velocity": 6 * 6}


def test_stream_positions_never_depend_on_decisions(loaded, monkeypatch):
    blocks = [Swap([("Na", "Cl")], every=2, attempts=3), Transmute(["Na", "K"], every=4, attempts=2)]
    positions = {}
    for accept in (True, False):
        force(monkeypatch, accept)
        sim = simulation(loaded)
        driver = MCDriver(blocks, seed=5)
        driver.attach(sim._engine)
        trail = []
        for step in (2, 4, 6):
            sim.run(steps=2)
            driver.run_block(sim._engine, step)
            trail.append(dict(driver.offsets))
        positions[accept] = trail
        assert sum(driver.counters[name]["accepted"] for name in driver.counters) == (3 * 3 + 2 if accept else 0)
    assert positions[True] == positions[False]
    assert positions[True][-1] == {"proposal": 3 * 3 * 17 + 2 * 9, "acceptance": 3 * 3 + 2,
                                   "velocity": 3 * 8}  # redraw_all: 3N once in the transmutation block


def test_the_counters_match_the_decisions(loaded, monkeypatch):
    decisions = []
    run = moves.Attempt._run

    def recording(self, torch_module):
        before = self.engine.state.species.clone()
        run(self, torch_module)
        changed = bool((self.engine.state.species != before).any())
        decisions.append((self.move.move, bool(self.accepted), changed))

    monkeypatch.setattr(moves.Attempt, "_run", recording)
    sim = simulation(loaded, integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=3000.0,
                                                             friction_per_fs=0.01))
    driver = MCDriver([Swap([("Na", "Cl")], every=2, attempts=4), Transmute(["Na", "K"], every=2, attempts=4)],
                      seed=11)
    driver.attach(sim._engine)
    for step in (2, 4, 6):
        sim.run(steps=2)
        driver.run_block(sim._engine, step)
    assert all(accepted == changed for _, accepted, changed in decisions)
    for name in ("swap", "transmute"):
        mine = [accepted for move, accepted, _ in decisions if move == name]
        assert driver.counters[name] == {"attempted": 12, "impossible": 0, "accepted": sum(mine)}
    assert 0 < sum(accepted for _, accepted, _ in decisions) < len(decisions)  # a real mix of decisions


def test_block_record_and_due_moves(loaded):
    sim = simulation(loaded)
    swap, transmute = Swap([("Na", "Cl")], every=2, attempts=2), Transmute(["Na", "K"], every=3, attempts=1)
    driver = MCDriver([swap, transmute], seed=1)
    driver.attach(sim._engine)
    assert driver.due(0) == () and driver.due(2) == (swap,) and driver.due(6) == (swap, transmute)
    assert driver.run_block(sim._engine, 5) is None
    sim.run(steps=6)
    record = driver.run_block(sim._engine, 6)
    assert [entry["move"] for entry in record["moves"]] == ["swap", "transmute"]
    assert record["step"] == 6 and record["device_ms"] is None and driver.last_block is record
    assert record["temperature_K"] == 600.0
    assert isinstance(record["baseline_eV"], float) and isinstance(record["energy_eV"], float)
    assert sim._engine.state.mc == driver.state_dict()
    assert driver.decisions == (("parity", "not applicable", "the model takes no charge or spin inputs"),)


# --------------------------------------------------------------------------- restart


def test_the_restart_record_resumes_the_chain(loaded):
    from fastmace.io import restart

    moves_ = [Swap([("Na", "Cl")], every=2, attempts=3), Transmute(["Na", "K"], every=2, attempts=2,
                                                                    mu={"K": 0.2})]
    runs = []
    for interrupted in (False, True):
        sim = simulation(loaded, integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=2000.0,
                                                                 friction_per_fs=0.01))
        driver = MCDriver(moves_, seed=3)
        driver.attach(sim._engine)
        for step in (2, 4, 6):
            sim.run(steps=2)
            if interrupted and step == 4:
                record = restart._mc_checked(driver.state_dict())
                assert record["streams"]["proposal"] == {"seed": mc_stream_seeds(3)["proposal"],
                                                         "offset": driver.offsets["proposal"]}
                driver = MCDriver(moves_, seed=3)
                driver.load_state_dict(record)
                driver.attach(sim._engine)
                driver.load_state_dict(record)
            driver.run_block(sim._engine, step)
        runs.append((driver.state_dict(), sim._engine.state.species.clone(), sim._engine.state.velocities.clone()))
    assert runs[0][0] == runs[1][0]
    torch.testing.assert_close(runs[0][1], runs[1][1], rtol=0, atol=0)
    torch.testing.assert_close(runs[0][2], runs[1][2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="moves"):
        MCDriver([Swap([("Na", "Cl")], every=2, attempts=1)], seed=3).load_state_dict(runs[0][0])
    with pytest.raises(ValueError, match="seed"):
        MCDriver(moves_, seed=4).load_state_dict(runs[0][0])


# --------------------------------------------------------------------------- momenta


def test_momentum_treatments():
    v = torch.tensor([[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]], dtype=torch.float64)
    m_old = torch.tensor([[1.0], [4.0]], dtype=torch.float64)
    m_new = m_old.flip(0)
    kT = torch.tensor([0.05], dtype=torch.float64)
    z = torch.ones_like(v)
    torch.testing.assert_close(momenta.exchange(v, m_old, m_new, z, kT), v.flip(0), rtol=0, atol=0)
    rescaled = momenta.rescale(v, m_old, m_new, z, kT)
    torch.testing.assert_close((m_new * rescaled ** 2).sum(dim=1), (m_old * v ** 2).sum(dim=1))  # per-atom KE
    torch.testing.assert_close(momenta.redraw(v, m_old, m_new, z, kT), torch.sqrt(kT / m_new) * z)
    masses = torch.tensor([[1.0], [4.0], [9.0]], dtype=torch.float64)
    normals = streams.normals(7, torch.arange(9)).view(3, 3)
    out = momenta.redraw_all(masses, normals, kT)
    torch.testing.assert_close((masses * out).sum(dim=0), torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-15)
    # Exchange keeps the kinetic energy and the total momentum; the §10.4 example of unchanged velocities does not.
    assert set(momenta.PER_MOVE) == {"exchange", "rescale", "redraw"}


# --------------------------------------------------------------------------- checks


def test_parity_rule_with_charge_inputs():
    na_mg = Transmute(["Na", "Mg"], every=1, attempts=1, velocities="redraw_all")
    with pytest.raises(ChargeParityRefused) as raised:
        check_mc_moves([na_mg], superset=(NA, MG, CL), charge_inputs=True, fixcm=True)
    assert [reason.code for reason in raised.value.reasons] == [CHARGE_PARITY_CODE]
    assert isinstance(raised.value, PlanRefused)
    with_charges = Transmute(["Na", "Mg"], every=1, attempts=1, velocities="redraw_all",
                             formal_charges={"Na": 1, "Mg": 2})
    assert check_mc_moves([with_charges], superset=(NA, MG, CL), charge_inputs=True, fixcm=True)[0][:2] == (
        "parity", "checked")
    # O <-> Cl: dZ = 9, so dq must be odd.
    o_cl = Transmute(["O", "Cl"], every=1, attempts=1, velocities="redraw_all", formal_charges={"O": -2, "Cl": -1})
    check_mc_moves([o_cl], superset=(8, 17), charge_inputs=True, fixcm=True)
    with pytest.raises(ChargeParityRefused):
        check_mc_moves([Transmute(["O", "Cl"], every=1, attempts=1, velocities="redraw_all",
                                  formal_charges={"O": -1, "Cl": -1})], superset=(8, 17), charge_inputs=True,
                       fixcm=True)
    # Without charge or spin inputs the rule does not apply, and that is recorded.
    assert check_mc_moves([na_mg], superset=(NA, MG, CL), charge_inputs=False, fixcm=True) == (
        ("parity", "not applicable", "the model takes no charge or spin inputs"),)
    assert check_mc_moves([Swap([("Na", "Cl")], 1, 1)], superset=(NA, CL), charge_inputs=True, fixcm=True) == ()


def test_fixcm_and_superset_checks():
    for velocities in ("rescale", "redraw"):
        with pytest.raises(PlanRefused) as raised:
            check_mc_moves([Swap([("Na", "Cl")], 1, 1, velocities=velocities)], superset=(NA, CL),
                           charge_inputs=False, fixcm=True)
        assert [reason.code for reason in raised.value.reasons] == ["mc-velocities-fixcm"]
        check_mc_moves([Swap([("Na", "Cl")], 1, 1, velocities=velocities)], superset=(NA, CL), charge_inputs=False,
                       fixcm=False)
    with pytest.raises(PlanRefused, match="fixcm"):
        check_mc_moves([Transmute(["Na", "K"], 1, 1, velocities="rescale")], superset=SUPERSET, charge_inputs=False,
                       fixcm=True)
    check_mc_moves([Transmute(["Na", "K"], 1, 1)], superset=SUPERSET, charge_inputs=False, fixcm=True)  # redraw_all
    with pytest.raises(SpeciesNotInSuperset) as raised:
        check_mc_moves([Swap([("Na", "Rb")], 1, 1)], superset=SUPERSET, charge_inputs=False, fixcm=True)
    assert [reason.code for reason in raised.value.reasons] == ["species-not-in-superset"]


def test_attach_refuses_a_move_the_checks_refuse(loaded):
    sim = simulation(loaded)
    with pytest.raises(PlanRefused, match="fixcm"):
        MCDriver([Transmute(["Na", "K"], 1, 1, velocities="rescale")], seed=1).attach(sim._engine)
    with pytest.raises(SpeciesNotInSuperset):
        MCDriver([Swap([("Na", "Rb")], 1, 1)], seed=1).attach(sim._engine)
    with pytest.raises(NotImplementedError, match="3-T10"):
        MCDriver([fm_displace()], seed=1).attach(sim._engine)
    with pytest.raises(ValueError, match="max_attempts_per_block"):
        from fastmace.config import MCConfig

        MCDriver([Swap([("Na", "Cl")], 1, 3)], seed=1, config=MCConfig(max_attempts_per_block=2))


def fm_displace():
    from fastmace.mc import Displace

    return Displace(0.1, 1, 1)


# --------------------------------------------------------------------------- charge and spin inputs


@pytest.fixture(scope="module")
def omol_dir(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    return tmp_path_factory.mktemp("omol")


def test_charge_inputs_refuse_an_odd_set_and_move_the_total_charge(omol_dir, monkeypatch):
    superset = (NA, MG, CL)
    loaded = tiny_loaded(omol_dir, superset, family="omol", charge=0, spin_multiplicity=1)
    sim = mc_simulation(loaded, structure(), superset, integrator=LANGEVIN)
    engine = sim._engine
    assert {"total_charge", "total_spin"} <= set(engine.force_field.derived_buffers())
    with pytest.raises(ChargeParityRefused):
        MCDriver([Transmute(["Na", "Mg"], 1, 1)], seed=1).attach(engine)
    force(monkeypatch, True)
    sim.run(steps=2)
    driver = MCDriver([Transmute(["Na", "Mg"], every=2, attempts=1, formal_charges={"Na": 1, "Mg": 2})], seed=6)
    driver.attach(engine)
    assert driver.decisions[0][:2] == ("parity", "checked")
    driver.run_block(engine, 2)
    assert int((engine.state.species == MG).sum()) == 1
    assert engine.state.total_charge == 1
    assert float(engine.force_field.derived_buffers()["total_charge"].reshape(-1)[0]) == 1.0
    # A fresh construction with the new total charge equals it.
    charged = tiny_loaded(omol_dir, superset, family="omol", charge=1, spin_multiplicity=1)
    state = engine.state
    atoms = structure(state.species.numpy())
    atoms.set_positions(state.positions.numpy())
    atoms.set_masses(state.masses[:, 0].numpy())
    other = mc_simulation(charged, atoms, superset, integrator=LANGEVIN)
    take_dynamics(other._engine, engine)
    # state.total_charge is the host record of a total-charge change (as Simulation.set_charges keeps it); a
    # plain construction leaves it None.
    other._engine.state.total_charge = engine.state.total_charge
    assert events.check_against(engine, other._engine) == []
    assert tuple(MC_STREAMS) == tuple(driver.offsets)
    # A resume restores the recorded total charge into an engine built with the starting one (3-T08's open point).
    from fastmace.io import restart

    path = omol_dir / "after-mc.npz"
    restart.write_engine_restart(path, engine, name="graph")
    resumed = mc_simulation(loaded, atoms, superset, integrator=LANGEVIN)
    restart.resume_engine(restart.read_restart(path), resumed._engine)
    assert resumed._engine.state.total_charge == 1
    torch.testing.assert_close(resumed._engine.force_field.derived_buffers()["total_charge"],
                               engine.force_field.derived_buffers()["total_charge"], rtol=0, atol=0)
    assert restart.mc_record(restart.read_restart(path)) == driver.state_dict()
