"""Displacement moves on the CPU (plan task 3-T10; SPEC §10.4 "Proposals", "Fixed positions versus displacements",
"Graph capture", §8.5 step 4, §12.2 "Mutation events").

- the proposal is uniform in a sphere of radius δ, from fixed counts of the proposal and acceptance streams;
- a displacement within the skin test stays on the attempt path, and its ΔE equals a direct evaluation;
- a displacement beyond skin/2 is evaluated on a list built around the trial geometry, accepted and
  rejected, without looping, and its ΔE equals a direct evaluation;
- a rejected displacement leaves the committed list (candidates, compacted edges, count, reference
  positions) and positions bit for bit;
- an overflowing displacement (edges beyond the active size, or a trial build that grows the capacity)
  is redone with the same numbers: the outcome equals a run that did not overflow;
- the refusals (variable cell, δ beyond ``Workload.max_displacement_A``) and adaptive δ.

Everything runs through ``MCDriver`` (3-T09's driver), as ``Simulation.run(mc=...)`` will; forced decisions
patch ``moves.metropolis`` before attach.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import EngineConfig, IntegratorConfig, PrecisionConfig, mc_stream_seeds
from fastmace.errors import PlanRefused
from fastmace.mc import Displace, MCDriver, moves, streams
from fastmace.mc import displace as dm
from fastmace.mc.moves import K_B_EV_PER_K

pytest.importorskip("mace")
torch.serialization.add_safe_globals([slice])

SKIN = 0.1  # skin/2 = 0.05 A: a displacement of 0.2 A needs a trial rebuild, one of 0.01 A rarely
SEED = 7


@pytest.fixture
def mace_engine(tmp_path, monkeypatch):
    """A fixed-planner graph engine (no capture) of a tiny MH-1 model on NaCl, on the CPU, float64."""
    from fastmace.engine import constraints
    from fastmace.model import loading
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    pytest.importorskip("cuequivariance_torch")
    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    built = {}

    def make(npt=False):
        planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"),
                               precision=PrecisionConfig(dtype="float64"), environ={})
        path = tmp_path / "tiny-mh1.model"
        if not path.exists():
            torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
        loaded = planner.load(str(path), trusted=True)
        atoms = nacl(rattle=0.05, seed=1)
        integrator = (IntegratorConfig(ensemble="npt", pressure_GPa=0.5, noise_batch=7) if npt
                      else IntegratorConfig(noise_batch=7))
        plan = planner.plan(loaded, atoms, integrator, Hardware("cpu", (0, 0), 0))
        plan = dataclasses.replace(plan, neighbours=dataclasses.replace(plan.neighbours, skin=SKIN))
        engine = planner.build(plan, loaded, atoms, temperature_K=300.0, constraints=constraints.ProcessRecord())
        built.update(path=path, plan=plan, loaded=loaded, atoms=atoms, planner=planner)
        return engine

    make.built = built
    return make


def attached(engine, delta, attempts, *, max_displacement_A=None, adaptive=False, seed=SEED):
    """An ``MCDriver`` with one ``Displace`` move, attached to ``engine`` (patch ``moves.metropolis`` first)."""
    driver = MCDriver([Displace(delta_A=delta, every=1, attempts=attempts, adaptive=adaptive)], seed=seed)
    driver.attach(engine, max_displacement_A=delta if max_displacement_A is None else max_displacement_A)
    return driver


def block(driver, engine, temperature_K=300.0):
    """One MC block at ``temperature_K``; the displacement move's entry of the block record."""
    engine.integrator.set_temperature(temperature_K)
    record = driver.run_block(engine, max(1, int(engine.state.step)))
    return record["moves"][0]


def force(monkeypatch, accept):
    """Every decision ``accept`` (before attach: each attempt binds ``moves.metropolis`` when it is built)."""
    monkeypatch.setattr(moves, "metropolis", lambda u, excess, kT: torch.full_like(u, accept, dtype=torch.bool))


def fresh_energy(make, positions):
    """A freshly built float64 force field's energy at ``positions`` (its own list, built there)."""
    from fastmace.model.forces import MACEForceField

    atoms = make.built["atoms"].copy()
    atoms.positions = positions
    field = MACEForceField(make.built["loaded"], atoms, neighbours=make.built["plan"].neighbours,
                           engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"))
    return float(field.energy_only(torch.as_tensor(positions, dtype=torch.float64))[0])




def proposal(driver, natoms):
    """The next attempt's atom and displacement, decoded on the host from the driver's stream offsets."""
    seeds, offsets = driver.seeds, driver.offsets
    numbers = streams.positions(torch.tensor(offsets["proposal"], dtype=torch.int64), dm.PROPOSAL_COUNT)
    atom, d = dm.decode_displace(streams.uniforms(seeds["proposal"], numbers),
                                 streams.normals(seeds["proposal"], numbers), natoms,
                                 torch.tensor([driver.attempts[0].delta_A], dtype=torch.float64))
    u = streams.uniforms(seeds["acceptance"],
                         streams.positions(torch.tensor(offsets["acceptance"], dtype=torch.int64), 1))
    return int(atom[0]), d[0].numpy(), float(u[0])


def compacted_edges(engine):
    """The compacted edges within the cutoff as a set of (sender, receiver, unit shift)."""
    nl = engine.force_field.neighbours
    count = int(nl.count)
    index = nl.edge_index[:, :count].numpy()
    shifts = nl.unit_shifts[:count].numpy().round().astype(int)
    return {(int(s), int(r), *map(int, k)) for s, r, k in zip(index[0], index[1], shifts, strict=True)}


def list_state(engine):
    nl = engine.force_field.neighbours
    out = {name: tensor.clone() for name, tensor in nl.buffers().items()}
    out["num_candidates"] = nl.num_candidates
    return out


def assert_same_list(before, after):
    assert before.keys() == after.keys()
    for name, value in before.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(after[name], value, rtol=0, atol=0, msg=name)
        else:
            assert after[name] == value, name


# --------------------------------------------------------------------------- proposals and records


def test_proposal_is_uniform_in_the_sphere():
    count = 20_000
    seed = mc_stream_seeds(SEED)["proposal"]
    atoms, radii, cosines = [], [], []
    delta = torch.tensor([0.3], dtype=torch.float64)
    for k in range(count):
        numbers = streams.positions(torch.tensor(k * dm.PROPOSAL_COUNT, dtype=torch.int64), dm.PROPOSAL_COUNT)
        atom, d = dm.decode_displace(streams.uniforms(seed, numbers), streams.normals(seed, numbers), 8, delta)
        atoms.append(int(atom[0]))
        r = float(torch.linalg.norm(d))
        radii.append(r)
        cosines.append(float(d[0, 2]) / r)
    radii = np.array(radii)
    assert radii.max() <= 0.3
    # Uniform in the ball: (r / delta)^3 is uniform on [0, 1); the direction is isotropic.
    assert abs(np.mean((radii / 0.3) ** 3) - 0.5) < 0.01
    assert abs(np.mean(cosines)) < 0.02
    assert abs(np.mean(np.square(cosines)) - 1 / 3) < 0.01
    assert np.bincount(atoms, minlength=8).min() > 0.11 * count


def test_counts_per_attempt_are_fixed():
    assert dm.PROPOSAL_COUNT == 5
    assert dm.ACCEPTANCE_COUNT == streams.ACCEPTANCE_COUNT == 1
    assert dm.attempt_counts(64) == {"proposal": 5, "acceptance": 1, "velocity": 0}
    assert dm.COUNTER_KEYS[:3] == ("attempted", "impossible", "accepted")
    assert dm.COUNTER_KEYS[3] == "trial_rebuilds"


def test_refusals_variable_cell_and_delta_beyond_the_workload():
    move = Displace(delta_A=0.2, every=5, attempts=4)
    dm.check_displace(move, max_displacement_A=0.2, variable_cell=False)
    with pytest.raises(PlanRefused) as raised:
        dm.check_displace(move, max_displacement_A=0.1, variable_cell=True)
    assert set(raised.value.codes) == {dm.VARIABLE_CELL_CODE, dm.BEYOND_MAX_CODE}
    with pytest.raises(PlanRefused) as raised:
        dm.check_displace(move, max_displacement_A=None, variable_cell=False)
    assert raised.value.codes == (dm.BEYOND_MAX_CODE,)
    assert dm.VARIABLE_CELL_CODE == "displace-variable-cell"


def test_adapted_delta_rule():
    # delta <- delta * clamp(acc / 0.3, 0.5, 2), at most delta_A and the workload's maximum.
    assert dm.adapted_delta(0.1, 3, 10, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.1)
    assert dm.adapted_delta(0.1, 0, 10, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.05)
    assert dm.adapted_delta(0.1, 10, 10, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.2)
    assert dm.adapted_delta(0.4, 10, 10, delta_A=0.5, max_displacement_A=0.45) == pytest.approx(0.45)
    assert dm.adapted_delta(0.1, 0, 0, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.1)  # nothing attempted


# --------------------------------------------------------------------------- attempts on an engine


def test_attach_refuses_a_variable_cell_and_a_delta_beyond_the_workload(mace_engine):
    with pytest.raises(PlanRefused) as raised:
        attached(mace_engine(npt=True), 0.05, 4)
    assert raised.value.codes == (dm.VARIABLE_CELL_CODE,)
    engine = mace_engine()
    with pytest.raises(PlanRefused) as raised:
        attached(engine, 0.2, 4, max_displacement_A=0.1)
    assert raised.value.codes == (dm.BEYOND_MAX_CODE,)
    driver = MCDriver([Displace(delta_A=0.2, every=1, attempts=4)], seed=SEED)
    with pytest.raises(PlanRefused) as raised:
        driver.attach(engine)  # no Workload.max_displacement_A given
    assert raised.value.codes == (dm.BEYOND_MAX_CODE,)


# --------------------------------------------------------------------------- attempts through the driver


def run_one_by_one(make, engine, driver, count, temperature_K):
    """``count`` blocks of one attempt each, each checked against a direct evaluation and the acceptance rule."""
    natoms = engine.state.positions.shape[0]
    kT = K_B_EV_PER_K * temperature_K
    attempt = driver.attempts[0]
    outcomes = []
    for _ in range(count):
        before = engine.state.positions.clone()
        reference = engine.force_field.neighbours.reference_positions.clone()
        atom, d, u = proposal(driver, natoms)
        trial = before.clone()
        trial[atom] += torch.from_numpy(d)
        needs = bool(((trial - reference) ** 2).sum(dim=1).max() > (SKIN / 2) ** 2)
        rebuilds = engine.force_field.neighbours.rebuilds
        entry = block(driver, engine, temperature_K)
        assert entry["attempted"] == 1 and entry["impossible"] == 0
        assert entry["trial_rebuilds"] == int(needs)
        accepted = bool(entry["accepted"])
        # One trial build, no loop (an acceptance adds the block's canonical rebuild).
        assert engine.force_field.neighbours.rebuilds == rebuilds + int(needs) + int(accepted)
        expected = fresh_energy(make, trial.numpy())
        assert float(attempt.trial_energy[0]) == pytest.approx(expected, rel=1e-12, abs=1e-12)
        delta_e = float(attempt.trial_energy[0]) - driver.last_block["baseline_eV"]
        assert accepted == (u == 0 or math.log(u) < -delta_e / kT)
        torch.testing.assert_close(engine.state.positions, trial if accepted else before, rtol=0, atol=0)
        outcomes.append((accepted, needs))
    return outcomes


def test_small_displacements_stay_on_the_attempt_path(mace_engine):
    engine = mace_engine()
    driver = attached(engine, 0.01, 1)
    outcomes = run_one_by_one(mace_engine, engine, driver, 6, temperature_K=300.0)
    assert not any(needs for _, needs in outcomes)
    assert driver.attempts[0].trial_rebuilds == 0


def test_displacement_beyond_half_the_skin_uses_a_list_around_the_trial(mace_engine):
    engine = mace_engine()
    driver = attached(engine, 0.2, 1)
    # A low temperature: downhill moves are accepted, uphill ones rejected; both happen in these attempts.
    outcomes = run_one_by_one(mace_engine, engine, driver, 10, temperature_K=5.0)
    rebuilt = [accepted for accepted, needs in outcomes if needs]
    assert True in rebuilt and False in rebuilt, outcomes
    assert driver.info()["displacements"]["displace:0"]["trial_rebuilds"] == sum(needs for _, needs in outcomes)


def test_accepted_trials_chain_on_their_trial_lists(mace_engine, monkeypatch):
    """Every attempt accepted in one block: each later trial runs on the list built around the previous
    accepted geometry, and the block's last energy equals a direct evaluation of the final positions."""
    force(monkeypatch, True)
    engine = mace_engine()
    driver = attached(engine, 0.2, 5)
    entry = block(driver, engine)
    assert entry["attempted"] == entry["accepted"] == 5 and entry["trial_rebuilds"] > 0
    final = fresh_energy(mace_engine, engine.state.positions.numpy())
    assert driver.last_block["energy_eV"] == pytest.approx(final, rel=1e-12, abs=1e-12)
    assert float(engine.state.energy[0]) == pytest.approx(final, rel=1e-12, abs=1e-12)  # the canonical point


def test_a_block_runs_through_halted_attempts(mace_engine, monkeypatch):
    force(monkeypatch, False)
    engine = mace_engine()
    driver = attached(engine, 0.2, 8)
    entry = block(driver, engine)
    assert entry["attempted"] == 8 and entry["accepted"] == 0 and 0 < entry["trial_rebuilds"] <= 8
    assert driver.offsets == {"proposal": 8 * dm.PROPOSAL_COUNT, "acceptance": 8, "velocity": 0}


def test_a_rejected_displacement_leaves_the_committed_list(mace_engine, monkeypatch):
    force(monkeypatch, False)
    for delta in (0.01, 0.2):  # without and with a trial rebuild
        engine = mace_engine()
        driver = attached(engine, delta, 5)
        before, positions = list_state(engine), engine.state.positions.clone()
        entry = block(driver, engine)
        assert entry["accepted"] == 0 and entry["attempted"] == 5
        assert (entry["trial_rebuilds"] > 0) == (delta > SKIN / 2)
        assert_same_list(before, list_state(engine))
        torch.testing.assert_close(engine.state.positions, positions, rtol=0, atol=0)


def test_stream_offsets_do_not_depend_on_decisions(mace_engine, monkeypatch):
    offsets = []
    for accept in (True, False):
        force(monkeypatch, accept)
        engine = mace_engine()
        driver = attached(engine, 0.2, 3)
        assert block(driver, engine)["accepted"] == (3 if accept else 0)
        offsets.append(dict(driver.offsets))
    assert offsets[0] == offsets[1] == {"proposal": 3 * dm.PROPOSAL_COUNT, "acceptance": 3, "velocity": 0}


def test_an_edge_overflow_is_redone_with_the_same_numbers(mace_engine, monkeypatch):
    """The active edges are shrunk below the count after the block's baseline (a forced overflow, as in 3-T04's
    tests): the first attempt overflows, is restored and redone with the same numbers on enough edges."""
    outcomes = []
    for shrink in (False, True):
        engine = mace_engine()
        driver = attached(engine, 0.01, 4)
        if shrink:
            baseline = driver.evaluator.baseline

            def shrunk(baseline=baseline, engine=engine):
                out = baseline()
                engine.force_field.use_edges(int(engine.force_field.edge_count) - 20)
                return out

            monkeypatch.setattr(driver.evaluator, "baseline", shrunk)
        redos = engine.info()["overflows_redone"]
        entry = block(driver, engine)
        outcomes.append((entry, engine.state.positions.clone(), dict(driver.offsets),
                         driver.last_block["energy_eV"], engine.info()["overflows_redone"] - redos))
        assert not engine.overflowed()
    (plain, plain_positions, plain_offsets, plain_e, plain_redos), (entry, positions, offsets, energy, redos) = outcomes
    # On the CPU the edges are exact counts, so a displacement that adds a pair overflows in the plain run too.
    assert redos > plain_redos
    for key in ("attempted", "accepted", "trial_rebuilds"):
        assert entry[key] == plain[key], key
    torch.testing.assert_close(positions, plain_positions, rtol=0, atol=0)
    assert offsets == plain_offsets
    assert energy == pytest.approx(plain_e, rel=1e-12, abs=1e-12)


def test_a_trial_build_that_grows_the_capacity_is_redone_with_the_same_numbers(mace_engine, monkeypatch):
    """The neighbour capacity is set to the candidates of the last build (no headroom): a trial build with
    more pairs grows it, which moves the buffers; the attempt is restored and redone with the same numbers."""
    force(monkeypatch, True)
    outcomes = []
    for tight in (False, True):
        engine = mace_engine()
        if tight:
            nl = engine.force_field.neighbours
            nl.set_capacity(nl.num_candidates)
            engine.canonicalise()  # build into the exact-size buffers
            nl.headroom = 1.0  # no headroom on growth either, so later trials with more pairs grow again
        driver = attached(engine, 0.2, 6)
        redos = engine.info()["overflows_redone"]
        entry = block(driver, engine)
        outcomes.append((entry, engine.state.positions.clone(), engine.info()["overflows_redone"] - redos))
    (plain, plain_positions, _), (entry, positions, redos) = outcomes
    assert redos >= 1
    for key in ("attempted", "accepted", "trial_rebuilds"):
        assert entry[key] == plain[key], key
    torch.testing.assert_close(positions, plain_positions, rtol=0, atol=0)


def test_adaptive_delta_changes_at_the_end_of_a_block_and_is_recorded(mace_engine, monkeypatch):
    from fastmace.io import restart

    force(monkeypatch, False)
    engine = mace_engine()
    driver = attached(engine, 0.2, 4, adaptive=True)
    entry = block(driver, engine)
    assert entry["delta_A"] == pytest.approx(0.2)  # the δ the block used
    assert entry["adapted_delta_A"] == pytest.approx(0.1)  # clamp(0 / 0.3) = 0.5
    state = driver.state_dict()
    assert state["delta_A"] == {"displace:0": pytest.approx(0.1)}
    assert driver.info()["displacements"]["displace:0"]["delta_history"] == pytest.approx([0.2, 0.1])
    restart._mc_checked(state)  # a valid restart record
    # A resumed driver continues with the adapted δ, attached before or after the record is loaded.
    for load_first in (True, False):
        again = MCDriver([Displace(delta_A=0.2, every=1, attempts=4, adaptive=True)], seed=SEED)
        if load_first:
            again.load_state_dict(state)
            again.attach(mace_engine(), max_displacement_A=0.2)
        else:
            again.attach(mace_engine(), max_displacement_A=0.2)
            again.load_state_dict(state)
        assert again.attempts[0].delta_A == pytest.approx(0.1)
        assert float(again.attempts[0].delta[0]) == pytest.approx(0.1)
    fixed = attached(mace_engine(), 0.2, 4)
    entry = block(fixed, fixed.engine)
    assert fixed.attempts[0].delta_A == pytest.approx(0.2) and "adapted_delta_A" not in entry


def test_md_after_a_rejected_block_equals_md_without_it(mace_engine, monkeypatch):
    """A rejected block leaves nothing behind: the following MD steps are bitwise those of an engine that never
    ran it, and the registry check against a fresh construction finds no difference (SPEC §8.5, §12.2)."""
    from fastmace import events

    force(monkeypatch, False)
    engine, untouched = mace_engine(), mace_engine()
    driver = attached(engine, 0.2, 6)
    entry = block(driver, engine)
    assert entry["accepted"] == 0 and entry["trial_rebuilds"] > 0
    assert events.check_against(engine, untouched) == []
    for _ in range(3):
        engine.step()
        untouched.step()
    torch.testing.assert_close(engine.state.positions, untouched.state.positions, rtol=0, atol=0)
    torch.testing.assert_close(engine.state.velocities, untouched.state.velocities, rtol=0, atol=0)
