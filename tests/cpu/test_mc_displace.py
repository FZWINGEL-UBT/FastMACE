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
from fastmace.mc import Displace, streams
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


def attempts_of(engine, delta, *, max_displacement_A=None, adaptive=False):
    from fastmace.mc.evaluator import MCEvaluator

    evaluator = MCEvaluator(engine)
    move = Displace(delta_A=delta, every=5, attempts=4, adaptive=adaptive)
    attempts = dm.DisplaceAttempts(move, engine, evaluator, seeds=mc_stream_seeds(SEED),
                                   max_displacement_A=delta if max_displacement_A is None else max_displacement_A)
    return attempts, evaluator


def fresh_energy(make, positions):
    """A freshly built float64 force field's energy at ``positions`` (its own list, built there)."""
    from fastmace.model.forces import MACEForceField

    atoms = make.built["atoms"].copy()
    atoms.positions = positions
    field = MACEForceField(make.built["loaded"], atoms, neighbours=make.built["plan"].neighbours,
                           engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"))
    return float(field.energy_only(torch.as_tensor(positions, dtype=torch.float64))[0])


def proposal(attempts, natoms):
    """The next attempt's atom and displacement, decoded on the host from the stream offsets."""
    seeds = attempts.seeds
    base = torch.tensor(attempts.offsets["proposal"], dtype=torch.int64)
    numbers = streams.positions(base, dm.PROPOSAL_COUNT)
    atom, d = dm.decode_displace(streams.uniforms(seeds["proposal"], numbers),
                                 streams.normals(seeds["proposal"], numbers), natoms,
                                 torch.tensor([attempts.delta_A], dtype=torch.float64))
    u = streams.uniforms(seeds["acceptance"],
                         streams.positions(torch.tensor(attempts.offsets["acceptance"], dtype=torch.int64), 1))
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


def test_attach_refuses_a_variable_cell(mace_engine):
    engine = mace_engine(npt=True)
    with pytest.raises(PlanRefused) as raised:
        attempts_of(engine, 0.05)
    assert raised.value.codes == (dm.VARIABLE_CELL_CODE,)


def test_adapted_delta_rule():
    # delta <- delta * clamp(acc / 0.3, 0.5, 2), at most delta_A and the workload's maximum.
    assert dm.adapted_delta(0.1, 3, 10, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.1)
    assert dm.adapted_delta(0.1, 0, 10, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.05)
    assert dm.adapted_delta(0.1, 10, 10, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.2)
    assert dm.adapted_delta(0.4, 10, 10, delta_A=0.5, max_displacement_A=0.45) == pytest.approx(0.45)
    assert dm.adapted_delta(0.1, 0, 0, delta_A=0.5, max_displacement_A=0.5) == pytest.approx(0.1)  # nothing attempted


# --------------------------------------------------------------------------- attempts on an engine


def run_one_by_one(make, engine, attempts, evaluator, count, temperature_K, trial_rebuild_expected=None):
    """``count`` blocks of one attempt each, each checked against a direct evaluation and the acceptance rule."""
    natoms = engine.state.positions.shape[0]
    kT = K_B_EV_PER_K * temperature_K
    outcomes = []
    for _ in range(count):
        before = engine.state.positions.clone()
        reference = engine.force_field.neighbours.reference_positions.clone()
        atom, d, u = proposal(attempts, natoms)
        trial = before.clone()
        trial[atom] += torch.from_numpy(d)
        needs = bool(((trial - reference) ** 2).sum(dim=1).max() > (SKIN / 2) ** 2)
        baseline = evaluator.baseline().clone()
        rebuilds = engine.force_field.neighbours.rebuilds
        record = attempts.run(1, temperature_K=temperature_K, baseline=baseline)
        assert record["attempted"] == 1
        assert record["trial_rebuilds"] == int(needs)
        assert engine.force_field.neighbours.rebuilds == rebuilds + int(needs)  # one build, no loop
        expected = fresh_energy(make, trial.numpy())
        assert float(attempts.trial_energy[0]) == pytest.approx(expected, rel=1e-12, abs=1e-12)
        delta_e = float(attempts.trial_energy[0]) - float(baseline[0])
        accepted = u == 0 or math.log(u) < -delta_e / kT
        assert record["accepted"] == int(accepted)
        torch.testing.assert_close(engine.state.positions, trial if accepted else before, rtol=0, atol=0)
        if accepted:
            # The committed list is the one around the accepted geometry: every pair within the cutoff.
            fresh = make()
            fresh.force_field.rebuild(trial)
            fresh.force_field.compact(trial)
            assert compacted_edges(engine) == compacted_edges(fresh)
            engine.canonicalise(reevaluate=True)  # a block with an acceptance ends at a canonical point
        outcomes.append((accepted, needs))
    return outcomes


def test_small_displacements_stay_on_the_attempt_path(mace_engine):
    engine = mace_engine()
    attempts, evaluator = attempts_of(engine, 0.01)
    outcomes = run_one_by_one(mace_engine, engine, attempts, evaluator, 6, temperature_K=300.0)
    assert not any(needs for _, needs in outcomes)
    assert attempts.counters["trial_rebuilds"] == 0


def test_displacement_beyond_half_the_skin_uses_a_list_around_the_trial(mace_engine):
    engine = mace_engine()
    attempts, evaluator = attempts_of(engine, 0.2)
    # A low temperature: downhill moves are accepted, uphill ones rejected; both happen in these attempts.
    outcomes = run_one_by_one(mace_engine, engine, attempts, evaluator, 10, temperature_K=5.0)
    rebuilt = [accepted for accepted, needs in outcomes if needs]
    assert True in rebuilt and False in rebuilt, outcomes
    assert attempts.counters["trial_rebuilds"] == sum(needs for _, needs in outcomes)


def test_a_block_of_attempts_runs_through_flagged_attempts(mace_engine):
    """Several attempts in one block: the flagged ones are handled on the host and the block resumes."""
    engine = mace_engine()
    attempts, evaluator = attempts_of(engine, 0.2)
    other = mace_engine()
    single, single_evaluator = attempts_of(other, 0.2)
    record = attempts.run(8, temperature_K=50.0, baseline=evaluator.baseline().clone())
    baseline = single_evaluator.baseline().clone()
    accepted = 0
    for _ in range(8):
        step = single.run(1, temperature_K=50.0, baseline=baseline)
        accepted += step["accepted"]
        baseline = single.baseline  # the block's baseline after the attempt, as the next attempt sees it
    assert record["attempted"] == 8 and record["accepted"] == accepted
    assert record["trial_rebuilds"] == single.counters["trial_rebuilds"] > 0
    torch.testing.assert_close(engine.state.positions, other.state.positions, rtol=0, atol=0)
    assert attempts.offsets == single.offsets == {"proposal": 8 * dm.PROPOSAL_COUNT, "acceptance": 8}


def test_a_rejected_displacement_leaves_the_committed_list(mace_engine):
    for delta in (0.01, 0.2):  # without and with a trial rebuild
        engine = mace_engine()
        attempts, evaluator = attempts_of(engine, delta)
        before, positions = list_state(engine), engine.state.positions.clone()
        # A baseline far below every trial: every attempt is rejected.
        record = attempts.run(5, temperature_K=300.0, baseline=evaluator.baseline() - 1e3)
        assert record["accepted"] == 0 and record["attempted"] == 5
        assert (record["trial_rebuilds"] > 0) == (delta > SKIN / 2)
        assert_same_list(before, list_state(engine))
        torch.testing.assert_close(engine.state.positions, positions, rtol=0, atol=0)
        # The streams advanced by the fixed counts all the same.
        assert attempts.offsets == {"proposal": 5 * dm.PROPOSAL_COUNT, "acceptance": 5}


def test_stream_offsets_do_not_depend_on_decisions(mace_engine):
    engine = mace_engine()
    attempts, evaluator = attempts_of(engine, 0.2)
    attempts.run(3, temperature_K=300.0, baseline=evaluator.baseline() + 1e3)  # every attempt accepted
    assert attempts.counters["accepted"] == 3
    assert attempts.offsets == {"proposal": 3 * dm.PROPOSAL_COUNT, "acceptance": 3}


def test_an_edge_overflow_is_redone_with_the_same_numbers(mace_engine):
    """The active edges are shrunk below the count before the block (a forced overflow, as in 3-T04's tests):
    the first attempt overflows, is restored and redone with the same numbers on enough edges."""
    plain, forced = mace_engine(), mace_engine()
    outcomes = []
    for engine, shrink in ((plain, False), (forced, True)):
        attempts, evaluator = attempts_of(engine, 0.01)
        baseline = evaluator.baseline().clone()
        if shrink:
            engine.force_field.use_edges(int(engine.force_field.edge_count) - 20)
        record = attempts.run(4, temperature_K=300.0, baseline=baseline)
        outcomes.append((record, engine.state.positions.clone(), dict(attempts.offsets),
                         float(attempts.baseline[0])))
    (plain_record, plain_positions, plain_offsets, plain_e), (record, positions, offsets, energy) = outcomes
    # On the CPU the edges are exact counts, so a displacement that adds a pair overflows in the plain run too.
    assert record["overflows_redone"] > plain_record["overflows_redone"]
    for key in ("attempted", "accepted", "trial_rebuilds"):
        assert record[key] == plain_record[key], key
    torch.testing.assert_close(positions, plain_positions, rtol=0, atol=0)
    assert offsets == plain_offsets
    assert energy == pytest.approx(plain_e, rel=1e-12, abs=1e-12)
    assert forced.info()["overflows_redone"] >= 1
    assert not forced.overflowed()


def test_a_trial_build_that_grows_the_capacity_is_redone_with_the_same_numbers(mace_engine):
    """The neighbour capacity is set to the candidates of the last build (no headroom): a trial build with
    more pairs grows it, which moves the buffers; the attempt is restored and redone with the same numbers."""
    plain, forced = mace_engine(), mace_engine()
    outcomes = []
    for engine, tight in ((plain, False), (forced, True)):
        attempts, evaluator = attempts_of(engine, 0.2)
        if tight:
            nl = engine.force_field.neighbours
            nl.set_capacity(nl.num_candidates)
            engine.canonicalise()  # build into the exact-size buffers
            nl.headroom = 1.0  # no headroom on growth either, so later trials with more pairs grow again
        baseline = evaluator.baseline().clone()
        record = attempts.run(6, temperature_K=300.0, baseline=baseline + 1e3)  # all accepted
        outcomes.append((record, engine.state.positions.clone(), engine.force_field.neighbours.capacity))
    (plain_record, plain_positions, _), (record, positions, capacity) = outcomes
    assert record["overflows_redone"] >= 1
    for key in ("attempted", "accepted", "trial_rebuilds"):
        assert record[key] == plain_record[key], key
    torch.testing.assert_close(positions, plain_positions, rtol=0, atol=0)
    assert capacity >= forced.force_field.neighbours.num_candidates


def test_adaptive_delta_changes_only_at_the_end_of_a_block_and_is_recorded(mace_engine):
    engine = mace_engine()
    attempts, evaluator = attempts_of(engine, 0.2, adaptive=True)
    record = attempts.run(4, temperature_K=300.0, baseline=evaluator.baseline() - 1e3)  # none accepted
    assert record["delta_A"] == pytest.approx(0.2)  # the δ the block used
    assert record["adapted_delta_A"] == pytest.approx(0.1)  # clamp(0 / 0.3) = 0.5
    assert attempts.delta_A == pytest.approx(0.1)
    state = attempts.state_dict()
    assert state["delta_A"] == pytest.approx(0.1)
    assert state["delta_history"] == pytest.approx([0.2, 0.1])
    fixed, fixed_evaluator = attempts_of(mace_engine(), 0.2)
    record = fixed.run(4, temperature_K=300.0, baseline=fixed_evaluator.baseline() - 1e3)
    assert fixed.delta_A == pytest.approx(0.2) and "adapted_delta_A" not in record


def test_state_dict_round_trip(mace_engine):
    engine = mace_engine()
    attempts, evaluator = attempts_of(engine, 0.2, adaptive=True)
    attempts.run(3, temperature_K=300.0, baseline=evaluator.baseline().clone())
    state = attempts.state_dict()
    again, _ = attempts_of(mace_engine(), 0.2, adaptive=True)
    again.load_state_dict(state)
    assert again.state_dict() == state
    assert set(state["counters"]) == set(dm.COUNTER_KEYS)


def test_md_after_a_rejected_block_equals_md_without_it(mace_engine):
    """A rejected block leaves nothing behind: the following MD steps are bitwise those of an engine that never
    ran it, and the registry check against a fresh construction finds no difference (SPEC §8.5, §12.2)."""
    from fastmace import events

    engine, untouched = mace_engine(), mace_engine()
    attempts, evaluator = attempts_of(engine, 0.2)
    record = attempts.run(6, temperature_K=300.0, baseline=evaluator.baseline() - 1e3)
    assert record["accepted"] == 0 and record["trial_rebuilds"] > 0
    assert events.check_against(engine, untouched) == []
    for _ in range(3):
        engine.step()
        untouched.step()
    torch.testing.assert_close(engine.state.positions, untouched.state.positions, rtol=0, atol=0)
    torch.testing.assert_close(engine.state.velocities, untouched.state.velocities, rtol=0, atol=0)
