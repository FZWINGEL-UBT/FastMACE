"""MC exact enumeration and hybrid MD/MC statistics on the CPU (plan task 3-T12; SPEC §12.2 "MC exact
enumeration", §10.4).

A tiny MH-1-like model (superset Na, Cl, K) in float64, through a ``Simulation``'s engine and the MC driver
(``MCDriver.run_block`` called repeatedly at fixed positions, no MD between the blocks). The reference is
MACE's own float64 calculator on the same weights, over every configuration (``_support.mc_reference``):

- **swaps:** NaCl's conventional cell, 8 sites, 4 Na + 4 Cl, 70 configurations;
- **transmutations:** NaCl 2×1×1, the 8 cation sites Na or K, Cl fixed, 256 configurations;
- kT by the reference's rule (the energy span over 3), Δμ_K by its least-squares slope.

Default tier:
- **exact replay:** every decision the driver takes is predicted from its streams (the documented layout,
  ``fastmace.mc.streams``) and the reference energies; the species after every block, the acceptances and
  the block's baseline and final energies match. Near-ties (|ln u + ΔX/kT| < 1e-6) are skipped and counted;
- **the first attempt after MD:** its baseline is a fresh evaluation with the MC evaluator, its ΔE equals
  the reference ΔE at the new geometry, and its decision follows min(1, exp(−β ΔE_ref));
- **momenta:** zero total momentum under ``fixcm`` across MD and blocks (``exchange``, ``redraw_all``);
  the negative control (velocities unchanged at a mass swap) changes the kinetic energy, ``exchange``
  keeps it;
- **charge/spin:** an odd ΔZ − Δq set is refused on a model with charge and spin inputs.

``slow``: the swap populations against Boltzmann, χ² pooled per energy level (fixed seed).
``acceptance``: the kinetic temperature with MC on and off for each momentum choice, and the negative
control, which must fail the same test. The transmutation χ² runs on the GPU
(``tests/gpu/test_mc_statistics_gpu.py``).
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

from _support import mc_reference as ref
from _support.mc_fixtures import cpu_planner, mc_simulation, tiny_loaded
from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import mc_stream_seeds
from fastmace.errors import CHARGE_PARITY_CODE, ChargeParityRefused
from fastmace.mc import MCDriver, Swap, Transmute, momenta, moves, streams
from fastmace.mc.driver import K_B_EV_PER_K
from fastmace.mc.evaluator import MCEvaluator
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA, CL, K, MG = 11, 17, 19, 12
SUPERSET = (NA, CL, K)
# float64 energies of the engine and of MACE's calculator agree to about 1e-10 eV.
ENERGY_TOLERANCE_EV = 1e-8
NEAR_TIE = 1e-6  # in units of kT
SEED_REPLAY, SEED_CHI2 = 20261026, 20261022


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    return tiny_loaded(tmp_path_factory.mktemp("models"), SUPERSET)


@pytest.fixture(autouse=True)
def planner(monkeypatch):
    return cpu_planner(monkeypatch)


@pytest.fixture(scope="module")
def calculator():
    """MACE's float64 calculator on the weights ``tiny_loaded`` saves (the builder seeds torch)."""
    from mace.calculators import MACECalculator

    return MACECalculator(models=[tiny_mace_model(list(SUPERSET), **FAMILIES["mh1"])], device="cpu",
                          default_dtype="float64")


def swap_atoms():
    return nacl(repeat=1)


def transmute_atoms():
    return nacl(repeat=1).repeat((2, 1, 1))


@pytest.fixture(scope="module")
def swap_case(calculator):
    return ref.swap_case(swap_atoms(), (NA, CL), calculator)


@pytest.fixture(scope="module")
def transmute_case(calculator):
    atoms = transmute_atoms()
    return ref.transmute_case(atoms, np.flatnonzero(atoms.numbers == NA), (NA, K), calculator)


def workload(temperature_K):
    return Workload(max_temperature_K=max(5000.0, temperature_K), species_changes=True,
                    min_mc_temperature_K=min(1.0, temperature_K), mc_moves=("swap", "transmute"))


def langevin(temperature_K, *, fixcm=True, friction_per_fs=0.01):
    return integrators.Langevin(timestep_fs=1.0, temperature_K=temperature_K, friction_per_fs=friction_per_fs,
                                fixcm=fixcm)


def simulation(loaded, atoms, temperature_K, *, fixcm=True, friction_per_fs=0.01, seed=1):
    return mc_simulation(loaded, atoms, SUPERSET, integrator=langevin(temperature_K, fixcm=fixcm,
                                                                         friction_per_fs=friction_per_fs),
                         workload=workload(temperature_K), seed=seed)


def force(monkeypatch, accept: bool):
    """Fault injection (tests only): every Metropolis decision accepts, or rejects."""
    monkeypatch.setattr(moves, "metropolis", lambda u, excess, kT: torch.full_like(u, accept, dtype=torch.bool))


def mixed_cations(rattle=0.05):
    """NaCl 2×1×1 (16 atoms) with every second Na replaced by K, rattled: masses 23 and 39 on one sublattice."""
    atoms = transmute_atoms()
    atoms.rattle(rattle, seed=3)
    numbers = atoms.numbers.copy()
    sodium = np.flatnonzero(numbers == NA)
    numbers[sodium[::2]] = K
    atoms.set_atomic_numbers(numbers)
    return atoms


def committed_atoms(sim, template):
    """``template`` with the committed species, positions and cell of ``sim``."""
    committed = sim.committed()
    atoms = template.copy()
    atoms.set_atomic_numbers(np.asarray(committed.species))
    atoms.set_cell(np.asarray(committed.cell))
    atoms.set_positions(np.asarray(committed.positions))
    return atoms


# --------------------------------------------------------------------------- the exact replay


def replay(loaded, case, atoms, move, *, blocks, attempts, seed=SEED_REPLAY):
    """Blocks of the driver at fixed positions against the predictor; returns the per-block outcomes."""
    temperature = case.plan.kT / K_B_EV_PER_K
    sim = simulation(loaded, atoms, temperature)
    engine = sim._engine
    driver = MCDriver([move], seed=seed)
    driver.attach(engine)
    kT = K_B_EV_PER_K * temperature  # as the driver fills it
    predictor = ref.StreamPredictor(case, driver, engine.state.natoms, kT, NEAR_TIE)
    outcomes = []
    for step in range(1, blocks + 1):
        species = torch.as_tensor(np.asarray(sim.committed().species), dtype=torch.int64)
        start = predictor.index(species)
        predicted, accepted, ties = predictor.block(species, dict(driver.offsets), attempts)
        record = driver.run_block(engine, step)
        actual = torch.as_tensor(np.asarray(sim.committed().species), dtype=torch.int64)
        outcomes.append(dict(record=record, start=start, ties=ties, predicted_accepted=accepted,
                             same=bool(torch.equal(actual, predicted)), end=predictor.index(actual)))
    return outcomes


def assert_replay(case, outcomes, attempts):
    compared = [entry for entry in outcomes if entry["ties"] == 0]
    assert len(outcomes) - len(compared) <= 1, "more than one block with a near tie"
    for entry in compared:
        record = entry["record"]
        assert entry["same"], entry
        assert record["accepted"] == entry["predicted_accepted"], entry
        assert record["moves"][0]["attempted"] == attempts and record["moves"][0]["impossible"] == 0
    for entry in outcomes:
        record = entry["record"]
        # The block's baseline is the MC evaluator's energy of the committed configuration; the energy after the
        # block is that of the configuration it ends in.
        assert abs(record["baseline_eV"] - case.energies[entry["start"]]) <= ENERGY_TOLERANCE_EV, entry
        assert abs(record["energy_eV"] - case.energies[entry["end"]]) <= ENERGY_TOLERANCE_EV, entry
    total = sum(entry["record"]["accepted"] for entry in outcomes)
    assert 0.05 < total / (attempts * len(outcomes)) < 0.95, "the replay must see acceptances and rejections"


def test_swap_decisions_replay_the_exact_chain(loaded, swap_case):
    move = Swap([("Na", "Cl")], every=1, attempts=10)
    outcomes = replay(loaded, swap_case, swap_atoms(), move, blocks=30, attempts=10)
    assert_replay(swap_case, outcomes, 10)


def test_transmute_decisions_replay_the_exact_chain(loaded, transmute_case):
    # redraw_all under fixcm draws its velocity numbers per block; proposals and acceptances are as for rescale.
    move = Transmute(["Na", "K"], every=1, attempts=10, velocities="redraw_all", mu={"K": transmute_case.delta_mu})
    outcomes = replay(loaded, transmute_case, transmute_atoms(), move, blocks=20, attempts=10)
    assert_replay(transmute_case, outcomes, 10)


def test_single_attempt_blocks_replay_each_decision(loaded, swap_case):
    """attempts=1: every block is one decision, compared one by one."""
    outcomes = replay(loaded, swap_case, swap_atoms(), Swap([("Na", "Cl")], every=1, attempts=1), blocks=40,
                      attempts=1)
    assert_replay(swap_case, outcomes, 1)


# --------------------------------------------------------------------------- the first attempt after MD


def first_attempts_after_md(loaded, calculator, monkeypatch, *, forced):
    """Blocks of one Na <-> K swap, each after 3 MD steps at 600 K; per block the fresh MC-evaluator energy
    before it, the reference energies of the current and proposed configurations, and the block record."""
    if forced:
        force(monkeypatch, True)
    template = mixed_cations()
    sim = simulation(loaded, template, 600.0)
    engine = sim._engine
    move = Swap([("Na", "K")], every=1, attempts=1)
    driver = MCDriver([move], seed=SEED_REPLAY)
    driver.attach(engine)
    seeds = mc_stream_seeds(driver.seed)
    natoms = engine.state.natoms
    pairs = torch.tensor([[NA, K]])
    out = []
    for step in range(1, 7):
        sim.run(steps=3)
        fresh = float(MCEvaluator(engine).evaluate()[0])
        atoms = committed_atoms(sim, template)
        offsets = dict(driver.offsets)
        u = streams.uniforms(seeds["proposal"], torch.arange(offsets["proposal"], offsets["proposal"] + 2 * natoms + 1))
        u_accept = float(streams.uniforms(seeds["acceptance"], torch.tensor([offsets["acceptance"]]))[0])
        species = torch.as_tensor(atoms.numbers, dtype=torch.int64)
        pair, possible = streams.decode_swap(u, species, pairs)
        assert bool(possible)
        trial = atoms.copy()
        numbers = trial.numbers.copy()
        numbers[pair.numpy()] = numbers[pair.numpy()[::-1]]
        trial.set_atomic_numbers(numbers)
        current, trial_energy = ref.energy(atoms, calculator), ref.energy(trial, calculator)
        record = driver.run_block(engine, step)
        out.append(dict(fresh=fresh, current_ref=current, trial_ref=trial_energy, u=u_accept, record=record,
                        kT=K_B_EV_PER_K * 600.0))
    return out


def test_first_attempt_after_md_has_a_fresh_baseline_and_the_reference_delta_e(loaded, calculator, monkeypatch):
    for entry in first_attempts_after_md(loaded, calculator, monkeypatch, forced=True):
        record = entry["record"]
        # The baseline is a fresh evaluation with the MC evaluator (not the MD energy), and the reference's.
        assert record["baseline_eV"] == entry["fresh"]
        assert abs(record["baseline_eV"] - entry["current_ref"]) <= ENERGY_TOLERANCE_EV
        # Forced acceptance: the energy after the block is the trial's, so ΔE is visible.
        delta = record["energy_eV"] - record["baseline_eV"]
        assert abs(delta - (entry["trial_ref"] - entry["current_ref"])) <= ENERGY_TOLERANCE_EV


def test_first_attempt_after_md_follows_the_reference_acceptance(loaded, calculator, monkeypatch):
    outcomes = first_attempts_after_md(loaded, calculator, monkeypatch, forced=False)
    decided = 0
    for entry in outcomes:
        exponent = (entry["trial_ref"] - entry["current_ref"]) / entry["kT"]
        probability = min(1.0, math.exp(-exponent))
        log_u = math.log(entry["u"]) if entry["u"] > 0 else -math.inf
        if abs(log_u + exponent) < NEAR_TIE:
            continue
        decided += 1
        # accepted with probability min(1, exp(-β ΔE_ref)): exactly when u < that probability.
        assert entry["record"]["accepted"] == int(entry["u"] < probability), entry
        assert entry["record"]["baseline_eV"] == entry["fresh"]
    assert decided >= 5


# --------------------------------------------------------------------------- momenta


def total_momentum(engine):
    state = engine.state
    return (state.masses * state.velocities).sum(dim=0)


def kinetic_energy(engine):
    state = engine.state
    return float(0.5 * (state.masses * state.velocities ** 2).sum())


def test_fixcm_keeps_zero_total_momentum_across_md_and_blocks(loaded, transmute_case):
    sim = simulation(loaded, mixed_cations(), 600.0, fixcm=True)
    engine = sim._engine
    moves_ = [Swap([("Na", "K")], every=1, attempts=5, velocities="exchange"),
              Transmute(["Na", "K"], every=1, attempts=3, velocities="redraw_all", mu={"K": transmute_case.delta_mu})]
    driver = MCDriver(moves_, seed=SEED_REPLAY)
    driver.attach(engine)
    state = engine.state
    scale = math.sqrt(state.natoms * float(state.masses.mean()) * K_B_EV_PER_K * 600.0)
    largest = float(total_momentum(engine).abs().max()) / scale
    for step in range(1, 11):
        sim.run(steps=5)
        largest = max(largest, float(total_momentum(engine).abs().max()) / scale)
        driver.run_block(engine, step)
        largest = max(largest, float(total_momentum(engine).abs().max()) / scale)
    assert driver.counters["swap"]["accepted"] > 0 and driver.counters["transmute"]["accepted"] > 0
    assert largest <= 1e-12


@pytest.mark.parametrize("treatment", ["exchange", "unchanged"])
def test_negative_control_changes_the_kinetic_energy(loaded, monkeypatch, treatment):
    """A Na <-> K swap that keeps every velocity (the negative control) changes E_kin (SPEC §10.4: masses
    (1, 4), velocities (2, 1): 4 before, 8.5 after); exchange keeps E_kin and the total momentum."""
    force(monkeypatch, True)
    if treatment == "unchanged":
        monkeypatch.setattr(momenta, "exchange", lambda velocities, masses_old, masses_new, normals, kT: velocities)
    sim = simulation(loaded, mixed_cations(), 600.0)
    engine = sim._engine
    driver = MCDriver([Swap([("Na", "K")], every=1, attempts=1)], seed=SEED_REPLAY)
    driver.attach(engine)
    sim.run(steps=3)
    engine.canonicalise()
    before, momentum = kinetic_energy(engine), total_momentum(engine).clone()
    species = engine.state.species.clone()
    assert driver.run_block(engine, 1)["accepted"] == 1
    assert int((engine.state.species != species).sum()) == 2
    after = kinetic_energy(engine)
    if treatment == "exchange":
        assert after == pytest.approx(before, rel=1e-13)
        torch.testing.assert_close(total_momentum(engine), momentum, rtol=0, atol=1e-12)
    else:
        assert abs(after - before) / before > 1e-3


# --------------------------------------------------------------------------- charge and spin


def test_parity_refusal_with_charge_and_spin_inputs(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    superset = (NA, MG, CL)
    loaded = tiny_loaded(tmp_path_factory.mktemp("omol"), superset, family="omol", charge=0, spin_multiplicity=1)
    sim = mc_simulation(loaded, nacl(repeat=1), superset, integrator=langevin(600.0), workload=workload(600.0))
    engine = sim._engine
    # Na <-> Mg changes Z by 1 at Δq = 0: an odd electron count change.
    with pytest.raises(ChargeParityRefused) as refused:
        MCDriver([Transmute(["Na", "Mg"], every=1, attempts=1)], seed=1).attach(engine)
    assert CHARGE_PARITY_CODE in refused.value.codes
    # With formal charges (Na +1, Mg +2), ΔZ − Δq = 0 is even: admitted.
    driver = MCDriver([Transmute(["Na", "Mg"], every=1, attempts=1, formal_charges={"Na": 1, "Mg": 2})], seed=1)
    driver.attach(engine)
    assert driver.decisions[0][:2] == ("parity", "checked")


# --------------------------------------------------------------------------- populations (slow)


@pytest.mark.slow
def test_swap_populations_follow_boltzmann(loaded, swap_case):
    """χ² of the configurations sampled every k attempts (λ2^k ≤ 0.01), pooled per energy level, at a fixed
    seed: they fit the Boltzmann populations and reject uniform populations and 1.5 β."""
    plan = ref.pooled_plan(swap_case.plan)
    temperature = plan.kT / K_B_EV_PER_K
    sim = simulation(loaded, swap_atoms(), temperature)
    engine = sim._engine
    driver = MCDriver([Swap([("Na", "Cl")], every=1, attempts=plan.thinning)], seed=SEED_CHI2)
    driver.attach(engine)
    predictor = ref.StreamPredictor(swap_case, driver, engine.state.natoms, K_B_EV_PER_K * temperature, NEAR_TIE)
    burn_in = 5
    samples = []
    for step in range(1, burn_in + plan.n_samples + 1):
        driver.run_block(engine, step)
        if step > burn_in:
            samples.append(predictor.index(torch.as_tensor(np.asarray(sim.committed().species))))
    result = ref.population_test(ref.counts_of(samples, len(plan.pi)), plan.x, plan.kT, plan.groups)
    print(dict(thinning=plan.thinning, samples=plan.n_samples, levels=int(plan.groups.max()) + 1, result=result))
    assert result.passes, result


# --------------------------------------------------------------------------- hybrid MD/MC (acceptance)

HYBRID_TEMPERATURE_K = 600.0
HYBRID_FRICTION_PER_FS = 0.05  # energy relaxation 1/(2 γ) = 10 fs, against 10 fs between samples
HYBRID_STEPS_PER_BLOCK = 10
HYBRID_ATTEMPTS = 10
HYBRID_EQUILIBRATION_BLOCKS = 100
HYBRID_BLOCKS = 600

HYBRID_CASES = [
    ("swap", "exchange", True),
    ("swap", "rescale", False),
    ("swap", "redraw", False),
    ("swap", "redraw_all", True),
    ("transmute", "rescale", False),
    ("transmute", "redraw_all", True),
]


def hybrid_run(loaded, move, fixcm, *, unchanged=False, seed=7):
    """Kinetic temperatures (total, Na, K) every block of a Langevin run with MC blocks (``move``) or none."""
    sim = simulation(loaded, mixed_cations(), HYBRID_TEMPERATURE_K, fixcm=fixcm,
                     friction_per_fs=HYBRID_FRICTION_PER_FS, seed=seed)
    engine = sim._engine
    driver = None
    if move is not None:
        driver = MCDriver([move], seed=seed)
        driver.attach(engine)
    natoms = engine.state.natoms
    dof = 3 * natoms - (3 if fixcm else 0)
    samples = {"total": [], "Na": [], "K": []}
    for block in range(1, HYBRID_EQUILIBRATION_BLOCKS + HYBRID_BLOCKS + 1):
        sim.run(steps=HYBRID_STEPS_PER_BLOCK)
        if block > HYBRID_EQUILIBRATION_BLOCKS:
            state = engine.state
            twice_kinetic = (state.masses * state.velocities ** 2).sum(dim=1)
            samples["total"].append(float(twice_kinetic.sum()) / (dof * K_B_EV_PER_K))
            for name, number in (("Na", NA), ("K", K)):
                mask = state.species == number
                samples[name].append(float(twice_kinetic[mask].sum()) / (3 * int(mask.sum()) * K_B_EV_PER_K))
        if driver is not None:
            driver.run_block(engine, block)
    return {name: ref.batch_mean(values) for name, values in samples.items()}, (
        None if driver is None else driver.counters)


@functools.cache
def _md_only(loaded, fixcm):
    return hybrid_run(loaded, None, fixcm)[0]


def hybrid_move(kind, velocities, delta_mu):
    if kind == "swap":
        return Swap([("Na", "K")], every=1, attempts=HYBRID_ATTEMPTS, velocities=velocities)
    return Transmute(["Na", "K"], every=1, attempts=HYBRID_ATTEMPTS, velocities=velocities, mu={"K": delta_mu})


def hybrid_agrees(on, off):
    """MC on and off agree in total and per-species kinetic temperature, and the total is the target."""
    return (all(ref.agree(on[name], off[name]) for name in on)
            and ref.agree(on["total"], (HYBRID_TEMPERATURE_K, 0.0)))


@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.parametrize(("kind", "velocities", "fixcm"), HYBRID_CASES)
def test_kinetic_temperature_with_mc_on_and_off(loaded, transmute_case, kind, velocities, fixcm):
    on, counters = hybrid_run(loaded, hybrid_move(kind, velocities, transmute_case.delta_mu), fixcm)
    off = _md_only(loaded, fixcm)
    print(dict(kind=kind, velocities=velocities, fixcm=fixcm, on=on, off=off, counters=counters))
    assert counters[kind]["accepted"] > 0.1 * counters[kind]["attempted"]
    assert hybrid_agrees(on, off), (on, off)


@pytest.mark.slow
@pytest.mark.acceptance
def test_kinetic_temperature_negative_control_fails(loaded, monkeypatch):
    """Velocities unchanged at a Na <-> K swap heat the crystal: the same test must fail."""
    monkeypatch.setattr(momenta, "exchange", lambda velocities, masses_old, masses_new, normals, kT: velocities)
    on, counters = hybrid_run(loaded, hybrid_move("swap", "exchange", 0.0), True)
    off = _md_only(loaded, True)
    print(dict(on=on, off=off, counters=counters))
    assert not hybrid_agrees(on, off), (on, off)
    # Sized in advance for at least 10 standard errors of heating.
    assert on["total"][0] - off["total"][0] > 10 * math.hypot(on["total"][1], off["total"][1])
