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
``acceptance``: the kinetic temperature with MC on and off for each momentum choice (Li and Cs on the cation
sites of the same cell, mass ratio 19, without fixcm; Na and K with fixcm), and the negative
control, which must fail the same test. The transmutation χ² runs on the GPU
(``tests/gpu/test_mc_statistics_gpu.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from ase.data import chemical_symbols
from ase.units import kB as K_B_EV_PER_K  # the integrators' and (after 3-T09's fix) the driver's k_B

from _support import mc_reference as ref
from _support.mc_fixtures import cpu_planner, mc_simulation, tiny_loaded
from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import mc_stream_seeds
from fastmace.errors import CHARGE_PARITY_CODE, ChargeParityRefused
from fastmace.mc import MCDriver, Swap, Transmute, momenta, moves, streams
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


def simulation(loaded, atoms, temperature_K, *, fixcm=True, friction_per_fs=0.01, seed=1, superset=SUPERSET):
    return mc_simulation(loaded, atoms, superset, integrator=langevin(temperature_K, fixcm=fixcm,
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
@pytest.mark.acceptance  # about 5.5 min on 8 CPU workers: run at phase acceptance, not in every merge (owner, 23 October 2026)
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

# Two cation pairs on NaCl 2×1×1 (16 atoms), the second species on every second cation site:
# - Li and Cs (masses 6.94 and 132.9, ratio 19), without fixcm: velocities left unchanged at a swap split the
#   species temperatures by many standard errors in 6 ps (Na and K, ratio 1.7, gave 3.3 σ, 22 October 2026).
#   The tiny model is random, so the chemistry is irrelevant;
# - Na and K (ratio 1.7) with fixcm: the integrator's fixcm (ASE's Langevin: the centre-of-mass part of the
#   noise removed and the rest scaled by sqrt(N / (N - 1))) is exact only for equal masses. With Li and Cs it
#   holds Li near 860 K and Cs near 570 K at a 600 K target without any MC (22 October 2026), so MC on and off
#   cannot agree there; with Na and K its bias is below the statistical error.
LI, CS = 3, 55
PAIRS = {"NaK": (NA, K), "LiCs": (LI, CS)}
SUPERSETS = {"NaK": SUPERSET, "LiCs": (LI, CL, CS)}
HYBRID_TEMPERATURE_K = 600.0
HYBRID_FRICTION_PER_FS = 0.05  # energy relaxation 1/(2 γ) = 10 fs, against 10 fs between samples
HYBRID_STEPS_PER_BLOCK = 10
HYBRID_ATTEMPTS = 10
HYBRID_EQUILIBRATION_BLOCKS = 100
HYBRID_BLOCKS = 600

HYBRID_CASES = [  # (move, momenta, fixcm, cation pair)
    ("swap", "exchange", True, "NaK"),
    ("swap", "redraw_all", True, "NaK"),
    ("transmute", "redraw_all", True, "NaK"),
    ("swap", "exchange", False, "LiCs"),
    ("swap", "rescale", False, "LiCs"),
    ("swap", "redraw", False, "LiCs"),
    ("transmute", "rescale", False, "LiCs"),
]


@pytest.fixture(scope="module")
def light_heavy(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    return tiny_loaded(tmp_path_factory.mktemp("light-heavy"), SUPERSETS["LiCs"])


def pair_mu(pair):
    """Δμ of the pair's second species (relative to the first): the least-squares slope of the enumerated
    energies of the 8 cation sites, as for the transmutation enumeration."""
    from mace.calculators import MACECalculator

    calculator = MACECalculator(models=[tiny_mace_model(list(SUPERSETS[pair]), **FAMILIES["mh1"])], device="cpu",
                                default_dtype="float64")
    first, second = PAIRS[pair]
    atoms = transmute_atoms()
    sites = np.flatnonzero(atoms.numbers == NA)
    numbers = atoms.numbers.copy()
    numbers[sites] = first
    atoms.set_atomic_numbers(numbers)
    return ref.transmute_case(atoms, sites, (first, second), calculator).delta_mu


def cations(pair, rattle=0.05):
    """NaCl 2×1×1 (16 atoms) with the cations alternately the pair's two species, rattled."""
    first, second = PAIRS[pair]
    atoms = transmute_atoms()
    atoms.rattle(rattle, seed=3)
    numbers = atoms.numbers.copy()
    sites = np.flatnonzero(numbers == NA)
    numbers[sites] = first
    numbers[sites[::2]] = second
    atoms.set_atomic_numbers(numbers)
    return atoms


def hybrid_run(loaded, pair, move, fixcm, *, seed=7):
    """Kinetic temperatures (total and per cation species) every block of a Langevin run with MC blocks
    (``move``) or none."""
    sim = simulation(loaded, cations(pair), HYBRID_TEMPERATURE_K, fixcm=fixcm,
                     friction_per_fs=HYBRID_FRICTION_PER_FS, seed=seed, superset=SUPERSETS[pair])
    engine = sim._engine
    driver = None
    if move is not None:
        driver = MCDriver([move], seed=seed)
        driver.attach(engine)
    natoms = engine.state.natoms
    # The degrees of freedom the thermostat fills, as the library reports them (decision 37): 3N - 3 under the
    # mass-weighted fixcm projection (the NVT fixcm default since 3-T22c), 3N under ASE's projection or without
    # fixcm. The 3N - dof removed degrees are the centre of mass's; atom i loses m_i / M of them on average, so
    # species s keeps 3 n_s - (3N - dof) M_s / M.
    dof = sim.degrees_of_freedom
    removed = 3 * natoms - dof
    names = {number: chemical_symbols[number] for number in PAIRS[pair]}
    split = tuple(names.values())
    samples = {"total": [], **{name: [] for name in names.values()}, "split": []}
    for block in range(1, HYBRID_EQUILIBRATION_BLOCKS + HYBRID_BLOCKS + 1):
        sim.run(steps=HYBRID_STEPS_PER_BLOCK)
        if block > HYBRID_EQUILIBRATION_BLOCKS:
            state = engine.state
            twice_kinetic = (state.masses * state.velocities ** 2).sum(dim=1)
            samples["total"].append(float(twice_kinetic.sum()) / (dof * K_B_EV_PER_K))
            per_species = {}
            for number, name in names.items():
                mask = state.species == number
                if bool(mask.any()):  # a transmutation may empty a species for a while
                    species_dof = 3 * int(mask.sum()) - removed * float(state.masses[mask].sum() / state.masses.sum())
                    per_species[name] = float(twice_kinetic[mask].sum()) / (species_dof * K_B_EV_PER_K)
                    samples[name].append(per_species[name])
            if len(per_species) == 2:
                # The split T(second) - T(first): zero in equilibrium whatever the masses, without the run's
                # common-mode temperature fluctuation.
                samples["split"].append(per_species[split[1]] - per_species[split[0]])
        if driver is not None:
            driver.run_block(engine, block)
    # A species present in fewer than half the samples has no per-species temperature to compare.
    return {name: ref.batch_mean(values) for name, values in samples.items()
            if len(values) >= HYBRID_BLOCKS // 2}, (None if driver is None else driver.counters)


_MD_ONLY = {}


def _md_only(loaded, pair, fixcm):
    """The run without MC, once per process, cation pair and fixcm setting."""
    if (pair, fixcm) not in _MD_ONLY:
        _MD_ONLY[pair, fixcm] = hybrid_run(loaded, pair, None, fixcm)[0]
    return _MD_ONLY[pair, fixcm]


def hybrid_move(kind, velocities, pair, delta_mu):
    first, second = (chemical_symbols[number] for number in PAIRS[pair])
    if kind == "swap":
        return Swap([(first, second)], every=1, attempts=HYBRID_ATTEMPTS, velocities=velocities)
    return Transmute([first, second], every=1, attempts=HYBRID_ATTEMPTS, velocities=velocities,
                     mu={second: delta_mu})


def hybrid_agrees(on, off):
    """MC on and off agree in the total and per-species kinetic temperatures and their split; the total is the
    target and the split is zero (equipartition: every species at the same temperature)."""
    return (all(ref.agree(on[name], off[name]) for name in on if name in off)
            and ref.agree(on["total"], (HYBRID_TEMPERATURE_K, 0.0))
            and ("split" not in on or ref.agree(on["split"], (0.0, 0.0))))


@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.parametrize(("kind", "velocities", "fixcm", "pair"), HYBRID_CASES)
def test_kinetic_temperature_with_mc_on_and_off(loaded, light_heavy, kind, velocities, fixcm, pair):
    model = loaded if pair == "NaK" else light_heavy
    on, counters = hybrid_run(model, pair, hybrid_move(kind, velocities, pair, pair_mu(pair)), fixcm)
    off = _md_only(model, pair, fixcm)
    print(dict(kind=kind, velocities=velocities, fixcm=fixcm, pair=pair, on=on, off=off, counters=counters,
               deviations=ref.deviations(on, off)))
    assert counters[kind]["accepted"] > 0.1 * counters[kind]["attempted"]
    assert hybrid_agrees(on, off), (on, off)


@pytest.mark.slow
@pytest.mark.acceptance
def test_kinetic_temperature_negative_control_fails(light_heavy, monkeypatch):
    """Velocities unchanged at a Li <-> Cs swap: the same test must fail."""
    monkeypatch.setattr(momenta, "exchange", lambda velocities, masses_old, masses_new, normals, kT: velocities)
    on, counters = hybrid_run(light_heavy, "LiCs", hybrid_move("swap", "exchange", "LiCs", 0.0), False)
    off = _md_only(light_heavy, "LiCs", False)
    print(dict(on=on, off=off, counters=counters, deviations=ref.deviations(on, off)))
    assert counters["swap"]["accepted"] > 0.1 * counters["swap"]["attempted"]
    # Frequent swaps decouple the speeds from the masses: the heavy atoms carry light-atom speeds.
    assert not hybrid_agrees(on, off), (on, off)
