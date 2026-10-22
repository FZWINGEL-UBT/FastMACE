"""MC exact enumeration and hybrid MD/MC statistics on the GPU with MH-1 (plan task 3-T12; SPEC §12.2 "MC
exact enumeration", §10.4).

MH-1 (``mh1-r2scan``, head ``matpes_r2scan``) in the hybrid float32 plan, superset Na, Cl, K, through a
``Simulation``'s engine and the MC driver with captured attempts. The MC energies are the force precision
(float32; 3-T11's gate is not part of these tests). The reference is MACE's own float64 calculator over every
configuration, in a child of this environment's Python; the MC runs in a child of the hybrid Python
(``FASTMACE_HYBRID_PYTHON``), one fresh process per test:

- **swaps:** NaCl's conventional cell, 8 sites, 4 Na + 4 Cl, 70 configurations;
- **transmutations:** NaCl 2×1×1, the 8 cation sites Na or K, Cl fixed, 256 configurations;
- kT by the reference's rule (the energy span over 3), Δμ_K by its least-squares slope.

Fast tier (``slow``): the exact replay of the driver's decisions (near-ties within 0.02 kT, twice the
move-energy gate, are skipped and counted), ΔE per block within 0.01 kT of the reference, and the first
attempt after MD (a fresh MC-evaluator baseline; ΔE within 0.01 kT of the float64 evaluator's; the decision
from that float64 ΔE). ``acceptance``: the χ² of the swap and transmutation populations per configuration
(fixed seeds), and the kinetic temperature with MC on and off for each momentum choice on (Na,K)Cl 2×1×1 at 2000 K with
the negative control, which must fail.
"""

import functools
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from _support import mc_reference as ref
from _support.structures import nacl


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_mc_statistics_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

NA, CL, K = 11, 17, 19
GATE_KT = 0.01  # the move-energy gate, |ΔE_prod − ΔE_ref| ≤ 0.01 kT (§12.3)
NEAR_TIE = 2 * GATE_KT
SEED_REPLAY, SEED_SWAP_CHI2, SEED_TRANSMUTE_CHI2 = 20261026, 20261022, 20261023

needs_mh1 = pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")


def transmute_atoms():
    return nacl(repeat=1).repeat((2, 1, 1))


REFERENCE_CHILD = """
import json
import numpy as np
from mace.calculators import MACECalculator
from _support import mc_reference as ref
from _support.structures import nacl

calc = MACECalculator(model_paths=SPEC["checkpoint"], device="cuda", default_dtype="float64", head="matpes_r2scan")
swap = ref.swap_case(nacl(repeat=1), (11, 17), calc)
atoms = nacl(repeat=1).repeat((2, 1, 1))
transmute = ref.transmute_case(atoms, np.flatnonzero(atoms.numbers == 11), (11, 19), calc)
print(json.dumps({"swap": swap.energies.tolist(), "transmute": transmute.energies.tolist()}))
"""


@functools.cache
def reference_energies():
    """The float64 reference energies of both enumerations (MACE's calculator, this environment's Python)."""
    return ENGINE.run_fresh(f"SPEC = {{'checkpoint': {str(ENGINE._mh1_path())!r}}}\n" + REFERENCE_CHILD)


def cases():
    energies = reference_energies()
    swap = ref.swap_case(nacl(repeat=1), (NA, CL), None, energies=energies["swap"])
    atoms = transmute_atoms()
    transmute = ref.transmute_case(atoms, np.flatnonzero(atoms.numbers == NA), (NA, K), None,
                                   energies=energies["transmute"])
    return swap, transmute


# The MC child: SPEC holds the experiment, its parameters and the reference energies.
MC_CHILD = """
import fastmace as fm
import json, math, time
import numpy as np
import torch
from ase.data import chemical_symbols
from _support import mc_reference as ref
from _support.structures import nacl
from fastmace import integrators
from fastmace.mc import MCDriver, Swap, Transmute, momenta, moves
from ase.units import kB as K_B_EV_PER_K  # the integrators' and (after 3-T09's fix) the driver's k_B
from fastmace.mc.evaluator import MCEvaluator, float64_model
from fastmace.model.forcefield import ForceField
from fastmace.planner.workload import Workload
from fastmace.simulation import Simulation

NA, CL, K, LI, CS = 11, 17, 19, 3, 55
SUPERSET = [NA, CL, K]
PAIRS = {"NaK": (NA, K), "LiCs": (LI, CS)}
SUPERSETS = {"NaK": SUPERSET, "LiCs": [LI, CL, CS]}
loaded = fm.load("mh1-r2scan")


def simulation(atoms, temperature, *, fixcm=True, friction=0.01, seed=1, superset=SUPERSET):
    field = ForceField(loaded, atoms, precision="float32", species=superset)
    workload = Workload(max_temperature_K=max(temperature, 2000.0), species_changes=True,
                        min_mc_temperature_K=min(temperature, 1.0), mc_moves=("swap", "transmute"))
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=temperature, friction_per_fs=friction, fixcm=fixcm)
    return Simulation(atoms, field, integrator=integrator, workload=workload, seed=seed)


def transmute_atoms():
    return nacl(repeat=1).repeat((2, 1, 1))


def mixed_cations(rattle=0.05, pair="NaK"):
    # NaCl 2x1x1 (16 atoms), the cations alternately the pair's two species, rattled.
    first, second = PAIRS[pair]
    atoms = transmute_atoms()
    atoms.rattle(rattle, seed=3)
    numbers = atoms.numbers.copy()
    sites = np.flatnonzero(numbers == NA)
    numbers[sites] = first
    numbers[sites[::2]] = second
    atoms.set_atomic_numbers(numbers)
    return atoms


def species_of(sim):
    return torch.as_tensor(np.asarray(sim.committed().species), dtype=torch.int64)


def enumeration(kind):
    if kind == "swap":
        return ref.swap_case(nacl(repeat=1), (NA, CL), None, energies=SPEC["energies"]["swap"]), nacl(repeat=1)
    atoms = transmute_atoms()
    case = ref.transmute_case(atoms, np.flatnonzero(atoms.numbers == NA), (NA, K), None,
                              energies=SPEC["energies"]["transmute"])
    return case, atoms


def enumeration_move(kind, case, attempts):
    if kind == "swap":
        return Swap([("Na", "Cl")], every=1, attempts=attempts)
    return Transmute(["Na", "K"], every=1, attempts=attempts, velocities="redraw_all", mu={"K": case.delta_mu})


def replay(kind, blocks, attempts):
    case, atoms = enumeration(kind)
    temperature = case.plan.kT / K_B_EV_PER_K
    sim = simulation(atoms, temperature)
    engine = sim._engine
    driver = MCDriver([enumeration_move(kind, case, attempts)], seed=SPEC["seed"])
    driver.attach(engine)
    kT = K_B_EV_PER_K * temperature
    predictor = ref.StreamPredictor(case, driver, engine.state.natoms, kT, SPEC["near_tie"])
    rows = []
    for step in range(1, blocks + 1):
        species = species_of(sim)
        start = predictor.index(species)
        predicted, accepted, ties = predictor.block(species, dict(driver.offsets), attempts)
        record = driver.run_block(engine, step)
        actual = species_of(sim)
        end = predictor.index(actual)
        delta = record["energy_eV"] - record["baseline_eV"]
        rows.append(dict(ties=ties, same=bool(torch.equal(actual, predicted)), accepted=record["accepted"],
                         predicted_accepted=accepted, attempted=record["moves"][0]["attempted"],
                         delta_error_kT=abs(delta - (case.energies[end] - case.energies[start])) / kT))
    return dict(rows=rows, temperature_K=temperature, backend=sim.plan.engine.backend,
                dtype=sim.plan.precision.dtype, mc_graphs=sum(str(key).startswith("('mc.") for key in engine._graphs))


def float64_evaluator(sim):
    errors = {}
    for backend in (sim.plan.engine.backend, "cueq"):
        try:
            evaluator = MCEvaluator(sim._engine, "float64", model=float64_model(sim.force_field, sim.plan,
                                                                                backend=backend))
            evaluator.evaluate()
            return evaluator, backend
        except Exception as error:
            errors[backend] = repr(error)
    raise RuntimeError(f"no float64 evaluator: {errors}")


def after_md(forced, blocks=6, temperature=600.0):
    if forced:
        moves.metropolis = lambda u, excess, kT: torch.ones_like(u, dtype=torch.bool)
    sim = simulation(mixed_cations(), temperature)
    engine = sim._engine
    driver = MCDriver([Swap([("Na", "K")], every=1, attempts=1)], seed=SPEC["seed"])
    driver.attach(engine)
    reference, backend = float64_evaluator(sim)
    kT = K_B_EV_PER_K * temperature
    pairs = torch.tensor([[NA, K]])
    seeds = fm.config.mc_stream_seeds(driver.seed)
    from fastmace.mc import streams
    rows = []
    for step in range(1, blocks + 1):
        sim.run(steps=3)
        fresh = float(MCEvaluator(engine).evaluate()[0])
        before64 = float(reference.evaluate()[0])
        offsets = dict(driver.offsets)
        natoms = engine.state.natoms
        u = streams.uniforms(seeds["proposal"], torch.arange(offsets["proposal"], offsets["proposal"] + 2 * natoms + 1))
        u_accept = float(streams.uniforms(seeds["acceptance"], torch.tensor([offsets["acceptance"]]))[0])
        species = species_of(sim)
        pair, possible = streams.decode_swap(u, species, pairs)
        trial = species.clone()
        trial[pair] = species[pair].flip(0)
        trial64 = None
        if not forced:
            # The trial's float64 energy: apply it as a host event, evaluate, and put the species back.
            sim.update(species=trial.numpy())
            trial64 = float(reference.evaluate()[0])
            sim.update(species=species.numpy())
            fresh = float(MCEvaluator(engine).evaluate()[0])
        record = driver.run_block(engine, step)
        after64 = float(reference.evaluate()[0])
        rows.append(dict(fresh=fresh, baseline=record["baseline_eV"], energy=record["energy_eV"],
                         before64=before64, after64=after64, trial64=trial64, u=u_accept,
                         accepted=record["accepted"], possible=bool(possible), kT=kT))
    return dict(rows=rows, reference_backend=backend, backend=sim.plan.engine.backend, dtype=sim.plan.precision.dtype)


def chi_squared(kind, pooled, seed, burn_in=5):
    case, atoms = enumeration(kind)
    plan = ref.pooled_plan(case.plan) if pooled else case.plan
    temperature = plan.kT / K_B_EV_PER_K
    sim = simulation(atoms, temperature)
    engine = sim._engine
    driver = MCDriver([enumeration_move(kind, case, plan.thinning)], seed=seed)
    driver.attach(engine)
    predictor = ref.StreamPredictor(case, driver, engine.state.natoms, K_B_EV_PER_K * temperature, SPEC["near_tie"])
    samples = []
    started = time.perf_counter()
    for step in range(1, burn_in + plan.n_samples + 1):
        driver.run_block(engine, step)
        if step > burn_in:
            samples.append(predictor.index(species_of(sim)))
    return dict(samples=samples, seconds=time.perf_counter() - started, thinning=plan.thinning,
                n_samples=plan.n_samples, temperature_K=temperature, counters=driver.counters)


def hybrid(kind, velocities, fixcm, unchanged, pair, spec):
    if unchanged:
        momenta.exchange = lambda velocities, masses_old, masses_new, normals, kT: velocities
    sim = simulation(mixed_cations(pair=pair), spec["temperature"], fixcm=fixcm, friction=spec["friction"],
                     seed=spec["seed"], superset=SUPERSETS[pair])
    engine = sim._engine
    first, second = PAIRS[pair]
    symbols = {first: chemical_symbols[first], second: chemical_symbols[second]}
    driver = None
    if kind is not None:
        if kind == "swap":
            move = Swap([(symbols[first], symbols[second])], every=1, attempts=spec["attempts"], velocities=velocities)
        else:
            move = Transmute([symbols[first], symbols[second]], every=1, attempts=spec["attempts"],
                             velocities=velocities, mu={symbols[second]: spec["delta_mu"]})
        driver = MCDriver([move], seed=spec["seed"])
        driver.attach(engine)
    natoms = engine.state.natoms
    # ASE's convention: with fixcm the Langevin noise is scaled by sqrt(N / (N - 1)), so the thermostat holds
    # sum m v^2 at 3 N kT.
    dof = 3 * natoms
    samples = {"total": [], **{name: [] for name in symbols.values()}, "split": []}
    largest_momentum = 0.0
    scale = math.sqrt(natoms * float(engine.state.masses.mean()) * K_B_EV_PER_K * spec["temperature"])
    started = time.perf_counter()
    for block in range(1, spec["equilibration"] + spec["blocks"] + 1):
        sim.run(steps=spec["steps_per_block"])
        state = engine.state
        if block > spec["equilibration"]:
            twice = (state.masses * state.velocities ** 2).sum(dim=1).double()
            samples["total"].append(float(twice.sum()) / (dof * K_B_EV_PER_K))
            per_species = {}
            for number, name in symbols.items():
                mask = state.species == number
                if bool(mask.any()):
                    per_species[name] = float(twice[mask].sum()) / (3 * int(mask.sum()) * K_B_EV_PER_K)
                    samples[name].append(per_species[name])
            if len(per_species) == 2:  # T(second) - T(first): zero in equilibrium, no common mode
                samples["split"].append(per_species[symbols[second]] - per_species[symbols[first]])
        if driver is not None:
            driver.run_block(engine, block)
        if fixcm:
            momentum = (engine.state.masses * engine.state.velocities).sum(dim=0)
            largest_momentum = max(largest_momentum, float(momentum.abs().max()) / scale)
    return dict(stats={name: ref.batch_mean(values) for name, values in samples.items()
                       if len(values) >= spec["blocks"] // 2},
                counters=None if driver is None else driver.counters, momentum=largest_momentum,
                seconds=time.perf_counter() - started, backend=sim.plan.engine.backend, dtype=sim.plan.precision.dtype)


experiment = SPEC["experiment"]
if experiment == "replay":
    out = {kind: replay(kind, *SPEC["sizes"][kind]) for kind in SPEC["sizes"]}
elif experiment == "after_md":
    out = {"forced": after_md(True)}
elif experiment == "after_md_decisions":
    out = {"decisions": after_md(False)}
elif experiment == "chi2":
    out = chi_squared(SPEC["kind"], SPEC["pooled"], SPEC["seed"])
else:
    out = {name: hybrid(*run[:5], {**SPEC["hybrid"], **(run[5] if len(run) > 5 else {})})
           for name, run in SPEC["runs"].items()}
out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fastmace=fm.__version__)
print(json.dumps(out, default=str))
"""


def run_mc(spec):
    if not ENGINE._has_oeq():
        pytest.skip("OpenEquivariance is not installed (hybrid environment needed)")
    spec = {"energies": reference_energies(), "near_tie": NEAR_TIE, "seed": SEED_REPLAY, **spec}
    return ENGINE.run_fresh(f"SPEC = {spec!r}\n" + MC_CHILD, python=ENGINE.HYBRID_PYTHON)


# --------------------------------------------------------------------------- fast tier (slow on the GPU)


@pytest.mark.slow
@needs_mh1
def test_mh1_decisions_replay_the_exact_chain():
    result = run_mc({"experiment": "replay", "sizes": {"swap": (30, 10), "transmute": (20, 10)}})
    summary = {}
    for kind in ("swap", "transmute"):
        rows = result[kind]["rows"]
        compared = [row for row in rows if row["ties"] == 0]
        summary[kind] = dict(blocks=len(rows), compared=len(compared), temperature_K=result[kind]["temperature_K"],
                             accepted=sum(row["accepted"] for row in rows),
                             attempted=sum(row["attempted"] for row in rows),
                             mismatches=[i for i, row in enumerate(rows) if row["ties"] == 0
                                         and not (row["same"] and row["accepted"] == row["predicted_accepted"])],
                             max_delta_error_kT=max(row["delta_error_kT"] for row in rows))
    ENGINE.report("3-T12-mh1-replay", {**summary, "gpu": result["gpu"], "fastmace": result["fastmace"],
                                       "backend": result["swap"]["backend"], "dtype": result["swap"]["dtype"]})
    assert (result["swap"]["backend"], result["swap"]["dtype"]) == ("hybrid", "float32")
    for kind, entry in summary.items():
        assert entry["mismatches"] == [], (kind, entry)
        # A near tie within 0.02 kT includes every ΔE = 0 move between symmetry-equivalent configurations with
        # u > 0.98, so on the GPU blocks with a tie are common; each block restarts from the driver's species.
        assert entry["compared"] >= entry["blocks"] // 2, (kind, entry)
        assert entry["max_delta_error_kT"] <= GATE_KT, (kind, entry)
        assert 0.05 < entry["accepted"] / entry["attempted"] < 0.95, (kind, entry)
        assert result[kind]["mc_graphs"] >= 1  # the attempts ran as captured graphs


@pytest.mark.slow
@needs_mh1
def test_mh1_first_attempt_after_md():
    forced = run_mc({"experiment": "after_md"})["forced"]
    decisions = run_mc({"experiment": "after_md_decisions"})["decisions"]
    rows = forced["rows"]
    errors = [abs((row["energy"] - row["baseline"]) - (row["after64"] - row["before64"])) / row["kT"] for row in rows]
    decided, agreements = 0, []
    for row in decisions["rows"]:
        exponent = (row["trial64"] - row["before64"]) / row["kT"]
        log_u = math.log(row["u"]) if row["u"] > 0 else -math.inf
        if abs(log_u + exponent) < NEAR_TIE:
            continue
        decided += 1
        agreements.append(row["accepted"] == int(log_u < -exponent))
    ENGINE.report("3-T12-mh1-after-md", {"delta_error_kT": errors, "decided": decided, "agreements": agreements,
                                         "reference_backend": forced["reference_backend"], "backend": forced["backend"],
                                         "baselines_fresh": [row["baseline"] == row["fresh"] for row in rows]})
    # The baseline is a fresh evaluation with the MC evaluator, not the MD energy.
    assert all(row["baseline"] == row["fresh"] for row in rows + decisions["rows"])
    assert max(errors) <= GATE_KT
    assert decided >= 4 and all(agreements)


# --------------------------------------------------------------------------- acceptance


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1
@pytest.mark.parametrize(("kind", "seed"), [("swap", SEED_SWAP_CHI2), ("transmute", SEED_TRANSMUTE_CHI2)])
def test_mh1_populations_follow_boltzmann(kind, seed):
    swap, transmute = cases()
    plan = (swap if kind == "swap" else transmute).plan
    result = run_mc({"experiment": "chi2", "kind": kind, "pooled": False, "seed": seed})
    test = ref.population_test(ref.counts_of(result["samples"], len(plan.pi)), plan.x, plan.kT)
    ENGINE.report(f"3-T12-mh1-chi2-{kind}", {"result": test, "thinning": plan.thinning, "n_samples": plan.n_samples,
                                             "attempts": plan.attempts, "seconds": result["seconds"],
                                             "temperature_K": result["temperature_K"], "counters": result["counters"],
                                             "gpu": result["gpu"], "fastmace": result["fastmace"]})
    assert test.passes, test


# (Na,K)Cl 2x1x1, 16 atoms, at 2000 K: Na <-> K swaps accept about 20 %, 10 fs between samples, energy
# relaxation 1/(2 γ) = 25 fs, 2000 samples.
HYBRID = dict(temperature=2000.0, friction=0.02, attempts=10, steps_per_block=10, equilibration=200, blocks=2000,
              seed=7)
NEGATIVE = dict(attempts=100)
HYBRID_RUNS = {  # (move, momenta, fixcm, velocities unchanged, cation pair)
    "off_fixcm": (None, None, True, False, "NaK"),
    "off_free": (None, None, False, False, "NaK"),
    "swap_exchange": ("swap", "exchange", True, False, "NaK"),
    "swap_rescale": ("swap", "rescale", False, False, "NaK"),
    "swap_redraw": ("swap", "redraw", False, False, "NaK"),
    "swap_redraw_all": ("swap", "redraw_all", True, False, "NaK"),
    "transmute_rescale": ("transmute", "rescale", False, False, "NaK"),
    "transmute_redraw_all": ("transmute", "redraw_all", True, False, "NaK"),
    # The negative control, without fixcm, and its own run without MC. With these settings unchanged velocities
    # split the Na and K temperatures by only 2.4 σ (22 October 2026), and MH-1 accepts almost no Li <-> Cs swap
    # (6 of 22 000), so the control keeps Na and K and lets the swaps outpace the thermostat: 100 attempts per
    # 10 steps. The speeds then decouple from the masses (T_K / T_Na towards 39/23). A weaker friction
    # (0.005/fs) split them more but lengthened the correlation time, and gave 3.9 σ.
    "off_negative": (None, None, False, False, "NaK", NEGATIVE),
    "negative_unchanged": ("swap", "exchange", False, True, "NaK", NEGATIVE),
}


# Open for acceptance item 5 (conductor, 22 October 2026): with fixcm, redraw_all draws the correct
# zero-momentum Maxwell-Boltzmann state, sum m v^2 = (3N - 3) kT, while the fixcm Langevin's sqrt(N / (N - 1))
# noise factor (ASE's convention) holds 3N kT, so every accepted block lowers the kinetic energy by (N - 1)/N
# and the runs sit below MC off (16 atoms: 3.5 and 5.5 sigma). A non-strict expected failure until 3-T22's
# corrected projection; then the check must pass and this goes.
FIXCM_REDRAW_ALL = ("swap_redraw_all", "transmute_redraw_all")
FIXCM_REDRAW_ALL_REASON = ("3-T22: redraw_all draws (3N-3) kT in the zero-momentum subspace, the fixcm Langevin's "
                           "sqrt(N/(N-1)) holds 3N kT")


@functools.cache
def hybrid_results(names):
    _, transmute = cases()
    runs = {name: HYBRID_RUNS[name] for name in names}
    return run_mc({"experiment": "hybrid", "runs": runs, "hybrid": {**HYBRID, "delta_mu": transmute.delta_mu}})


def hybrid_agrees(on, off, target):
    """As on the CPU: MC on and off agree (total, per species, split), the total is the target and the split zero."""
    return (all(ref.agree(tuple(on[name]), tuple(off[name])) for name in on if name in off)
            and ref.agree(tuple(on["total"]), (target, 0.0))
            and ("split" not in on or ref.agree(tuple(on["split"]), (0.0, 0.0))))


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1
# The negative control runs without fixcm: the integrator's fixcm is exact only for equal masses (see the CPU
# file), so the runs with fixcm keep to the Na <-> K moves whose correct treatments it must pass.
@pytest.mark.parametrize("names", [("off_fixcm", "swap_exchange", "swap_redraw_all", "transmute_redraw_all"),
                                   ("off_free", "swap_rescale", "swap_redraw", "transmute_rescale"),
                                   ("off_negative", "negative_unchanged")],
                         ids=["fixcm", "free", "negative"])
def test_mh1_kinetic_temperature_with_mc_on_and_off(names):
    result = hybrid_results(names)
    off_name = names[0]
    off = result[off_name]["stats"]
    summary = {name: {key: result[name][key] for key in ("stats", "counters", "momentum", "seconds")}
               for name in names}
    for name in names[1:]:
        summary[name]["deviations_sigma"] = ref.deviations(result[name]["stats"], off)
    ENGINE.report(f"3-T12-mh1-hybrid-{off_name}", {**summary, "hybrid": HYBRID, "gpu": result["gpu"],
                                                   "fastmace": result["fastmace"]})
    known = []
    for name in names[1:]:
        on = result[name]["stats"]
        kind = HYBRID_RUNS[name][0]
        counters = result[name]["counters"][kind]
        assert counters["accepted"] > 0.02 * counters["attempted"], (name, counters)
        if name == "negative_unchanged":
            assert not hybrid_agrees(on, off, HYBRID["temperature"]), (name, on, off)
        elif name in FIXCM_REDRAW_ALL and not hybrid_agrees(on, off, HYBRID["temperature"]):
            known.append(name)
        else:
            assert hybrid_agrees(on, off, HYBRID["temperature"]), (name, on, off)
        if HYBRID_RUNS[name][2]:
            assert result[name]["momentum"] <= 1e-5, (name, result[name]["momentum"])
    assert ref.agree(tuple(off["total"]), (HYBRID["temperature"], 0.0)), off
    if known:  # every other check of the group has passed
        pytest.xfail(f"{known}: {FIXCM_REDRAW_ALL_REASON}")

