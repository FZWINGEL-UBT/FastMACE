"""ASE equivalence of the integrators on the GPU (SPEC §12.2 "ASE equivalence"; plan 1b-T09).

MH-1 (``FASTMACE_MODELS``, head ``matpes_r2scan``) on NaCl 3x3x3, plain cuEq float64. Every run is a
fresh process, so no FastMACE patch reaches the ASE references:

- **Langevin:** 0 -> 400 K in 1 ps, then 400 -> 0 K in 8 ps (18,000 steps of 0.5 fs) from the 0 K
  lattice, as the research engine's MD checks (``studies/other_models/README.md``). The engine that
  ``FixedPlanner`` builds and ASE's ``Langevin`` with MACE's calculator, same seed: the same
  temperature and potential energy in every 50 K bin of heating and cooling, |ΔPE| < 0.01 meV/atom.
- **BAOAB, full cell:** 2,000 steps at 300 K and 1 GPa from a 300 K Maxwell-Boltzmann start. The ASE
  variant (``ase_trace_drag_all_elements=True``) against ASE's ``LangevinBAOAB``, the production
  variant against ``fastmace.reference.integrators.CorrectedLangevinBAOAB``. Plain cuEq float64 is not
  bitwise between engines (cuEq's run-time kernel choice), so each pair must agree far more closely
  than the two variants differ from each other.

Results go to ``FASTMACE_RESULTS`` when set (``docs/validation/1b-T09-integrators.md``).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_ase_equivalence_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
needs_mh1 = pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")

SEED = 20261019
HEAT_STEPS, COOL_STEPS, PEAK_K = 2000, 16000, 400.0
BAOAB_STEPS, BAOAB_K, BAOAB_GPA = 2000, 300.0, 1.0
BIN_K = 50.0
PE_LIMIT = 1e-5      # eV/atom: 0.01 meV/atom (SPEC §12.2)
T_LIMIT = 0.1        # K, per bin mean
TIMEOUT = 4 * 3600

COMMON = """
import json, sys, time
import numpy as np
SPEC = {spec!r}

def structure():
    from ase.build import bulk
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    if SPEC["start_K"]:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        MaxwellBoltzmannDistribution(atoms, temperature_K=SPEC["start_K"], rng=np.random.default_rng(11))
    return atoms

def target(step):
    if SPEC["integrator"] == "baoab":
        return SPEC["start_K"]
    heat, cool, peak = SPEC["heat"], SPEC["cool"], SPEC["peak"]
    return peak * step / heat if step <= heat else peak * (1.0 - (step - heat) / cool)

def config():
    from fastmace.config import IntegratorConfig
    if SPEC["integrator"] == "langevin":
        return IntegratorConfig(ensemble="nvt", seed=SPEC["seed"])
    return IntegratorConfig(ensemble="npt", seed=SPEC["seed"], cell_mode="full", pressure_GPa=SPEC["pressure"],
                            ase_trace_drag_all_elements=SPEC["trace_all"])

def temperature(kinetic, natoms):
    from ase import units
    return 2.0 * kinetic / (3.0 * natoms * units.kB)
"""

ASE_CHILD = """
import warnings
import torch
from ase import units
from fastmace.model import labels
label = labels.LABELS["mh1-r2scan"]
from mace.calculators import MACECalculator
import os
calc = MACECalculator(model_paths=os.path.join(os.environ["FASTMACE_MODELS"], label.checkpoint.file_name),
                      head=label.head, device="cuda", default_dtype="float64", enable_cueq=True)
atoms = structure()
atoms.calc = calc
c = config()
rng = np.random.default_rng(c.seed)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if SPEC["integrator"] == "langevin":
        from ase.md.langevin import Langevin
        dyn = Langevin(atoms, timestep=c.timestep_fs * units.fs, friction=c.friction, fixcm=c.fixcm,
                       temperature_K=target(0), rng=rng)
    else:
        from ase.md.langevinbaoab import LangevinBAOAB
        from fastmace.reference.integrators import CorrectedLangevinBAOAB
        cls = LangevinBAOAB if SPEC["trace_all"] else CorrectedLangevinBAOAB
        t_tau = 1.0 / c.friction
        p_tau = 20.0 * t_tau if c.barostat_tau_fs is None else c.barostat_tau_fs * units.fs
        dyn = cls(atoms, timestep=c.timestep_fs * units.fs, T_tau=t_tau, P_tau=p_tau, hydrostatic=False,
                  externalstress=-c.pressure_GPa * units.GPa, P_mass_factor=c.barostat_mass_factor,
                  temperature_K=target(0), rng=rng)
n = len(atoms)
rows = []
start = time.perf_counter()
for step in range(1, SPEC["steps"] + 1):
    dyn.set_temperature(temperature_K=target(step))
    dyn.run(1)
    rows.append([step, target(step), temperature(atoms.get_kinetic_energy(), n),
                 atoms.get_potential_energy() / n, atoms.get_volume()])
out = {"rows": rows, "positions": atoms.positions.tolist(), "cell": atoms.cell.array.tolist(),
       "seconds": time.perf_counter() - start, "gpu": torch.cuda.get_device_name()}
print(json.dumps(out))
"""

FASTMACE_CHILD = """
import fastmace
from fastmace.config import EngineConfig
from fastmace.planner import fixed
from fastmace.model import compile as fc
planner = fixed.FixedPlanner(engine=EngineConfig(backend="cueq"))
hardware = planner.hardware()
loaded = planner.load("mh1-r2scan")
import torch
atoms = structure()
plan = planner.plan(loaded, atoms, config(), hardware)
engine = planner.build(plan, loaded, atoms, temperature_K=target(0), compile_cache=fc.CompiledModelCache())
n = len(atoms)
rows = []
start = time.perf_counter()
for step in range(1, SPEC["steps"] + 1):
    engine.integrator.set_temperature(target(step))
    engine.step()
    values = engine.sync()
    volume = float(abs(np.linalg.det(values["cell"]))) if "cell" in values else float(atoms.get_volume())
    rows.append([step, target(step), temperature(float(values["kinetic"]), n), float(values["energy"][0]) / n,
                 volume])
out = {"rows": rows, "positions": values["positions"].tolist(),
       "cell": values["cell"].tolist() if "cell" in values else atoms.cell.array.tolist(),
       "seconds": time.perf_counter() - start, "gpu": torch.cuda.get_device_name(), "mode": engine.mode,
       "records": list(engine.force_field.resolution.enabled), "plan": plan.describe()}
print(json.dumps(out, default=str))
"""


def run(side, integrator, trace_all=False):
    spec = {"integrator": integrator, "seed": SEED, "trace_all": trace_all}
    if integrator == "langevin":
        spec.update(steps=HEAT_STEPS + COOL_STEPS, heat=HEAT_STEPS, cool=COOL_STEPS, peak=PEAK_K, start_K=0.0)
    else:
        spec.update(steps=BAOAB_STEPS, start_K=BAOAB_K, pressure=BAOAB_GPA)
    child = ASE_CHILD if side == "ase" else FASTMACE_CHILD
    return ENGINE.run_fresh(COMMON.format(spec=spec) + child, timeout=TIMEOUT)


def bins(rows):
    """Mean temperature and potential energy per (segment, 50 K target bin): heating, then cooling."""
    rows = np.asarray(rows)
    step, target = rows[:, 0], rows[:, 1]
    segment = np.where(step <= HEAT_STEPS, 0, 1)
    index = np.minimum(np.floor(target / BIN_K), PEAK_K / BIN_K - 1).astype(int)
    out = {}
    for key in sorted(set(zip(segment.tolist(), index.tolist()))):
        mask = (segment == key[0]) & (index == key[1])
        out[key] = (rows[mask, 2].mean(), rows[mask, 3].mean())
    return out


def max_position_difference(a, b):
    return float(np.abs(np.subtract(a["positions"], b["positions"])).max())


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1
def test_langevin_reproduces_ase_in_every_50_K_bin():
    ase, ours = run("ase", "langevin"), run("fastmace", "langevin")
    ase_bins, our_bins = bins(ase["rows"]), bins(ours["rows"])
    assert set(ase_bins) == set(our_bins) and len(ase_bins) == 2 * PEAK_K / BIN_K
    table = {f"{'heat' if s == 0 else 'cool'} {int(b * BIN_K)}-{int((b + 1) * BIN_K)} K":
             {"dT_K": our_bins[(s, b)][0] - ase_bins[(s, b)][0],
              "dPE_meV_per_atom": 1e3 * (our_bins[(s, b)][1] - ase_bins[(s, b)][1]),
              "T_K": ase_bins[(s, b)][0]}
             for s, b in sorted(ase_bins)}
    last_ase, last_ours = ase["rows"][-1], ours["rows"][-1]
    summary = {"bins": table,
               "max_abs_dT_K": max(abs(v["dT_K"]) for v in table.values()),
               "max_abs_dPE_meV_per_atom": max(abs(v["dPE_meV_per_atom"]) for v in table.values()),
               "final_T_K": [last_ase[2], last_ours[2]],
               "final_positions_max_diff_A": max_position_difference(ours, ase),
               "seconds": {"ase": ase["seconds"], "fastmace": ours["seconds"]},
               "gpu": ours["gpu"], "mode": ours["mode"], "records": ours["records"], "plan": ours["plan"]}
    ENGINE.report("1b-T09-langevin-ase-equivalence", summary)
    assert summary["max_abs_dPE_meV_per_atom"] < 1e3 * PE_LIMIT, summary["max_abs_dPE_meV_per_atom"]
    assert summary["max_abs_dT_K"] < T_LIMIT, summary["max_abs_dT_K"]
    assert max(last_ase[2], last_ours[2]) < 50.0  # the cooling ramp really cooled it


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1
def test_baoab_variants_reproduce_their_references():
    stock, corrected = run("ase", "baoab", trace_all=True), run("ase", "baoab", trace_all=False)
    ase_variant, production = run("fastmace", "baoab", trace_all=True), run("fastmace", "baoab", trace_all=False)

    def compare(a, b):
        rows_a, rows_b = np.asarray(a["rows"]), np.asarray(b["rows"])
        return {"positions_A": max_position_difference(a, b),
                "cell_A": float(np.abs(np.subtract(a["cell"], b["cell"])).max()),
                "T_K": float(np.abs(rows_a[:, 2] - rows_b[:, 2]).max()),
                "PE_meV_per_atom": 1e3 * float(np.abs(rows_a[:, 3] - rows_b[:, 3]).max()),
                "volume_A3": float(np.abs(rows_a[:, 4] - rows_b[:, 4]).max())}

    summary = {"ase_variant_vs_ase": compare(ase_variant, stock),
               "production_vs_corrected": compare(production, corrected),
               "variants": compare(production, ase_variant),
               "references": compare(corrected, stock),
               "volume_change": stock["rows"][-1][4] / stock["rows"][0][4] - 1,
               "seconds": {"ase": stock["seconds"], "corrected": corrected["seconds"],
                           "fastmace": [ase_variant["seconds"], production["seconds"]]},
               "gpu": production["gpu"], "mode": production["mode"], "records": production["records"]}
    ENGINE.report("1b-T09-baoab-ase-equivalence", summary)
    for pair in ("ase_variant_vs_ase", "production_vs_corrected"):
        assert summary[pair]["PE_meV_per_atom"] < 1e3 * PE_LIMIT, (pair, summary[pair])
        # each engine follows its own reference far more closely than the variants follow each other
        assert summary[pair]["positions_A"] < 0.01 * summary["variants"]["positions_A"], (pair, summary)
    assert summary["references"]["positions_A"] > 0.5 * summary["variants"]["positions_A"]
