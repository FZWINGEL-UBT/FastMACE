"""Plan task 4-T04 on the GPU: ``FixAtoms`` in the graph engine and the fused integrator stage.

1. **Fused stage (slow tier):** OPT3-T03's stage-level byte comparison (``test_fused_integrator_gpu.py``) with
   ``FixAtoms`` on 4 atoms of NaCl 3x3x3 and ``fixcm=False``: ``_stage_acp`` and ``_stage_ac`` through PyTorch
   and fused from the same buffers write the same bytes, the moved-reference trials included.
2. **Trajectories (slow tier):** ``reproducible=True`` (plain cuEq float64), NaCl 3x3x3 with ``FixAtoms``,
   600 steps of 2 fs at 1000 K, each in a fresh process: Langevin with the fused stage on and off,
   pipelined and checked, and NVE and CSVR pipelined and checked. Positions and momenta bitwise equal across
   each pair, the fixed atoms bitwise in place with zero momenta, and the neighbour list rebuilt.
3. **ASE equivalence (acceptance tier, conductor's choice 2):** as ``test_ase_equivalence_gpu.py``'s Langevin
   (MH-1 ``matpes_r2scan``, NaCl 3x3x3, plain cuEq float64, 0 -> 400 -> 0 K over 4,000 steps of 0.5 fs, ASE's
   noise stream), with ``FixAtoms`` on 10 atoms and ``fixcm=False`` on both sides: |ΔT| < 1e-6 K and
   |ΔPE| < 1e-7 meV/atom in every 50 K bin, temperatures over 3(N - n_fixed) degrees of freedom (decision 45).

Results go to ``FASTMACE_RESULTS`` when set (``docs/validation/4-T04.md``).
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


ENGINE = _module("_fix_atoms_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
FUSED = _module("_fix_atoms_gpu_fused_helpers", Path(__file__).parent / "test_fused_integrator_gpu.py")
EQUIVALENCE = _module("_fix_atoms_gpu_equivalence_helpers", Path(__file__).parent / "test_ase_equivalence_gpu.py")

needs_mh1 = pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")

STAGE_FIXED = [0, 7, 30, 101]
EQUIVALENCE_FIXED = list(range(0, 216, 22))  # 10 atoms of both species

# OPT3-T03's stage driver with the constraint set before the Maxwell-Boltzmann draw (the fixed atoms start at
# rest, as ASE's set_momenta leaves them).
_ANCHOR = "atoms = nacl(repeat=3)\n"
assert FUSED.STAGE_DRIVER.count(_ANCHOR) == 1
STAGE_DRIVER = FUSED.STAGE_DRIVER.replace(_ANCHOR, _ANCHOR + "from ase.constraints import FixAtoms\n"
                                          "atoms.set_constraint(FixAtoms(indices=SPEC['fixed']))\n")

MD_DRIVER = """
import json, tempfile
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig
SPEC = {spec!r}
atoms = nacl(repeat=3)
atoms.set_constraint(FixAtoms(indices=SPEC["fixed"]))
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
start = atoms.positions.copy()
if SPEC["scheme"] == "langevin":
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
elif SPEC["scheme"] == "nve":
    integrator = fm.integrators.NVE(timestep_fs=2.0)
else:
    integrator = fm.integrators.CSVR(timestep_fs=2.0, temperature_K=1000.0, tau_fs=50.0)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261023, workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(pipelined=SPEC["pipelined"], **SPEC["engine"]))
sim.run(steps=SPEC["steps"])
committed = sim.committed()
info = dict(sim._engine.info())
fixed = SPEC["fixed"]
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy,
        "fixed_in_place": bool(np.array_equal(committed.positions[fixed], start[fixed])),
        "fixed_at_rest": bool(not committed.momenta[fixed].any()),
        "free_moved_A": float(np.abs(np.delete(committed.positions - start, fixed, axis=0)).max()),
        "fixcm": sim.plan.integrator.fixcm, "degrees_of_freedom": sim.degrees_of_freedom,
        "fused": info.get("fused_integrator"), "rebuilds": info["neighbour_rebuilds"], "graphs": info["graphs"],
        "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier,
        "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""


@pytest.mark.gpu
@pytest.mark.slow
@needs_mh1
def test_the_fused_stage_writes_the_pytorch_stage_bytes_with_fixed_atoms():
    run = ENGINE.run_fresh(STAGE_DRIVER.format(spec={"fixcm": False, "fixed": STAGE_FIXED}),
                           python=ENGINE.HYBRID_PYTHON)
    ENGINE.report("4-T04-fused-stage-fix-atoms", run)
    trials = run["trials"]
    assert len(trials) == 10, run
    for trial in trials:
        assert all(count == 0 for count in trial["mismatched_bytes"].values()), trial
    none, early, hard = trials[4], trials[6], trials[8]  # pipelined trials 2, 3, 4 (atom 0 is fixed)
    assert none["pipe_flags"][0] == 0 and none["pipe_flags"][2] == 0, none
    assert early["pipe_flags"][0] == 0 and early["pipe_flags"][2] == 1, early
    assert hard["pipe_flags"][0] == 1 and hard["flags"][0] == 1, hard


GROUPS = {
    "langevin": {f"{pipelined}-{fused}": dict(scheme="langevin", pipelined=pipelined,
                                              engine=FUSED._engine(fused))
                 for pipelined in ("on", "off") for fused in (False, True)},
    "nve-csvr": {f"{scheme}-{pipelined}": dict(scheme=scheme, pipelined=pipelined, engine={})
                 for scheme in ("nve", "csvr") for pipelined in ("on", "off")},
}


@pytest.mark.gpu
@pytest.mark.slow
@needs_mh1
@pytest.mark.parametrize("group", list(GROUPS))
def test_trajectories_with_fixed_atoms_are_bitwise_equal(group):
    runs = {name: ENGINE.run_fresh(MD_DRIVER.format(spec={**case, "steps": 600, "fixed": STAGE_FIXED}))
            for name, case in GROUPS[group].items()}
    summary = {name: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for name, run in runs.items()}
    ENGINE.report(f"4-T04-fix-atoms-trajectories-{group}", summary)
    for name, run in runs.items():
        assert run["step"] == 600 and run["rebuilds"] > 0 and run["tier"] == "graph", (name, summary)
        assert run["fixed_in_place"] and run["fixed_at_rest"] and run["free_moved_A"] > 0.1, (name, summary)
        assert run["fixcm"] is False and run["degrees_of_freedom"] == 3 * (216 - len(STAGE_FIXED)), (name, summary)
    if group == "langevin":
        for name, run in runs.items():
            off = runs[name.rsplit("-", 1)[0] + "-False"]
            assert run["fused"] == (True if name.endswith("True") else None), (name, summary)
            assert (run["positions"], run["momenta"]) == (off["positions"], off["momenta"]), (name, summary)
    else:
        for scheme in ("nve", "csvr"):
            on, off = runs[f"{scheme}-on"], runs[f"{scheme}-off"]
            assert (on["positions"], on["momenta"]) == (off["positions"], off["momenta"]), (scheme, summary)


# -------------------------------------------------------------------------- ASE equivalence (choice 2)

_FIX = """
from ase.constraints import FixAtoms
_structure = structure

def structure():
    atoms = _structure()
    atoms.set_constraint(FixAtoms(indices=SPEC["fixed"]))
    return atoms

def config():
    from fastmace.config import IntegratorConfig
    return IntegratorConfig(ensemble="nvt", seed=SPEC["seed"], fixcm=False, fixcm_projection="ase")

def temperature(kinetic, natoms):
    from ase import units
    return 2.0 * kinetic / (3.0 * (natoms - len(SPEC["fixed"])) * units.kB)
"""

_FIXED_CHECK = """
fixed = SPEC["fixed"]
out["fixed_in_place"] = bool(np.array_equal(np.asarray(out["positions"])[fixed], structure().positions[fixed]))
print(json.dumps(out, default=str))
"""


def run(side):
    spec = {"integrator": "langevin", "seed": EQUIVALENCE.SEED, "trace_all": False,
            "steps": EQUIVALENCE.HEAT_STEPS + EQUIVALENCE.COOL_STEPS, "heat": EQUIVALENCE.HEAT_STEPS,
            "cool": EQUIVALENCE.COOL_STEPS, "peak": EQUIVALENCE.PEAK_K, "start_K": 0.0, "fixed": EQUIVALENCE_FIXED}
    child = EQUIVALENCE.ASE_CHILD if side == "ase" else EQUIVALENCE.FASTMACE_CHILD
    return ENGINE.run_fresh(EQUIVALENCE.COMMON.format(spec=spec) + _FIX + child + _FIXED_CHECK,
                            timeout=EQUIVALENCE.TIMEOUT)


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1
def test_langevin_with_fixed_atoms_reproduces_ase_in_every_50_K_bin():
    ase, ours = run("ase"), run("fastmace")
    ase_bins, our_bins = EQUIVALENCE.bins(ase["rows"]), EQUIVALENCE.bins(ours["rows"])
    assert set(ase_bins) == set(our_bins)
    table = {f"{'heat' if s == 0 else 'cool'} {int(b * EQUIVALENCE.BIN_K)}-{int((b + 1) * EQUIVALENCE.BIN_K)} K":
             {"dT_K": our_bins[(s, b)][0] - ase_bins[(s, b)][0],
              "dPE_meV_per_atom": 1e3 * (our_bins[(s, b)][1] - ase_bins[(s, b)][1]),
              "T_K": ase_bins[(s, b)][0]}
             for s, b in sorted(ase_bins)}
    summary = {"bins": table, "fixed": EQUIVALENCE_FIXED,
               "max_abs_dT_K": max(abs(v["dT_K"]) for v in table.values()),
               "max_abs_dPE_meV_per_atom": max(abs(v["dPE_meV_per_atom"]) for v in table.values()),
               "final_T_K": [ase["rows"][-1][2], ours["rows"][-1][2]],
               "final_positions_max_diff_A": EQUIVALENCE.max_position_difference(ours, ase),
               "fixed_in_place": {"ase": ase["fixed_in_place"], "fastmace": ours["fixed_in_place"]},
               "seconds": {"ase": ase["seconds"], "fastmace": ours["seconds"]},
               "gpu": ours["gpu"], "mode": ours["mode"], "records": ours["records"], "plan": ours["plan"]}
    ENGINE.report("4-T04-langevin-fix-atoms-ase-equivalence", summary)
    assert ours["fixed_in_place"] and ase["fixed_in_place"], summary
    assert summary["max_abs_dPE_meV_per_atom"] < 1e3 * EQUIVALENCE.PE_LIMIT, summary["max_abs_dPE_meV_per_atom"]
    assert summary["max_abs_dT_K"] < EQUIVALENCE.T_LIMIT, summary["max_abs_dT_K"]
    assert np.max([row[2] for row in ase["rows"]]) > EQUIVALENCE.PEAK_K / 2  # it really heated
