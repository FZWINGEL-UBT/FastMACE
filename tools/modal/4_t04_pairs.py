"""4-T04: NVE, CSVR and Langevin pipelined against checked without FixAtoms (the conductor's question, 23 October 2026).

Is the pair bitwise equal on a tree without 4-T04? Runs the trajectory protocol of tests/gpu/test_fix_atoms_gpu.py
(NaCl 3x3x3, MH-1 matpes_r2scan, plain cuEq float64, reproducible, 600 steps of 2 fs at 1000 K) without any
constraint, pipelined and checked, each in a fresh process (``test_engine_gpu.run_fresh`` of the tree it runs in),
and writes one JSON per run into ``--output`` as it goes, then ``pairs.json`` with each pair's verdict: bitwise or
not, the largest position and momentum differences, the energies and the rebuild counts.

Usage: python tools/modal/4_t04_pairs.py --output DIR [--schemes nve,csvr,langevin]
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DRIVER = """
import json
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig
SPEC = {spec!r}
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
if SPEC["scheme"] == "langevin":
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
elif SPEC["scheme"] == "nve":
    integrator = fm.integrators.NVE(timestep_fs=2.0)
else:
    integrator = fm.integrators.CSVR(timestep_fs=2.0, temperature_K=1000.0, tau_fs=50.0)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261023, workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(pipelined=SPEC["pipelined"]))
sim.run(steps=SPEC["steps"])
committed = sim.committed()
info = dict(sim._engine.info())
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy,
        "fixcm": sim.plan.integrator.fixcm, "fused": info.get("fused_integrator"),
        "rebuilds": info["neighbour_rebuilds"], "graphs": info["graphs"], "backend": sim.plan.engine.backend,
        "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier, "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--schemes", default="nve,csvr,langevin")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / "tests" / "gpu"), str(ROOT / "tests")]
    spec = importlib.util.spec_from_file_location("engine_helpers", ROOT / "tests" / "gpu" / "test_engine_gpu.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    pairs = {}
    for scheme in args.schemes.split(","):
        runs = {}
        for mode in ("on", "off"):
            start = time.perf_counter()
            run = engine.run_fresh(DRIVER.format(spec=dict(scheme=scheme, pipelined=mode, steps=args.steps)))
            run["seconds"] = time.perf_counter() - start
            runs[mode] = run
            (output / f"{scheme}-{mode}.json").write_text(json.dumps(run, indent=1))
        positions = [np.frombuffer(bytes.fromhex(runs[mode]["positions"])) for mode in ("on", "off")]
        momenta = [np.frombuffer(bytes.fromhex(runs[mode]["momenta"])) for mode in ("on", "off")]
        pairs[scheme] = {"bitwise": (runs["on"]["positions"], runs["on"]["momenta"])
                         == (runs["off"]["positions"], runs["off"]["momenta"]),
                         "max_position_diff_A": float(np.abs(positions[0] - positions[1]).max()),
                         "max_momentum_diff": float(np.abs(momenta[0] - momenta[1]).max()),
                         "energy_eV": [runs[mode]["energy"] for mode in ("on", "off")],
                         "rebuilds": [runs[mode]["rebuilds"] for mode in ("on", "off")],
                         "fixcm": runs["on"]["fixcm"], "fused": [runs[mode]["fused"] for mode in ("on", "off")],
                         "seconds": [runs[mode]["seconds"] for mode in ("on", "off")], "gpu": runs["on"]["gpu"]}
        (output / "pairs.json").write_text(json.dumps(pairs, indent=1))
        print(scheme, json.dumps(pairs[scheme]), flush=True)


if __name__ == "__main__":
    main()
