"""The phase 4 MD classes end to end: stock ASE with MACE's calculator against ``fastmace.ase`` (4-T13, recorded).

One side per process (the effect is a multiple, decision 39), one class and structure per call:

- ``stock``: ASE 3.27's class (``ase.md``) on ``mace.calculators.MACECalculator`` with cuEquivariance in
  float32, the way a user runs MACE MD today;
- ``fastmace``: the ``fastmace.ase`` class of the same name with the planner's default plan (MH-1: hybrid
  float32), ``seed=`` instead of ``rng``.

Both get ASE's arguments from ``tools/ab_frontends.py`` (``ase_arguments``: 100 fs thermostats, 1000 fs
barostats, pressure 0) at ``--timestep-fs``, and the same Maxwell-Boltzmann momenta (``thermalize``). After
``--warmup`` steps (the shim's plan, gate and captures included), one block of ``--block`` steps is timed between
two ``torch.cuda.synchronize`` calls. Output: ``<output>/<side>-<class>-<structure>.json`` with ms/step, the
setup time and the versions. Run on an otherwise idle GPU; compare the two files of a pair.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]
SIDES = ("stock", "fastmace")
STOCK_CLASSES = {
    "Langevin": "ase.md.langevin", "VelocityVerlet": "ase.md.verlet", "Bussi": "ase.md.bussi",
    "NoseHooverChainNVT": "ase.md.nose_hoover_chain", "IsotropicMTKNPT": "ase.md.nose_hoover_chain",
    "MTKNPT": "ase.md.nose_hoover_chain", "NVTBerendsen": "ase.md.nvtberendsen",
    "NPTBerendsen": "ase.md.nptberendsen", "Inhomogeneous_NPTBerendsen": "ase.md.nptberendsen",
}
NEEDS_RNG = ("Langevin", "Bussi")


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=SIDES, required=True)
    p.add_argument("--integrator", choices=tuple(STOCK_CLASSES), required=True)
    p.add_argument("--structure", default="NaCl",
                   help="Benchmark name (under FASTMACE_STRUCTURES) or a file with PATH@NXxNYxNZ")
    p.add_argument("--model", default="mh1-r2scan", help="Label (its file under FASTMACE_MODELS)")
    p.add_argument("--output", required=True, type=Path, help="Directory (created if missing)")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    return p.parse_args(argv)


def stock_dynamics(args, atoms, label):
    """ASE's own class on MACE's calculator (cuEquivariance, float32)."""
    import numpy as np
    from ase import units
    from mace.calculators import MACECalculator

    frontends = _tool("ab_frontends")
    atoms.calc = MACECalculator(model_paths=str(Path(os.environ["FASTMACE_MODELS"]) / label.checkpoint.file_name),
                                head=label.head, device="cuda", default_dtype="float32", enable_cueq=True)
    cls = getattr(importlib.import_module(STOCK_CLASSES[args.integrator]), args.integrator)
    options = frontends.ase_arguments(args.integrator, args.temperature_K)
    if args.integrator in NEEDS_RNG:
        options["rng"] = np.random.default_rng(args.seed)
    return cls(atoms, args.timestep_fs * units.fs, **options)


def fastmace_dynamics(args, atoms, label):
    from ase import units

    import fastmace.ase

    frontends = _tool("ab_frontends")
    return getattr(fastmace.ase, args.integrator)(atoms, args.timestep_fs * units.fs,
                                                  **frontends.ase_arguments(args.integrator, args.temperature_K),
                                                  model=args.model, seed=args.seed)


def measure(dyn, synchronize, warmup, block, clock=time.perf_counter):
    """Seconds of the warm-up and ms per step of one block after it."""
    started = clock()
    dyn.run(warmup)
    synchronize()
    warmup_s = clock() - started
    started = clock()
    dyn.run(block)
    synchronize()
    return warmup_s, 1e3 * (clock() - started) / block


def main(argv=None):
    args = parse_args(argv)
    import torch

    from fastmace.model import labels

    sys.path.insert(0, str(TOOLS))
    import ab_step_time

    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    label = labels.LABELS[args.model]
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    _tool("ab_frontends").thermalize(atoms, args.temperature_K, args.seed)
    started = time.perf_counter()
    build = stock_dynamics if args.side == "stock" else fastmace_dynamics
    dyn = build(args, atoms, label)
    build_s = time.perf_counter() - started
    warmup_s, ms_per_step = measure(dyn, torch.cuda.synchronize, args.warmup, args.block)
    import ase
    import mace

    record = dict(tool="ase_integrators_speed", created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(),
                  side=args.side, integrator=args.integrator, model=args.model, structure=structure,
                  natoms=len(atoms), temperature_K=args.temperature_K, timestep_fs=args.timestep_fs,
                  warmup=args.warmup, block=args.block, build_s=build_s, warmup_s=warmup_s, ms_per_step=ms_per_step,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, ase=ase.__version__,
                  mace=getattr(mace, "__version__", None),
                  plan=getattr(dyn, "plan", None) and dyn.plan.describe())
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.side}-{args.integrator}-{Path(str(structure)).stem}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"{args.side} {args.integrator} {structure}: {ms_per_step:.3f} ms/step ({path})")


if __name__ == "__main__":
    main()
