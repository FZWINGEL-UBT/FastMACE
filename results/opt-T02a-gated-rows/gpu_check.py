"""OPT-T02 stage A on the GPU: ``gated_rows`` off against on, same process, same structure (one backend family).

For each structure (3x3x3, rattled 0.05 A, seed 20261020) it builds three engines through the minimal
planner: off, on (``RewriteConfig(gated_rows="on")``) and off again as a control. It compares their first
force evaluation (the compiled force call the engine starts with): energy, forces and, with
``--stress``, stress. Plain cuEq float64
(``--backend cueq``) is expected to be bitwise equal; the hybrid float32 result is recorded the same way.
Plain cuEq plans leave the MH-1 package and the block linears off (they are ``auto`` for the hybrid only),
so there both sides request them explicitly (``CUEQ_PACKAGE``); ``reassociate_first`` stays off, as the
planner refuses it on plain cuEq.
Writes ``<output>.json``. Run in a fresh process per backend (SPEC §8.7) on an idle GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(fastmace.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402

CUEQ_PACKAGE = dict(factor_species="on", fold_cutoff="on", density_first="on", joint_radial_input="on",
                    block_linear="on", grouped_linear="on", merged_residual="on")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--backend", choices=("hybrid", "cueq"), required=True)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--stress", action="store_true", help="NPT engines (stress path)")
    p.add_argument("--reproducible", action="store_true",
                   help="Workload(reproducible=True): plain cuEq with deterministic kernels (2-T16), so that off "
                        "against off is bitwise and on against off decides")
    args = p.parse_args(argv)
    if args.backend == "hybrid":
        ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend=args.backend))
    hardware = planner.hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    precision = "float32" if args.backend == "hybrid" else "float64"
    out = dict(model=args.model, backend=args.backend, precision=precision, gpu=torch.cuda.get_device_name(),
               torch=torch.__version__, fastmace_commit=ab_step_time.fastmace_commit(), stress=args.stress,
               reproducible=args.reproducible,
               package=CUEQ_PACKAGE if args.backend == "cueq" else "planner defaults",
               structures={})
    integrator = IntegratorConfig(ensemble="npt" if args.stress else "nvt", temperature_K=300.0)
    for spec in args.structures:
        name, atoms = ab_step_time.read_structure(spec, None)
        rng = np.random.default_rng(20261020)
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        results = {}
        for setting in ("off", "on", "off2"):  # off2: an off-against-off control in the same process
            base = CUEQ_PACKAGE if args.backend == "cueq" else {}
            request = "on" if setting == "on" else "off"
            rewrites = (RewriteConfig(**base, gated_rows=request) if request == "on" or base else "auto")
            field = ForceField(loaded, atoms, precision=precision, rewrites=rewrites)
            plan = planner.plan(field, atoms, integrator, Workload(max_temperature_K=300.0,
                                                                       reproducible=args.reproducible), hardware)
            engine = planner.build(plan, field, atoms.copy(), temperature_K=0.0)
            state = engine.state
            enabled = [line for line in plan.describe().splitlines() if "gated_rows" in line]
            results[setting] = dict(energy=state.energy.detach().double().cpu().numpy().copy(),
                                    forces=state.forces.detach().double().cpu().numpy().copy(),
                                    stress=(state.stress.detach().double().cpu().numpy().copy()
                                            if state.stress is not None else None),
                                    plan_line=enabled)
            del engine, field
            torch.cuda.empty_cache()
        off = results["off"]
        entry = dict(natoms=len(atoms), plan_lines={k: v["plan_line"] for k, v in results.items()})
        for other, label in (("on", "on_vs_off"), ("off2", "off_vs_off")):
            entry[label] = {}
            for key in ("energy", "forces", "stress"):
                if off[key] is None:
                    continue
                b = results[other][key]
                entry[label][key] = dict(bitwise_equal=bool(np.array_equal(b, off[key])),
                                         max_abs_diff=float(np.abs(b - off[key]).max()),
                                         max_abs=float(np.abs(off[key]).max()))
        out["structures"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "plan_lines"}), flush=True)
    args.output.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
