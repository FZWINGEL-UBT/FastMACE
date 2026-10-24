"""Phase 5's runtime adaptation on the GPU (plan task 5-T11, runbook item 6; SPEC §9.6, decisions 52 and 53).

- ``storm --side switched|unswitched``: 5-T07's forced re-capture storm (MH-1 ``matpes_r2scan``, NaCl 3x3x3,
  600 K, Langevin 2 fs, ``EngineConfig(graph_variant_cap=1)``, a restart every 50 steps;
  ``tests/gpu/test_adapt_switch_gpu.py``'s set-up) in this process, one side per process: ``switched`` with the
  default ``AdaptationConfig`` (decision 52's trigger switches graph -> compiled), ``unswitched`` with
  ``AdaptationConfig(adapt="off")`` (the monitor records, nothing switches). The clock covers ``run()`` only, after
  the build; ``storm-<side>.json`` records the step time, the tier switches and the monitor's segments. One block per
  side, cross-process: the switched side must be faster beyond the 3 % floor (runbook, timing design).
- ``nacl-config DIR``: the NVT anneal on NaCl 3x3x3 the runbook defines from ``main.py``'s ``--profile anneal``
  (relaxation, 0 -> 1000 K in 20 ps, then 3 K/ps to 250 K, 1 fs, samples and frames every 10 fs), as a protocol file
  and a CIF in DIR for ``fastmace run`` (``results/3-T14-protocol/run.sh``). No shipped profile is added.
- ``check RUN_DIR...``: each run's ``metadata.json`` (the newest piece): ``counters.tier_switches`` must be empty,
  and the re-capture time fraction, rebuild counts and overflows of ``counters.segments`` are recorded (decision 52:
  recorded, not acted on). Exit 1 if any run switched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NACL_PROTOCOL = """\
# 5-T11 (runbook item 6): the NVT anneal on NaCl 3x3x3 from main.py's --profile anneal (heat_ps 20, peak 1000 K,
# then 3 K/ps to final 250 K, samples and frames every 10 fs, relax fmax 0.01). Not a shipped profile.
structure = "{structure}"
model = "mh1"
head = "matpes_r2scan"
repeat = [3, 3, 3]
timestep_fs = 1.0
ensemble = "NVT"

[[schedule]]
kind = "ramp"
start_K = 0.0
end_K = 1000.0
ps = 20.0

[[schedule]]
kind = "ramp"
start_K = 1000.0
end_K = 250.0
rate_K_per_ps = -3.0

[engine]
backend = "hybrid"

[output]
trajectory_every = 10
thermo_every = 10
restart_every = 10000

[relax]
fmax = 0.01
optimizer = "gpu-LBFGS"
max_steps = 5000
"""


def storm(side: str, steps: int, output: Path) -> dict:
    import numpy as np
    import torch
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    import fastmace  # noqa: F401  (before torch: soft file locks, the OEq runtime)
    from fastmace import integrators
    from fastmace.config import AdaptationConfig, EngineConfig
    from fastmace.io.output import Output
    from fastmace.model import compile as compiling
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import adapt
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    compiling.configure_allocator()
    loaded = loading.load_pristine("mh1-r2scan")
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
    adaptation = AdaptationConfig() if side == "switched" else AdaptationConfig(adapt="off")
    directory = output / side
    sim = Simulation(atoms, ForceField(loaded, atoms, precision="float32"),
                     integrator=integrators.Langevin(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.02),
                     seed=7, workload=Workload(max_temperature_K=600.0), adaptation=adaptation,
                     engine=EngineConfig(graph_variant_cap=1),
                     output=Output(str(directory), thermo_every=5, trajectory_every=None, restart_every=50))
    first_plan = sim.plan.describe()
    torch.cuda.synchronize()
    start = time.perf_counter()
    sim.run(steps=steps)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    switches = [switch.record() for switch in adapt.tier_switches(sim.plan)]
    final_plan = sim.plan.describe()
    sim.close()
    metadata = json.loads((directory / "metadata.json").read_text())
    return {"side": side, "steps": steps, "wall_s": wall, "ms_per_step": 1e3 * wall / steps,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "source_commit": os.environ.get("FASTMACE_SOURCE_COMMIT"), "switches": switches, "tier": sim.plan.tier,
            "segments": metadata["counters"].get("segments", []), "plan_first": first_plan, "plan_final": final_plan}


def nacl_config(directory: Path) -> Path:
    from ase.build import bulk
    from ase.io import write

    directory.mkdir(parents=True, exist_ok=True)
    cif = directory / "NaCl_conventional.cif"
    write(cif, bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    protocol = directory / "nvt_anneal_NaCl.toml"
    protocol.write_text(NACL_PROTOCOL.format(structure=cif.resolve()))
    return protocol


def _newest_metadata(run: Path) -> Path:
    found = sorted(run.rglob("metadata.json"), key=lambda path: path.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"{run}: no metadata.json")
    return found[-1]


def check(runs: list[Path]) -> dict:
    out = {}
    for run in runs:
        metadata = json.loads(_newest_metadata(run).read_text())
        counters = metadata["counters"]
        segments = counters.get("segments") or []
        wall = sum(segment.get("wall_s") or 0.0 for segment in segments)
        recapture = sum(segment.get("recapture_s") or 0.0 for segment in segments)
        out[str(run)] = {
            "tier_switches": counters.get("tier_switches", []),
            "segments": len(segments),
            "recapture_fraction": recapture / wall if wall else None,
            "max_segment_recapture_fraction": max(((segment.get("recapture_s") or 0.0) / segment["wall_s"]
                                                   for segment in segments if segment.get("wall_s")), default=None),
            "recaptures": counters.get("recaptures"), "overflows_redone": counters.get("overflows_redone"),
            "neighbour_lists": counters.get("neighbour_lists"),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    storming = commands.add_parser("storm", help="the forced re-capture storm, one side")
    storming.add_argument("--side", choices=("switched", "unswitched"), required=True)
    storming.add_argument("--steps", type=int, default=2000)
    storming.add_argument("--output", type=Path, required=True)
    configuring = commands.add_parser("nacl-config", help="write the NaCl NVT anneal protocol and its CIF")
    configuring.add_argument("directory", type=Path)
    checking = commands.add_parser("check", help="no tier switch in these runs; record the monitor's segments")
    checking.add_argument("runs", type=Path, nargs="+")
    checking.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "storm":
        args.output.mkdir(parents=True, exist_ok=True)
        result = storm(args.side, args.steps, args.output)
        (args.output / f"storm-{args.side}.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
        print(f"{args.side}: {result['ms_per_step']:.2f} ms/step, tier {result['tier']}, "
              f"{len(result['switches'])} switch(es)")
        return 0
    if args.command == "nacl-config":
        print(nacl_config(args.directory))
        return 0
    result = check(args.runs)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    for run, entry in result.items():
        print(f"{run}: switches {len(entry['tier_switches'])}, re-capture fraction {entry['recapture_fraction']}")
    return 1 if any(entry["tier_switches"] for entry in result.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
