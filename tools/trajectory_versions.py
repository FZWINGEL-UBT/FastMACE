"""Default trajectories of this checkout against the 1b release, one version per process (2-T23, phase 2 item 8).

Phase 2 must not change a default trajectory. Each process builds one version's engine the way its
``Simulation`` does (the minimal planner, ``tools/ab_fastmace.py``'s ``Version``) and runs MH-1 from
the structure at 0 K with the thermostat target ramped linearly, as 1b-T14's canonical-restart test
does. Rows every --every steps hold the temperature, the potential energy per atom and, with
--positions, the positions. ``compare`` then applies 1b-T14's measures across versions:

- plain cuEq float64: positions of the two versions within 1e-12 Å at every row (1b's bound for that
  backend, which is not bitwise even in one process); a repeat of the checkout gives the floor;
- hybrid float32: per seed, the mean temperature, mean potential energy and the potential energy in
  50 K target bins differ between versions by no more than max(2 x the difference between the two
  seeds of the release, floor), the floors of 1b-T14 (1 K; 0.05 meV/atom).

    python tools/trajectory_versions.py run --side 2 --backend cueq --output DIR --checkpoint mh1.model ...
    python tools/trajectory_versions.py compare --release DIR_1b --current DIR_2 [--repeat DIR_2b]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
from fastmace.bench import versions

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_fastmace  # noqa: E402
import ab_step_time  # noqa: E402

FLOAT64_LIMIT_A = 1e-12
STATISTIC_FLOORS = {"temperature_K": 1.0, "energy_meV_per_atom": 0.05, "bins_meV_per_atom_rms": 0.05}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="One version's runs in this process")
    run.add_argument("--output", required=True, type=Path, help="Fresh directory")
    run.add_argument("--side", required=True, choices=("1b", "2"), help="1b: the release copy; 2: this checkout")
    run.add_argument("--renamed", type=Path, help="Directory holding the exported fastmace_1b (for --side 1b)")
    run.add_argument("--expected-commit", default=ab_fastmace.RELEASES["1b"].commit)
    run.add_argument("--model", default="mh1-r2scan")
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--structure", default="NaCl", help="Benchmark name (under FASTMACE_STRUCTURES) or PATH@NxNxN")
    run.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    run.add_argument("--steps", type=int, default=20_000)
    run.add_argument("--every", type=int, default=20)
    run.add_argument("--seeds", type=int, nargs="+", default=[20261001, 20261101])
    run.add_argument("--targets", type=float, nargs=2, default=[300.0, 600.0], help="Thermostat ramp, K")
    run.add_argument("--positions", action="store_true", help="Record positions in each row")
    compare = sub.add_parser("compare", help="Apply 1b-T14's measures across versions")
    compare.add_argument("--release", required=True, type=Path, help="Output directory of the 1b side")
    compare.add_argument("--current", required=True, type=Path, help="Output directory of this checkout's side")
    compare.add_argument("--repeat", type=Path, help="A second run of this checkout (the plain cuEq floor)")
    args = p.parse_args(argv)
    if args.command == "run" and args.side == "1b" and args.renamed is None:
        p.error("--side 1b needs --renamed DIR (python -m fastmace.bench.versions export --ref fastmace-1b "
                "--name fastmace_1b --destination DIR)")
    return args


def target(step, steps, targets):
    first, last = targets
    return first + (last - first) * step / steps


def run(args):
    from ase import units
    release = ab_fastmace.RELEASES["1b"]
    hybrid = args.backend == "hybrid"
    from fastmace import _patches
    from fastmace.model import loading
    oeq_runtime = str(loading.prepare_hybrid()) if hybrid else None
    package, source, shared = fastmace, None, None
    if args.side == "1b":
        package, source = versions.load_renamed(args.renamed, release.renamed, args.expected_commit)
        import importlib
        for name in ("config", "planner.fixed", "planner.minimal", "model.loading", "model.forcefield",
                     "planner.workload", "_patches"):
            importlib.import_module(f"{release.renamed}.{name}")
        if hybrid:
            shared = versions.share_record(_patches, package._patches, "oeq_runtime")
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    # ab_fastmace.Version reads these fields: this checkout is side "2" against release 1b.
    side_args = argparse.Namespace(current="2", release="1b", planner_1b="minimal", model=args.model,
                                   backend=args.backend, ensemble="nvt", cell_mode="hydrostatic", seed=None,
                                   temperature_K=args.targets[0], max_temperature_K=max(args.targets))
    meta = dict(tool="trajectory_versions", status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(),
                side=args.side, release_source=source, oeq_runtime=oeq_runtime, oeq_runtime_shared=shared,
                model=args.model, checkpoint=str(checkpoint), backend=args.backend, structure=args.structure,
                steps=args.steps, every=args.every, targets=list(args.targets), seeds=list(args.seeds), runs={})
    path = args.output / "trajectory.json"

    def save():
        path.write_text(json.dumps(meta, default=str) + "\n")

    save()
    try:
        import torch
        _, base = ab_step_time.read_structure(args.structure, None)
        natoms = len(base)
        masses = base.get_masses()[:, None]
        version = ab_fastmace.Version(args.side, side_args, package)
        version.prepare()
        meta["load"] = version.load(checkpoint)
        meta["gpu"], meta["torch"] = torch.cuda.get_device_name(0), torch.__version__
        for seed in args.seeds:
            side_args.seed = seed
            started = time.perf_counter()
            engine, plan = version.build(base)
            setup_s = time.perf_counter() - started
            rows = []
            started = time.perf_counter()
            for step in range(1, args.steps + 1):
                engine.integrator.set_temperature(target(step, args.steps, args.targets))
                engine.step()
                if step % args.every == 0:
                    values = engine.sync()
                    kinetic = 0.5 * float((values["momenta"] ** 2 / masses).sum())
                    rows.append({"step": step, "target": target(step, args.steps, args.targets),
                                 "energy": float(values["energy"][0]) / natoms,
                                 "temperature": 2.0 * kinetic / (3.0 * natoms * units.kB),
                                 "positions": values["positions"].tolist() if args.positions else None})
            meta["runs"][str(seed)] = dict(plan=plan, setup_s=setup_s, run_s=time.perf_counter() - started,
                                           info={k: v for k, v in engine.info().items()
                                                 if isinstance(v, (int, float, str, bool, type(None)))},
                                           rows=rows)
            save()
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        meta["patches_applied"] = [str(record) for record in version.module("_patches").applied()]
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return meta


# ----------------------------------------------------------------------------- compare (1b-T14's measures)


def binned(rows, width=50.0):
    out = {}
    for row in rows:
        key = int(row["target"] // width)
        out.setdefault(key, []).append(row["energy"])
    return {key: float(np.mean(values)) for key, values in sorted(out.items())}


def statistics(rows):
    return {"temperature_K": float(np.mean([row["temperature"] for row in rows])),
            "energy_eV_per_atom": float(np.mean([row["energy"] for row in rows])), "bins": binned(rows)}


def differences(a, b):
    bins = [abs(a["bins"][key] - b["bins"][key]) * 1000 for key in a["bins"]]
    return {"temperature_K": abs(a["temperature_K"] - b["temperature_K"]),
            "energy_meV_per_atom": abs(a["energy_eV_per_atom"] - b["energy_eV_per_atom"]) * 1000,
            "bins_meV_per_atom_max": max(bins), "bins_meV_per_atom_rms": float(np.sqrt(np.mean(np.square(bins))))}


def max_deviation(rows, reference):
    return max(float(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"])).max())
               for a, b in zip(rows, reference, strict=True))


def _load(directory):
    meta = json.loads((Path(directory) / "trajectory.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError(f"{directory}: run status {meta.get('status')!r}")
    return meta


def compare(release, current, repeat=None):
    """The verdict record of two (or three) ``run`` outputs with the same backend, steps and seeds."""
    a, b = _load(release), _load(current)
    for key in ("backend", "steps", "every", "seeds", "targets", "structure", "model"):
        if a[key] != b[key]:
            raise ValueError(f"the runs differ in {key}: {a[key]!r} against {b[key]!r}")
    seeds = [str(seed) for seed in a["seeds"]]
    result = dict(backend=a["backend"], release=dict(side=a["side"], commit=a["fastmace_commit"]),
                  current=dict(side=b["side"], commit=b["fastmace_commit"]),
                  plans_equal={seed: a["runs"][seed]["plan"]["options"] == b["runs"][seed]["plan"]["options"]
                               for seed in seeds})
    if a["backend"] == "cueq":
        result["positions_max_A"] = {seed: max_deviation(b["runs"][seed]["rows"], a["runs"][seed]["rows"])
                                     for seed in seeds}
        if repeat is not None:
            c = _load(repeat)
            result["repeat_positions_max_A"] = {seed: max_deviation(c["runs"][seed]["rows"], b["runs"][seed]["rows"])
                                                for seed in seeds}
        result["limit_A"] = FLOAT64_LIMIT_A
        result["passed"] = (all(result["plans_equal"].values())
                            and all(value <= FLOAT64_LIMIT_A for value in result["positions_max_A"].values()))
        return result
    stats = {side: {seed: statistics(meta["runs"][seed]["rows"]) for seed in seeds}
             for side, meta in (("release", a), ("current", b))}
    between = {side: differences(stats[side][seeds[0]], stats[side][seeds[1]]) for side in stats}
    across = {seed: differences(stats["current"][seed], stats["release"][seed]) for seed in seeds}
    # The release's seeds set the scale, as the uninterrupted runs did in 1b-T14: the side under test must not
    # widen its own limit.
    limits = {key: max(2.0 * between["release"][key], floor) for key, floor in STATISTIC_FLOORS.items()}
    result.update(statistics=stats, between_seeds=between, across_versions=across, limits=limits,
                  passed=(all(result["plans_equal"].values())
                          and all(across[seed][key] <= limits[key] for seed in seeds for key in limits)))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        return run(args)
    result = compare(args.release, args.current, args.repeat)
    print(json.dumps({key: value for key, value in result.items() if key != "statistics"}, indent=2))
    return result


if __name__ == "__main__":
    main()
