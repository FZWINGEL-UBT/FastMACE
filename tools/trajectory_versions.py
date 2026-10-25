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

Phase 3 (3-T21, item 10) runs it against main before phase 3 with ``run --against p3b --side p3b|3``
(``tools/ab_fastmace.py``'s release p3b; side 3 pins ``fixcm_projection="ase"``), and asks for bitwise
equality: ``--device cpu`` (CPU float64, the plan's backend) or ``--reproducible`` (the reproducible plan on
the GPU), then ``compare --bitwise``: positions, energies and temperatures equal at every row.

Phase 3b (3b-T10, item 5) runs it against the ``fastmace-3`` tag with ``run --against 3 --side 3|3b
--expected-commit SHA`` (D3 off), in NVT and with ``--ensemble npt`` (hydrostatic BAOAB at 0 GPa): NPT rows
add the cell and the stress, and ``compare --bitwise`` compares them too.

Phase 5 (5-T11, item 4) runs it against the ``fastmace-4`` tag (dae3848) with ``run --against 4 --side 4|5``.
Phase 6 (6-T13, item 6) runs it against the ``fastmace-5`` tag (6722f29) with ``run --against 5 --side 5|6``.
Phase 4 (4-T13, item 5) runs it against the 3b release with ``run --against 3b --side 3b|4 --expected-commit SHA``,
in NVT and NPT, with ``--dispersion none`` and ``d3bj`` (D3(BJ) on both sides; ``compare`` checks it matches).
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
    run.add_argument("--against", choices=("1b", "p3b", "3", "3b", "4", "5"), default="1b",
                     help="The earlier version (tools/ab_fastmace.py's releases): 1b (sides 1b, 2), p3b (p3b, 3), "
                          "3 (3, 3b), 3b (3b, 4), 4 (4, 5) or 5 (5, 6)")
    run.add_argument("--side", required=True, choices=("1b", "2", "p3b", "3", "3b", "4", "5", "6"),
                     help="The release copy (1b, p3b, 3, 3b, 4, 5) or this checkout (2, 3, 3b, 4, 5, 6)")
    run.add_argument("--renamed", type=Path, help="Directory holding the exported release package (release side)")
    run.add_argument("--expected-commit", default=None, help="Commit (prefix) of the release copy (default: its own)")
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                     help="cpu: hide the GPUs before torch starts (CPU float64 with --backend auto)")
    run.add_argument("--ensemble", choices=("nvt", "npt"), default="nvt",
                     help="npt: hydrostatic BAOAB at 0 GPa; rows add the cell and the stress (3b-T10)")
    run.add_argument("--dispersion", choices=("none", "d3bj"), default="none",
                     help="d3bj: D3(BJ) on both sides (releases from 3b on; 4-T13)")
    run.add_argument("--reproducible", action="store_true", help="Plan with Workload(reproducible=True)")
    run.add_argument("--model", default="mh1-r2scan")
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--structure", default="NaCl", help="Benchmark name (under FASTMACE_STRUCTURES) or PATH@NxNxN")
    run.add_argument("--backend", choices=("hybrid", "cueq", "auto"), default="hybrid")
    run.add_argument("--steps", type=int, default=20_000)
    run.add_argument("--every", type=int, default=20)
    run.add_argument("--seeds", type=int, nargs="+", default=[20261001, 20261101])
    run.add_argument("--targets", type=float, nargs=2, default=[300.0, 600.0], help="Thermostat ramp, K")
    run.add_argument("--positions", action="store_true", help="Record positions in each row")
    compare = sub.add_parser("compare", help="Apply 1b-T14's measures across versions")
    compare.add_argument("--release", required=True, type=Path, help="Output directory of the 1b side")
    compare.add_argument("--current", required=True, type=Path, help="Output directory of this checkout's side")
    compare.add_argument("--repeat", type=Path, help="A second run of this checkout (the plain cuEq floor)")
    compare.add_argument("--allow-option-changes", nargs="*", default=[], metavar="OPTION",
                         help="Plan options whose difference between the versions is expected")
    compare.add_argument("--bitwise", action="store_true",
                         help="Positions (recorded), energies and temperatures equal at every row (3-T21)")
    args = p.parse_args(argv)
    if args.command == "run":
        release = ab_fastmace.RELEASES[args.against]
        if args.side not in (args.against, release.current):
            p.error(f"--against {args.against} takes --side {args.against} or {release.current}")
        if args.side == args.against and args.renamed is None:
            p.error(f"--side {args.side} needs --renamed DIR (python -m fastmace.bench.versions export --ref "
                    f"{release.tag} --name {release.renamed} --destination DIR)")
        if args.expected_commit is None:
            args.expected_commit = release.commit
        if args.expected_commit is None and args.side == args.against:
            p.error(f"--against {args.against}: give --expected-commit (the {release.tag} commit; no default yet)")
    return args


def npt_fields(values):
    """The cell and stress of an NPT row (BAOAB's output values), nothing for NVT."""
    return {key: values[key].tolist() for key in ("cell", "stress") if key in values}


def target(step, steps, targets):
    first, last = targets
    return first + (last - first) * step / steps


def run(args):
    if getattr(args, "device", "cuda") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ase import units
    against = getattr(args, "against", "1b")
    release = ab_fastmace.RELEASES[against]
    hybrid = args.backend == "hybrid"
    from fastmace import _patches
    from fastmace.model import loading
    oeq_runtime = str(loading.prepare_hybrid()) if hybrid else None
    package, source, shared = fastmace, None, None
    if args.side == against:
        package, source = versions.load_renamed(args.renamed, release.renamed, args.expected_commit)
        import importlib
        for name in ("config", "planner.fixed", "planner.minimal", "model.loading", "model.forcefield",
                     "planner.workload", "_patches"):
            importlib.import_module(f"{release.renamed}.{name}")
        if hybrid:
            shared = versions.share_record(_patches, package._patches, "oeq_runtime")
    if getattr(args, "device", "cuda") == "cuda" and ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    # ab_fastmace.Version reads these fields: this checkout is side release.current against the release, with
    # the release's integrator pins (p3b: fixcm_projection="ase").
    side_args = argparse.Namespace(current=release.current, release=against, planner_1b="minimal", model=args.model,
                                   backend=args.backend, ensemble=getattr(args, "ensemble", "nvt"),
                                   cell_mode="hydrostatic", seed=None,
                                   temperature_K=args.targets[0], max_temperature_K=max(args.targets),
                                   integrator_options=dict(ab_fastmace.RELEASE_INTEGRATOR_PINS.get(against, {})),
                                   reproducible=getattr(args, "reproducible", False),
                                   device=getattr(args, "device", "cuda"),
                                   dispersion=getattr(args, "dispersion", "none"))
    meta = dict(tool="trajectory_versions", status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(),
                side=args.side, against=against, ensemble=side_args.ensemble, device=getattr(args, "device", "cuda"),
                dispersion=side_args.dispersion,
                reproducible=side_args.reproducible,
                integrator_options=side_args.integrator_options if args.side == release.current else {},
                release_source=source, oeq_runtime=oeq_runtime, oeq_runtime_shared=shared,
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
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        meta["torch"] = torch.__version__
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
                                 "positions": values["positions"].tolist() if args.positions else None,
                                 **npt_fields(values)})
            meta["runs"][str(seed)] = dict(plan=plan, setup_s=setup_s, run_s=time.perf_counter() - started,
                                           info={k: v for k, v in engine.info().items()
                                                 if isinstance(v, (int, float, str, bool, type(None)))},
                                           rows=rows)
            save()
            del engine
            gc.collect()
            if torch.cuda.is_available():
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


def bitwise(rows, reference):
    """Rows equal in every recorded number: energy, temperature and (when recorded) positions, cell and stress."""
    return all(x["step"] == y["step"] and x["energy"] == y["energy"] and x["temperature"] == y["temperature"]
               and all(x.get(key) == y.get(key) for key in ("positions", "cell", "stress"))
               for x, y in zip(rows, reference, strict=True))


def compare(release, current, repeat=None, allowed=(), exact=False):
    """The verdict record of two (or three) ``run`` outputs with the same backend, steps and seeds.

    ``exact`` (3-T21): every row equal bitwise, with positions recorded on both sides."""
    a, b = _load(release), _load(current)
    for meta in (a, b):
        meta.setdefault("ensemble", "nvt")  # runs before 3b-T10 were NVT only
    for meta in (a, b):
        meta.setdefault("dispersion", "none")  # runs before 4-T13 had no D3
    for key in ("backend", "steps", "every", "seeds", "targets", "structure", "model", "ensemble", "dispersion"):
        if a[key] != b[key]:
            raise ValueError(f"the runs differ in {key}: {a[key]!r} against {b[key]!r}")
    seeds = [str(seed) for seed in a["seeds"]]

    def commit(meta):  # the release copy's own commit, else this checkout's
        return (meta.get("release_source") or {}).get("commit") or meta["fastmace_commit"]

    result = dict(backend=a["backend"], release=dict(side=a["side"], commit=commit(a)),
                  current=dict(side=b["side"], commit=commit(b)),
                  plan_checks={seed: ab_fastmace.compare_options(a["runs"][seed]["plan"]["options"],
                                                                 b["runs"][seed]["plan"]["options"], allowed)
                               for seed in seeds},
                  plans_equal={seed: a["runs"][seed]["plan"]["options"] == b["runs"][seed]["plan"]["options"]
                               for seed in seeds})
    if exact:
        recorded = all(row.get("positions") is not None for meta in (a, b) for seed in seeds
                       for row in meta["runs"][seed]["rows"])
        result["bitwise"] = {seed: bitwise(b["runs"][seed]["rows"], a["runs"][seed]["rows"]) for seed in seeds}
        result["positions_max_A"] = ({seed: max_deviation(b["runs"][seed]["rows"], a["runs"][seed]["rows"])
                                      for seed in seeds} if recorded else None)
        result["passed"] = (recorded and all(check["as_expected"] for check in result["plan_checks"].values())
                            and all(result["bitwise"].values()))
        return result
    if a["backend"] == "cueq":
        result["positions_max_A"] = {seed: max_deviation(b["runs"][seed]["rows"], a["runs"][seed]["rows"])
                                     for seed in seeds}
        if repeat is not None:
            c = _load(repeat)
            result["repeat_positions_max_A"] = {seed: max_deviation(c["runs"][seed]["rows"], b["runs"][seed]["rows"])
                                                for seed in seeds}
        result["limit_A"] = FLOAT64_LIMIT_A
        result["passed"] = (all(check["as_expected"] for check in result["plan_checks"].values())
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
                  passed=(all(check["as_expected"] for check in result["plan_checks"].values())
                          and all(across[seed][key] <= limits[key] for seed in seeds for key in limits)))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        return run(args)
    result = compare(args.release, args.current, args.repeat, args.allow_option_changes, exact=args.bitwise)
    print(json.dumps({key: value for key, value in result.items() if key != "statistics"}, indent=2))
    return result


if __name__ == "__main__":
    main()
