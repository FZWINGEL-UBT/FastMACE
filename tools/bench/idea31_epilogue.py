"""Idea 31 (H200 ideas plan): bitwise check of a module-constant switch on FastMACE's production force call.

Builds engines from the minimal planner (as ``tools/ab_rewrites.py``) for one structure and model: side "a" with the
defaults, side "a2" a second build of the same (the run-to-run floor) and side "b" with ``--b-constant`` set while it
is built and stepped. Compares, bitwise, the force calls on ``--configs`` rattled copies of the structure (energy,
forces, stress where the plan computes it) and the MD state after ``--steps`` Langevin steps from the same start
(same seed: equal forces give equal trajectories). Writes ``<output>.json`` and prints one line per comparison.

``--dump-only``: build side "a" and make one force call, for ``TORCH_LOGS=output_code`` / ``TRITON_CACHE_DIR``
dumps of Inductor's generated code (idea 31 step 1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_rewrites  # noqa: E402
import ab_step_time  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--b-constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--configs", type=int, default=3)
    p.add_argument("--rattle-A", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--temperature-K", type=float, default=600.0)
    p.add_argument("--dump-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    constants = ab_rewrites.constant_overrides(args.b_constant)
    import importlib
    defaults = {key: getattr(importlib.import_module(key[0]), key[1]) for key in constants}
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    name, base = ab_step_time.read_structure(args.structure, None)
    rng = np.random.default_rng(31)
    configs = []
    for _ in range(args.configs):
        atoms = base.copy()
        atoms.positions += rng.normal(scale=args.rattle_A, size=atoms.positions.shape)
        configs.append(atoms)

    def build(values):
        with ab_rewrites._Constants(values):
            field = ForceField(loaded, base, precision="float32")
            integrator = IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=args.timestep_fs)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
        return engine

    def force_calls(engine, values):
        out = []
        with ab_rewrites._Constants(values):
            for atoms in configs:
                positions = torch.as_tensor(atoms.positions, dtype=engine.state.positions.dtype,
                                            device=engine.state.positions.device)
                result = engine.force_field(positions)
                torch.cuda.synchronize()
                out.append({key: getattr(result, key).detach().cpu().numpy().copy()
                            for key in ("energy", "forces", "stress") if getattr(result, key, None) is not None})
        return out

    def trajectory(engine, values):
        with ab_rewrites._Constants(values):
            for _ in range(args.steps):
                engine.step()
            return engine.sync()

    meta = dict(tool="idea31_epilogue", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                torch=torch.__version__, fastmace_commit=ab_step_time.fastmace_commit(), model=args.model,
                structure=args.structure, natoms=len(base), b_constant=args.b_constant, configs=args.configs,
                rattle_A=args.rattle_A, steps=args.steps, comparisons={})
    if args.dump_only:
        engine = build({})
        force_calls(engine, {})
        print("dumped", name, len(base), flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        return 0

    from fastmace.kernels.cutlass import tf32x3
    launches = {"count": 0}
    remap_div_into = tf32x3.remap_div_into

    def counted(*a, **k):  # eager launches of the epilogue kernel (graph replays do not count)
        launches["count"] += 1
        return remap_div_into(*a, **k)

    tf32x3.remap_div_into = counted
    results = {}
    meta["epilogue_launches"] = {}
    for side, values in (("a", defaults), ("b", {**defaults, **constants}), ("a2", defaults)):
        launches["count"] = 0
        engine = build(values)
        md = trajectory(engine, values)  # first: the eager calls before a capture leave stale autograd streams
        results[side] = dict(calls=force_calls(engine, values), md=md)
        meta["epilogue_launches"][side] = launches["count"]
        print(f"side {side}: epilogue launches {launches['count']}", flush=True)
        del engine
        torch.cuda.empty_cache()

    def compare(x, y):
        rows = {}
        for i, (cx, cy) in enumerate(zip(x["calls"], y["calls"])):
            for key in cx:
                diff = np.abs(cx[key].astype(np.float64) - cy[key].astype(np.float64))
                rows[f"call{i}.{key}"] = dict(bitwise=bool(np.array_equal(cx[key].view(np.uint8),
                                                                          cy[key].view(np.uint8))),
                                              max_abs=float(diff.max()) if diff.size else 0.0)
        for key, value in x["md"].items():
            other = y["md"][key]
            if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
                continue
            diff = np.abs(value.astype(np.float64) - other.astype(np.float64))
            rows[f"md.{key}"] = dict(bitwise=bool(np.array_equal(value.view(np.uint8), other.view(np.uint8))),
                                     max_abs=float(diff.max()) if diff.size else 0.0)
        return dict(bitwise=all(r["bitwise"] for r in rows.values()), rows=rows)

    for pair in (("a", "a2"), ("a", "b")):
        entry = compare(results[pair[0]], results[pair[1]])
        meta["comparisons"]["/".join(pair)] = entry
        worst = max((r["max_abs"] for r in entry["rows"].values()), default=0.0)
        print(f"{pair[1]} vs {pair[0]}: bitwise {entry['bitwise']} worst max_abs {worst:.3e}", flush=True)
        for key, row in entry["rows"].items():
            if not row["bitwise"]:
                print(f"  {key}: max_abs {row['max_abs']:.3e}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return 0 if meta["comparisons"]["a/b"]["bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
