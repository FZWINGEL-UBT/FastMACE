"""compose-outside-graph GPU smoke (12 November 2026): OMAT-0 medium, the production plan, one process.

Three engines in turn, each built and stepped alone: ``a`` with ``main``'s bind (``reference_kernels.
compose_in_graph_bind``: the capture composes the weights, the graph replays the composition), then ``b`` and ``b2``
with the library's (composed once, outside the capture). Per engine: the Python calls of ``compose_skip.composed``
inside and outside a capture (graph replays make none: a call inside a capture is a composition in the graph), the
library's ``CAPTURE_ADOPTS``, and after the steps the forces, positions, velocities and every composed weight on the
host. ``b`` against ``b2`` is the run-to-run control; ``a`` against ``b`` is the claim (bitwise where the control is).

Usage: python gpu_smoke.py --checkpoint CK --structure SrTiO3.cif@3x3x3 --steps 200 --output out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import ab_step_time  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    import reference_kernels
    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
    from fastmace.kernels.triton import compose_skip
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload
    from fastmace.rewrites.mh1 import modules as mh1

    calls = {"in_capture": 0, "outside": 0}
    composed = compose_skip.composed

    def counted(*a, **k):
        calls["in_capture" if torch.cuda.is_current_stream_capturing() else "outside"] += 1
        return composed(*a, **k)

    compose_skip.composed = counted
    library_bind = mh1.FirstInteractionReassociation.bind
    structure, base = ab_step_time.read_structure(args.structure, None)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=NeighbourConfig(skin=0.5))
    hardware = planner.hardware()
    integrator = IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=1.0)
    runs = {}
    for name in ("a", "b", "b2"):
        mh1.FirstInteractionReassociation.bind = (reference_kernels.compose_in_graph_bind if name == "a"
                                                  else library_bind)
        before, adopts = dict(calls), mh1.CAPTURE_ADOPTS
        field = ForceField(loaded, base, rewrites="auto")
        plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
        engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
        built = dict(calls)
        for _ in range(args.steps):
            engine.step()
        torch.cuda.synchronize()
        ff = engine.force_field
        weights = [w.detach().cpu().clone() for _, reassociation, table, _ in ff._tables
                   for w in reassociation.composed_buffers(table)]
        runs[name] = dict(
            state={key: getattr(engine.state, key).detach().cpu().clone() for key in ("positions", "velocities",
                                                                                       "forces")},
            weights=weights,
            counts=dict(build_in_capture=built["in_capture"] - before["in_capture"],
                        build_outside=built["outside"] - before["outside"],
                        steps_in_capture=calls["in_capture"] - built["in_capture"],
                        steps_outside=calls["outside"] - built["outside"],
                        capture_adopts=mh1.CAPTURE_ADOPTS - adopts,
                        captures=getattr(engine, "capture_count", None), composed_tensors=len(weights)))
        del engine, field, plan
        torch.cuda.empty_cache()
    mh1.FirstInteractionReassociation.bind = library_bind

    def compare(x, y):
        out = {}
        for key in x["state"]:
            a, b = x["state"][key], y["state"][key]
            out[key] = dict(bitwise=bool(torch.equal(a, b)), max_abs=float((a - b).abs().max()))
        out["weights_bitwise"] = (len(x["weights"]) == len(y["weights"])
                                  and all(torch.equal(a, b) for a, b in zip(x["weights"], y["weights"])))
        return out

    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, structure=structure,
                  natoms=len(base), steps=args.steps, commit=ab_step_time.fastmace_commit(),
                  counts={name: run["counts"] for name, run in runs.items()},
                  a_vs_b=compare(runs["a"], runs["b"]), b_vs_b2=compare(runs["b"], runs["b2"]))
    ok = (result["counts"]["b"]["build_in_capture"] == 0 and result["counts"]["b"]["steps_in_capture"] == 0
          and result["counts"]["b2"]["steps_in_capture"] == 0 and result["counts"]["b"]["composed_tensors"] > 0
          and result["counts"]["a"]["build_in_capture"] + result["counts"]["a"]["steps_in_capture"] > 0
          and result["a_vs_b"]["weights_bitwise"])
    result["ok"] = ok
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
