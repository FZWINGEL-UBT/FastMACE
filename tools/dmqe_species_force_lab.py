"""Whole-force-call A/B of a lab-only frozen first-interaction species plan.

Both independent models use the same hybrid MD plan. Only side b's private
FirstInteractionReassociation._composed method reads a precomputed species
plan and row maps. Source replacement asserts the exact current statements,
so every other operation remains the owning method's verbatim code. The
class, source model, side a and production files are never patched.

Fixed-species screen only: host snapshot validation is outside timing. Any
species change requires refreshing buffers before trial evaluation, and
restoring them on rejection. Production integration needs DerivedStateRegistry
and MC coverage. This tool makes no default change and is not the formal
force/stress gate. E0 sums and contraction clones remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import platform
import statistics
import sys
import textwrap
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
from dmqe_species_lab import _capture, _timed, bitwise_equal, source_commit  # noqa: E402


def frozen_composed_source():
    """Change exactly the three species-plan regions of the current method."""
    from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

    source = textwrap.dedent(inspect.getsource(FirstInteractionReassociation._composed))
    replacements = (
        ("    onehot = node_attrs[:nodes, skip.columns]\n"
         "    kind = onehot.argmax(dim=1)\n"
         "    order, counts = species_plan(kind, onehot.shape[1], counting=self.count_plan)\n",
         "    kind, order, counts = self._dmqe_kind, self._dmqe_order, self._dmqe_counts\n"),
        ("        _, scatters = joined_rows(order, sizes)\n",
         "        scatters = [self._buffers[f'_dmqe_scatter_{k}'] for k in range(len(sizes))]\n"),
        ("        gathers, scatters = joined_rows(order, [left.shape[1] for left in lefts])\n",
         "        gathers = [self._buffers[f'_dmqe_gather_{k}'] for k in range(len(lefts))]\n"
         "        scatters = [self._buffers[f'_dmqe_scatter_{k}'] for k in range(len(lefts))]\n"),
    )
    original = source
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("owning _composed source changed; review frozen-plan replacement before running")
        source = source.replace(old, new)
    return original, source


def install_frozen_plan(field):
    """Bind snapshots and override only modules owned by this force field."""
    import torch

    from fastmace.kernels.triton.species_gemm import joined_rows, species_plan
    from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

    original, source = frozen_composed_source()
    namespace = {"torch": torch}
    exec(compile(source, "<dmqe_lab_frozen_composed>", "exec"), namespace)
    method = namespace["_composed"]
    entries = []
    for name, module in field.model.named_modules():
        if not isinstance(module, FirstInteractionReassociation) or module.compose_skip is None:
            continue
        skip = module.compose_skip[2]
        attrs = field.data["node_attrs"][:, skip.columns]
        kind = attrs.argmax(1)
        order, counts = species_plan(kind, attrs.shape[1], counting=module.count_plan)
        sizes = [2 * degree + 1 for degree in module.ls]
        gathers, scatters = joined_rows(order, sizes)
        tensors = {"_dmqe_kind": kind, "_dmqe_order": order, "_dmqe_counts": counts}
        tensors.update({f"_dmqe_gather_{i}": tensor for i, tensor in enumerate(gathers)})
        tensors.update({f"_dmqe_scatter_{i}": tensor for i, tensor in enumerate(scatters)})
        for key, tensor in tensors.items():
            module.register_buffer(key, tensor.detach().clone(), persistent=False)
        module._composed = types.MethodType(method, module)
        entries.append({"module": module, "columns": skip.columns.detach().clone(),
                        "snapshot": attrs.detach().clone(), "metadata": {
                            "name": name, "nodes": attrs.shape[0], "species": attrs.shape[1],
                            "counting": module.count_plan, "joined_layout": module.joined_layout,
                            "compose_joined": module.compose_joined, "sizes": sizes,
                            "plan_bytes": sum(t.numel() * t.element_size() for t in tensors.values())}})
    if not entries:
        raise RuntimeError("no qualifying FirstInteractionReassociation with compose_skip; this GPU's defaults "
                           "do not exercise the idea, or simplify inlined it")
    return entries, {"original_method_sha256": hashlib.sha256(original.encode()).hexdigest(),
                     "frozen_method_sha256": hashlib.sha256(source.encode()).hexdigest()}


def validate_snapshot(field, entries):
    """Intentionally outside capture/timing; this lab forbids changed species."""
    import torch

    for entry in entries:
        attrs = field.data["node_attrs"][:, entry["columns"]]
        snapshot = entry["snapshot"]
        if attrs.shape != snapshot.shape or attrs.dtype != snapshot.dtype or attrs.device != snapshot.device:
            raise ValueError("stale frozen species plan metadata; rebuild this lab's field")
        if not torch.equal(attrs, snapshot):
            raise ValueError("stale frozen species plan; rebuild before evaluating changed species")


def _build(args, loaded, base, hardware, frozen):
    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig, RewriteConfig
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", model_template_cache=False),
                                     neighbours=NeighbourConfig(skin=args.skin))
    integrator = IntegratorConfig(temperature_K=300.0, timestep_fs=1.0,
                                  ensemble="npt" if args.stress else "nvt", cell_mode="full")
    force_field = ForceField(loaded, base, precision="float32",
                            rewrites=RewriteConfig(**args.rewrites) if args.rewrites else "auto")
    plan = planner.plan(force_field, base, integrator, Workload(max_temperature_K=300.0), hardware)
    started = time.perf_counter()
    engine = planner.build(plan, force_field, base.copy(), temperature_K=300.0)
    entries, method_meta = install_frozen_plan(engine.force_field) if frozen else ([], {})
    return engine, entries, {"build_seconds": time.perf_counter() - started,
                             "plan": plan.describe(full=True), "frozen_plan": [e["metadata"] for e in entries],
                             **method_meta}


def _copy_output(output):
    return {name: None if tensor is None else tensor.detach().clone()
            for name, tensor in zip(("energy", "forces", "stress"), output)}


def _assert_independent_fields(a, b):
    from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

    if a.model is b.model or a.source_model is b.source_model:
        raise RuntimeError("A/B fields unexpectedly share an execution/source model")
    modules = [[m for m in field.model.modules() if isinstance(m, FirstInteractionReassociation)]
               for field in (a, b)]
    if set(map(id, modules[0])) & set(map(id, modules[1])):
        raise RuntimeError("A/B first-interaction modules share identity")
    storages = []
    for group in modules:
        tensors = [t for module in group for t in (*module.parameters(), *module.buffers()) if t.numel()]
        storages.append({(str(t.device), t.untyped_storage().data_ptr()) for t in tensors})
    if storages[0] & storages[1]:
        raise RuntimeError("A/B first-interaction tensors share storage")
    if any("_composed" in module.__dict__ for module in modules[0]):
        raise RuntimeError("baseline first-interaction method was unexpectedly overridden")
    return {"model_template_cache": False, "source_models_distinct": True,
            "first_interaction_modules_distinct": True, "first_interaction_storage_disjoint": True}


def _differences(a, b):
    import torch

    result = {}
    for key in a:
        if a[key] is None or b[key] is None:
            result[key] = None
        else:
            result[key] = {"bitwise_equal": bitwise_equal(a[key], b[key]),
                           "max_abs": float((a[key].double() - b[key].double()).abs().max()),
                           "reference_max_abs": float(a[key].abs().max()),
                           "finite": bool(torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all())}
    return result


def _geometry(engine, positions):
    field = engine.force_field
    field.rebuild(positions)
    field.compact(positions)
    edges = engine._padded(int(field.edge_count))
    field.use_edges(edges)
    return edges


def _case(args, built, order, base, rattle):
    import numpy as np
    import torch

    positions = np.asarray(base.positions).copy()
    if rattle:
        positions += rattle * np.random.default_rng(1).standard_normal(positions.shape)
    graphs, outputs, states, edges = {}, {}, {}, {}
    for side in order:
        engine, entries, _ = built[side]
        field = engine.force_field
        state = torch.as_tensor(positions, dtype=torch.float64, device=field.device)
        states[side] = state
        edges[side] = _geometry(engine, state)
        validate_snapshot(field, entries)
        graphs[side], outputs[side] = _capture(lambda field=field, state=state: field(state))
    checks = {}
    copies = {}
    for side in order:
        graphs[side].replay()
        torch.cuda.synchronize()
        first = _copy_output(outputs[side])
        graphs[side].replay()
        torch.cuda.synchronize()
        second = _copy_output(outputs[side])
        checks[f"{side}_same_build_spread"] = _differences(first, second)
        copies[side] = first
    checks["b_against_a"] = _differences(copies["a"], copies["b"])
    times = {"a": [], "b": []}
    for block in range(args.blocks):
        for side in (("a", "b") if block % 2 == 0 else ("b", "a")):
            times[side].append(_timed(graphs[side], args.rounds))
    medians = {side: statistics.median(samples) for side, samples in times.items()}
    return {"rattle_A": rattle, "padded_edges": edges, "median_us": medians,
            "ratio_b_over_a": medians["b"] / medians["a"],
            "removed_us": medians["a"] - medians["b"], "samples_us": times, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="omat0-medium")
    parser.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@6x6x6")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--constant", nargs="*", default=[])
    parser.add_argument("--rewrites", nargs="*", default=[])
    parser.add_argument("--stress", action="store_true", help="NPT-planned field; include stress in calls and checks")
    parser.add_argument("--rattles", default="0,0.05")
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--order", choices=("ab", "ba", "both"), default="both")
    args = parser.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from ab_rewrites import constant_overrides, rewrite_overrides

    from fastmace.config import EngineConfig
    from fastmace.model import loading
    from fastmace.planner import minimal

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    args.rewrites = rewrite_overrides(args.rewrites)
    for (module, name), value in constant_overrides(args.constant).items():
        setattr(sys.modules[module], name, value)
    hardware = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid")).hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    label, base = ab_step_time.read_structure(args.structure, None)
    commit = source_commit()
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
              "gpu": torch.cuda.get_device_name(), "python": sys.executable, "torch": torch.__version__,
              "fastmace_commit": commit, "baseline_commit": commit, "model": args.model,
              "checkpoint_sha256": sha, "structure": label, "natoms": len(base),
              "backend": "hybrid", "precision": "float32", "stress": args.stress,
              "constants": args.constant, "rewrites": args.rewrites,
              "source_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "CUDA-graph force-call replay from the MD plan, no integration or neighbour rebuild in timing",
              "lifecycle": "fixed species only; production event/MC refresh and rollback remain unimplemented",
              "orders": {}}
    for order in (("ab", "ba") if args.order == "both" else (args.order,)):
        built = {}
        for side in order:
            print(f"building {order}/{side}", flush=True)
            built[side] = _build(args, loaded, base, hardware, side == "b")
        independence = _assert_independent_fields(built["a"][0].force_field, built["b"][0].force_field)
        record = {"builds": {side: entry[2] for side, entry in built.items()},
                  "independence": independence, "cases": []}
        for rattle in (float(value) for value in args.rattles.split(",")):
            case = _case(args, built, order, base, rattle)
            record["cases"].append(case)
            print(f"order={order} rattle={rattle}: {case['median_us']}, "
                  f"b/a={case['ratio_b_over_a']:.6f}", flush=True)
        result["orders"][order] = record
        (args.output / "species_force.json").write_text(json.dumps(result, indent=2) + "\n")
        built = None
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
