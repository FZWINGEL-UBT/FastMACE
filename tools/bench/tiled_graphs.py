"""Tiled capacity path, T1: one padded CUDA graph per sweep (plans/capacity-ideas-20261030.md).

``falsifier``: the eager tiled force field in ``mode="padded"`` makes one call (slot, globals and state set
up), then sweep 2's body runs on the largest tile in the slot, eagerly and as a captured CUDA graph, ``--iters``
times each (CUDA events, after two warm-ups). Records both times, their ratio (target >= 1.8) and the largest
difference between the graph's and the eager body's outputs.

``compiled``: the same on a compiled force field (the default evaluation plan, whose model is the simplified,
traceable copy): sweep 2 on the largest tile with its forward compiled (``TileConfig(mode="compiled")``) against
the same body uncompiled on the same modules, ``--iters`` each after warm-ups, target >= 1.3; the largest
difference between their outputs.

``call``: whole force calls in ``--modes`` (eager, padded, graphs), ``--calls`` each, in one process: the time of
each call, its peak above the memory allocated before it, and the largest force, energy and stress differences
against the first mode's last call.

Writes ``<output>/graphs-<kind>-<name>-<atoms>-R<receivers>.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiled_capacity  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("falsifier", "compiled", "call"))
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--modes", nargs="+", default=["eager", "graphs"])
    p.add_argument("--calls", type=int, default=3)
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--strain", type=float, default=0.0)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def build(args, compiled=False):
    import fastmace as fm
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, args.rattle, args.strain)
    user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    if compiled:
        return atoms, planner.build_evaluation(plan, user, atoms)
    eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                   dynamic_edges=False))
    return atoms, planner.build_evaluation(eager, user, atoms)


def timed(torch, function, iters):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        function()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def falsifier(args, torch, atoms, field, record):
    from fastmace.model.tiled import TileConfig, TiledForceField

    tiled = TiledForceField(field, TileConfig(args.receivers, mode="padded"))
    positions = field._prepare_one_off(atoms)
    tiled(positions)  # slots, globals and state of this structure
    slot, g = tiled._slots[max(tiled._slots)], tiled._g
    record["slot"] = list(slot["sizes"])
    record["tiles"] = tiled.last["tiles"]
    # The largest tile (by edges) in the slot, as the call fills it.
    edge_index = field.data["edge_index"]
    n = positions.shape[0]
    row_ptr = torch.searchsorted(edge_index[1], torch.arange(n + 1, device=field.device))
    from fastmace.model.tiled import plan_tiles

    tiles, _ = plan_tiles(tiled.field.positions.detach(), field.data["cell"].reshape(3, 3), row_ptr,
                          args.receivers, tiled.last["edge_cap"])
    local_of = torch.full((n,), -1, dtype=torch.int64, device=field.device)
    pieces = [tiled._tile_graph(t, row_ptr, edge_index, local_of, slot["sizes"][0]) for t in tiles]
    piece = max(pieces, key=lambda p: p[1].shape[0])
    record["tile"] = dict(receivers=piece[3], halo=piece[4], edges=int(piece[1].shape[0]))
    del pieces
    from fastmace import _patches

    _patches.set_capture_safe_mask_head(True)  # as GraphEngine before a capture
    saved = tiled._bind(slot["count"])
    try:
        tiled._fill(slot, piece)
        tiled.heads, tiled.arange = g["heads"], g["arange"]
        for _ in range(2):
            tiled._body(2, slot, g)
        eager_ms = timed(torch, lambda: tiled._body(2, slot, g), args.iters)
        eager_out = {k: v.clone() for k, v in slot["out"][2].items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                tiled._body(2, slot, g)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.cuda.graph(graph, stream=stream):
            tiled._body(2, slot, g)
        torch.cuda.synchronize()
        record["graph_pool_allocated_bytes"] = int(torch.cuda.memory_allocated() - before)
        graph.replay()
        graph_ms = timed(torch, graph.replay, args.iters)
        diffs = {k: float((slot["out"][2][k] - eager_out[k]).abs().max()) for k in eager_out}
    finally:
        for module, count in saved:
            module.register_buffer("streaming_count", count, persistent=False)
    record.update(eager_ms=eager_ms, graph_ms=graph_ms, speedup=eager_ms / graph_ms, max_output_diff=diffs,
                  iters=args.iters)


def largest_tile(torch, tiled, field, positions, receivers):
    """The largest tile (by edges) as ``_tile_graph`` gives it at the slot's halo start."""
    from fastmace.model.tiled import plan_tiles

    edge_index = field.data["edge_index"]
    n = positions.shape[0]
    row_ptr = torch.searchsorted(edge_index[1], torch.arange(n + 1, device=field.device))
    tiles, _ = plan_tiles(tiled.field.positions.detach(), field.data["cell"].reshape(3, 3), row_ptr, receivers,
                          tiled.last["edge_cap"])
    local_of = torch.full((n,), -1, dtype=torch.int64, device=field.device)
    halo_start = tiled._slots[max(tiled._slots)]["sizes"][0]
    pieces = [tiled._tile_graph(t, row_ptr, edge_index, local_of, halo_start) for t in tiles]
    return max(pieces, key=lambda p: p[1].shape[0])


def compiled_falsifier(args, torch, atoms, field, record):
    from fastmace.model.tiled import TileConfig, TiledForceField

    config = TileConfig(args.receivers, mode="compiled")
    tiled = TiledForceField(field, config)
    positions = field._prepare_one_off(atoms)
    started = time.perf_counter()
    tiled(positions)  # slot, globals, state; compiles sweep 2 at the slot's shapes
    torch.cuda.synchronize()
    record["first_call_s"] = time.perf_counter() - started
    slot, g = tiled._slots[max(tiled._slots)], tiled._g
    record["slot"] = list(slot["sizes"])
    record["tiles"] = tiled.last["tiles"]
    piece = largest_tile(torch, tiled, field, positions, args.receivers)
    record["tile"] = dict(receivers=piece[3], halo=piece[4], edges=int(piece[1].shape[0]))
    saved = tiled._bind(slot["count"])
    try:
        tiled._fill(slot, piece)
        tiled.heads, tiled.arange = g["heads"], g["arange"]
        results = {}
        for name, mode in (("compiled", "compiled"), ("plain", "padded"), ("compiled_again", "compiled")):
            tiled.config = dataclasses.replace(config, mode=mode)
            for _ in range(3):
                tiled._body(2, slot, g)
            results[name] = dict(ms=timed(torch, lambda: tiled._body(2, slot, g), args.iters),
                                 out={k: v.clone() for k, v in slot["out"][2].items()})
        tiled.config = config
    finally:
        for module, count in saved:
            module.register_buffer("streaming_count", count, persistent=False)
    compiled_ms = min(results["compiled"]["ms"], results["compiled_again"]["ms"])
    plain = results["plain"]
    record.update(plain_ms=plain["ms"], compiled_ms=results["compiled"]["ms"],
                  compiled_again_ms=results["compiled_again"]["ms"], speedup=plain["ms"] / compiled_ms,
                  max_output_diff={k: float((results["compiled"]["out"][k] - plain["out"][k]).abs().max())
                                   for k in plain["out"]},
                  max_output_abs={k: float(v.abs().max()) for k, v in plain["out"].items()}, iters=args.iters)


def calls(args, torch, atoms, field, record):
    from fastmace.model.tiled import TileConfig, TiledForceField

    positions = field._prepare_one_off(atoms)
    results, first = {}, None
    for mode in args.modes:
        tiled = TiledForceField(field, TileConfig(args.receivers, mode=mode))
        runs = []
        for _ in range(args.calls):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            out = tiled(positions)
            torch.cuda.synchronize()
            runs.append(dict(seconds=time.perf_counter() - started,
                             peak_above_before=int(torch.cuda.max_memory_allocated() - before),
                             allocated_before=int(before), reserved=int(torch.cuda.memory_reserved())))
        entry = dict(calls=runs, stats={k: v for k, v in tiled.last.items()},
                     energy=float(out.energy.reshape(-1)[0]))
        forces = out.forces.double()
        stress = out.stress.double() if out.stress is not None else None
        if first is None:
            first = (entry["energy"], forces, stress)
        else:
            entry["max_force_diff"] = float((forces - first[1]).abs().max())
            entry["energy_diff_meV_per_atom"] = 1e3 * (entry["energy"] - first[0]) / len(atoms)
            if stress is not None:
                entry["max_stress_diff"] = float((stress - first[2]).abs().max())
        results[mode] = entry
        del tiled, out
        torch.cuda.empty_cache()
    record["modes"] = results


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    atoms, field = build(args, compiled=args.kind == "compiled" or "compiled" in args.modes)
    record = dict(kind=args.kind, structure=args.structure, repeat=args.repeat, atoms=len(atoms),
                  receivers=args.receivers, device=torch.cuda.get_device_name(),
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), edges=int(field.neighbours.count))
    {"falsifier": falsifier, "compiled": compiled_falsifier, "call": calls}[args.kind](args, torch, atoms, field,
                                                                                    record)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"graphs-{args.kind}-{args.structure}-{len(atoms)}-R{args.receivers}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
