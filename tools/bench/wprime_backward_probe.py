"""Backward-only w'-table probe in float32 (fp32 profile follow-up E4, 29 October 2026).

The second MH-1 interaction's backward forms ``da = dw W`` (``dw [E, 1280]``, the folded radial layer
``W [1280, 65]``; Triton skinny kernel) and then the radial table's input gradient ``dL/dr = da . a'(r)``. With the
derivative series folded through ``W`` (ideas D2, ``fastmace.kernels.triton.w_table.fold_tables``), the same
``dL/dr`` is ``dw . G'(x)``: about ``terms`` multiply-adds per weight instead of 65. This times both on the real
tables, folded weight and edges of one cell (the default float32 plan's engine after a short warm-up), on the same
random ``dw``, as CUDA-graph replays:

- ``old``: ``folded_linear_input_grad`` + ``radial_table_backward`` (the chain's backward of the radial path);
- ``new sorted``: locate + sort by (pair, piece) + ``w_table_backward`` (D2's kernel, its own tiles);
- ``new unsorted``: the same kernel on edges in receiver order (the folded series must then stay in L2).

The density column of the table is left out of ``new`` (its slope term stays with the table backward), so
``new`` is a lower bound of a real implementation. Stop rule: ``new`` must be at least 25 % faster than ``old``.
The error of ``new`` against ``old`` is the max abs difference relative to max |old|. Measurement only; idle GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import ab_step_time  # noqa: E402
import step_breakdown  # noqa: E402


def timed(torch, fn, replays, repeats):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(replays):
            out = fn()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / replays)
    return statistics.median(times), out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--structures", nargs="+", default=["NaNbO3", "NbCl5", "NaCl"])
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from fastmace.kernels.triton import w_table
    from fastmace.kernels.triton.radial_table import radial_table_backward
    from fastmace.kernels.triton.skinny_linear import folded_linear_input_grad

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    build_args = argparse.Namespace(backend="hybrid", model="mh1-r2scan", checkpoint=args.checkpoint.expanduser(),
                                    rewrites={}, temperature_K=1000.0, timestep_fs=1.0)
    results = {}
    for structure in args.structures:
        _, atoms = ab_step_time.read_structure(structure, None)
        MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(20261007))
        engine, _ = step_breakdown.build_engine(build_args, atoms)
        for _ in range(50):
            engine.step()
        engine.flush()
        field = engine.force_field
        model = getattr(field.model, "_orig_mod", field.model)
        interaction = model.interactions[1]
        table = interaction.radial_table
        weight = interaction.conv_tp_weights.folded_weight().detach().float().contiguous()
        data = field.data
        edges = int(field.edge_count)
        index = data["edge_index"][:, :edges]
        vectors = (data["positions"][index[1]] - data["positions"][index[0]] + data["shifts"][:edges])
        r = torch.linalg.norm(vectors.float(), dim=1).contiguous()
        species = data["node_attrs"][:, table.columns].argmax(dim=1).to(torch.int32)
        pair = (species[index[0]] * table.species + species[index[1]]).contiguous()
        generator = torch.Generator(device=r.device).manual_seed(1)
        dw = torch.randn((edges, weight.shape[0]), generator=generator, device=r.device, dtype=torch.float32)
        zeros = torch.zeros((edges, 1), device=r.device, dtype=torch.float32)
        constants = (float(table.r_lo), float(table.bin_scale), float(table.r_max))
        _, dg = w_table.fold_tables(table.coef, table.dcoef, weight)

        def old():
            da = folded_linear_input_grad(dw, weight)
            return radial_table_backward(r, pair, table.dcoef, table.starts, table.scales, table.bins,
                                         da, zeros, *constants)

        def new_sorted():
            x, group, order, slope = w_table._locate(r, pair, table.starts.shape[0], table.starts, table.scales,
                                                     table.bins, *constants)
            group = torch.where(group < 0, torch.full_like(group, dg.shape[0] - 1), group)
            return w_table.w_table_backward(dw, dg, x[order].contiguous(), group[order].to(torch.int32).contiguous(),
                                            order.to(torch.int32).contiguous(), slope)

        x, group, _, slope = w_table._locate(r, pair, table.starts.shape[0], table.starts, table.scales, table.bins,
                                             *constants)
        group = torch.where(group < 0, torch.full_like(group, dg.shape[0] - 1), group).to(torch.int32).contiguous()
        identity = torch.arange(edges, device=r.device, dtype=torch.int32)

        def new_unsorted():
            return w_table.w_table_backward(dw, dg, x.contiguous(), group, identity, slope)

        entry = dict(edges=edges, pairs=int(table.coef.shape[0]), pieces=int(table.coef.shape[1]),
                     terms=int(dg.shape[1]), folded_table_mb=dg.numel() * 4 / 1e6)
        reference = None
        for name, fn in (("old", old), ("new_sorted", new_sorted), ("new_unsorted", new_unsorted)):
            us, out = timed(torch, fn, args.replays, args.repeats)
            out = out.reshape(-1)
            if reference is None:
                reference = out.clone()
                error = 0.0
            else:
                error = float((out - reference).abs().max() / reference.abs().max())
            entry[name] = dict(us=us, rel_error_vs_old=error)
        entry["best_new_over_old"] = min(entry["new_sorted"]["us"], entry["new_unsorted"]["us"]) / entry["old"]["us"]
        results[structure] = entry
        print(structure, json.dumps(entry), flush=True)
        engine = field = model = None
        torch.cuda.empty_cache()
    (args.output / "wprime.json").write_text(json.dumps(dict(
        tool="wprime_backward_probe", created_utc=datetime.now(timezone.utc).isoformat(),
        gpu=torch.cuda.get_device_name(), fastmace_commit=ab_step_time.fastmace_commit(), results=results),
        indent=2) + "\n")


if __name__ == "__main__":
    main()
