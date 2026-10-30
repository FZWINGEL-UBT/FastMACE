"""Cost bound of rebuilding w inside the wprime backward (plans/capacity.md, the conductor's w-rebuild falsifier).

The second MH-1 interaction's streaming convolution with ``fused_folded_forward`` and ``wprime_backward`` stores the
per-edge weights ``w = a @ W.T`` (``[E, Q]``, ``Q = 1280``) in the forward and reads them in the backward. The
candidate never stores ``w``: the forward builds it on chip for the message only, and the backward builds it again
per tile from ``a`` and ``W``. Before writing that backward kernel, this tool bounds its cost from parts timed on the
real interaction's arguments (captured from one eager force call of the default plan):

- ``forward``: today's ``fused_folded_forward`` (``w`` stored); ``forward_nostore``: the same kernel without the
  ``w`` store (``store_w=False``);
- ``backward``: ``streaming_conv_wprime_backward`` (reads ``w``); ``backward_recompute``: CAP-T07's chunked
  rebuild (``streaming_conv_wprime_backward_recompute``);
- ``w_read``: one pass over ``w`` (``w.sum()``), what the backward spends reading it at best;
- ``gemm``: ``a @ W.T`` by cuBLAS over all edges, the floor of any rebuild's arithmetic (an on-chip rebuild does the
  same 2·E·Q·K flops, at best at cuBLAS's rate).

Today = forward + backward; CAP-T07 = forward + backward_recompute; the candidate's optimistic bound =
forward_nostore + (backward - w_read) + gemm. Each part is timed with CUDA events in alternating rounds, medians
over ``--rounds``. Writes ``<output>/wrebuild-<name>-<atoms>.json``. Run inside a local GPU reservation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activation_profile  # noqa: E402
import capacity_scan  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=30)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward, fused_folded_forward
    from fastmace.kernels.oeq_streaming.wrebuild_backward import WRebuildBackward
    from fastmace.kernels.triton.radial_table import radial_table_forward

    name, cell = capacity_scan.base_cell(args.structure)
    atoms = cell.repeat(tuple(int(n) for n in args.repeat.lower().split("x")))
    atoms.set_pbc(True)
    spec = importlib.util.spec_from_file_location("_ab_frontends", activation_profile.TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, 300.0, 1)
    from fastmace.config import EngineConfig

    # Eager, so that the force call reaches WPrime.convolution in Python (the kernels are the compiled plan's).
    sim = fm.Simulation(atoms, fm.ForceField(fm.load(args.model), atoms, precision="float32"), seed=1,
                        integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                           friction_per_fs=0.002),
                        engine=EngineConfig(compile_mode="none"))
    sim.run(steps=2)
    captured = []
    original = wp.WPrime.convolution

    def capture(self, *call, **kwargs):
        captured.append((self, call, kwargs))
        return original(self, *call, **kwargs)

    wp.WPrime.convolution = capture
    sim._engine.force_field(sim._engine.state.positions)
    wp.WPrime.convolution = original
    torch.cuda.synchronize()
    if not captured:
        raise SystemExit("the plan has no wprime_backward convolution")
    module, call, kwargs = captured[-1]
    lengths, pair, table, weight, x, sh, senders, receivers, row_ptr, kernels, fused = call[:11]
    fused = kwargs.get("fused", fused)
    with torch.no_grad():
        x, sh = x.detach(), sh.detach()
        r = lengths.detach()
        consts = (float(table.r_lo), float(table.bin_scale), float(table.r_max))
        a, _ = radial_table_forward(r, pair, table.coef, table.starts, table.scales, table.bins, *consts)
        senders32, receivers32 = senders.to(torch.int32), receivers.to(torch.int32)
        nostore = FusedFoldedForward(kernels, a.shape[1], x.device, store_w=False)
        rebuild = WRebuildBackward(kernels, a.shape[1], module.gprime.shape[1], x.device)
        variants = {f"backward_rebuild_p{parts}_b{batch}": WRebuildBackward(
            kernels, a.shape[1], module.gprime.shape[1], x.device, parts=parts, batch=batch)
            for parts, batch in ((1, 8), (1, 4), (2, 4))}
        out = x.new_empty((x.shape[0], nostore.output_dim))
        dummy = x.new_empty((1,))
        _, w = fused_folded_forward(a, weight, x, sh, senders32, receivers32, row_ptr, fused.id)
        dout = torch.randn_like(out)
        wbuf = torch.empty_like(w)
        interaction_rows = max(1, int(getattr(sim._engine.force_field.model.interactions[1], "chunked_backward_edges",
                                              4096)))
        parts = {
            "forward": lambda: fused_folded_forward(a, weight, x, sh, senders32, receivers32, row_ptr, fused.id),
            "forward_nostore": lambda: nostore(a, weight, x, sh, senders32, receivers32, row_ptr, out, dummy),
            "backward": lambda: wp.streaming_conv_wprime_backward(
                x, sh, w, senders32, receivers32, dout, r, pair, module.gprime, table.starts, table.scales,
                table.bins, *consts, module.kernel.id),
            "backward_recompute": lambda: wp.streaming_conv_wprime_backward_recompute(
                x, sh, senders32, receivers32, dout, r, pair, module.gprime, table.coef, weight, table.starts,
                table.scales, table.bins, *consts, module.kernel.id, interaction_rows),
            "backward_rebuild": lambda: rebuild(a, weight, x, sh, senders32, receivers32, dout, r, pair,
                                                module.gprime, table.starts, table.scales, table.bins, *consts),
            **{key: (lambda kernel=kernel: kernel(a, weight, x, sh, senders32, receivers32, dout, r, pair,
                                                   module.gprime, table.starts, table.scales, table.bins, *consts))
               for key, kernel in variants.items()},
            "w_read": lambda: w.sum(),
            "gemm": lambda: torch.matmul(a, weight.t(), out=wbuf),
        }
        # The candidate against today's backward on the stored w: dx, dsh and dL/dr (rounding-class: another
        # summation order; w itself is bitwise the forward's).
        reference = parts["backward"]()
        candidate = parts["backward_rebuild"]()
        errors = {}
        for label, ref, got in zip(("dx", "dsh", "dr"), reference, candidate):
            scale = float(ref.abs().max())
            errors[label] = dict(max_abs=float((got - ref).abs().max()), scale=scale,
                                 relative=float((got - ref).abs().max()) / scale if scale else 0.0)
        times = {key: [] for key in parts}
        for fn in parts.values():
            fn()
        torch.cuda.synchronize()
        for round_ in range(args.rounds):
            keys = list(parts) if round_ % 2 == 0 else list(reversed(parts))
            for key in keys:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                parts[key]()
                stop.record()
                stop.synchronize()
                times[key].append(start.elapsed_time(stop))
    med = {key: statistics.median(values) for key, values in times.items()}
    today = med["forward"] + med["backward"]
    recompute = med["forward"] + med["backward_recompute"]
    bound = med["forward_nostore"] + max(0.0, med["backward"] - med["w_read"]) + med["gemm"]
    best = min((key for key in med if key.startswith("backward_rebuild")), key=med.get)
    candidate_ms = med["forward_nostore"] + med[best]
    record = dict(tool="wrebuild_bound", structure=name, natoms=len(atoms), edges=int(w.shape[0]),
                  q=int(w.shape[1]), k=int(a.shape[1]), gpu=torch.cuda.get_device_name(), rounds=args.rounds,
                  chunk_rows=interaction_rows, median_ms=med, today_ms=today, recompute_ms=recompute,
                  bound_ms=bound, recompute_ratio=recompute / today, bound_ratio=bound / today,
                  candidate=best, candidate_ms=candidate_ms, candidate_ratio=candidate_ms / today, errors=errors,
                  times_ms=times)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"wrebuild-{name}-{len(atoms)}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{name} {len(atoms)} atoms, {w.shape[0]} edges, w [{w.shape[0]}, {w.shape[1]}] ({path})")
    for key, value in med.items():
        print(f"  {key:20s} {value:8.3f} ms")
    print(f"  today (forward + backward)            {today:8.3f} ms")
    print(f"  CAP-T07 (forward + backward_recompute) {recompute:8.3f} ms = {recompute / today:.3f} x today")
    print(f"  rebuild bound (nostore + backward - w_read + gemm) {bound:8.3f} ms = {bound / today:.3f} x today")
    print(f"  candidate (forward_nostore + {best}) {candidate_ms:8.3f} ms = {candidate_ms / today:.3f} x today")
    for label, values in errors.items():
        print(f"  {label}: max |candidate - today| {values['max_abs']:.3e} of max |today| {values['scale']:.3e} "
              f"({values['relative']:.2e})")
    sim.close()
    sys.stdout.flush()  # os._exit skips the buffers of a redirected stdout
    os._exit(0)


if __name__ == "__main__":
    main()
