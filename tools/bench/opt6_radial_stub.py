"""OPT6-T04 stub ceiling: the full-step gain a radial table op could give, before any table is built.

Side a is the default plan. Side b is the same plan with every folded MH-1 interaction's radial chain (the
radial embedding's Bessel and Agnesi terms, the source and target species embeddings, the joint first
layer, both radial networks' hidden layers and the density network) replaced by a stand-in for the table
op's own traffic:

- forward: the folded layer's input ``[h c, c] = [c * row, c]`` is written as ``[E, 65]`` from the cutoff
  ``c`` (``row``: 64 constants), and the edge density is ``c / 2``;
- backward: autograd reads the ``[E, 65]`` gradient and reduces it per edge to the cutoff's gradient, which
  is what the table's ``dL/dr = sum dL/dg * g'(r)`` reads.

The radial embedding's outputs other than the cutoff are then unused, so Inductor drops them; the folded
last layer, the streaming convolution and everything after stay as in side a. Side b's forces are wrong
(timing only): both sides build with ``check=False``, so no self-check or gate runs. The stub forward is the
live ``cutoff_folded_forward`` source with its radial section replaced (asserted by text, so a change of
that function stops this tool instead of timing something else).

Timing as ``tools/ab_rewrites.py`` (same-process alternating blocks, ``fastmace.bench.ab.run_ab``), but by default
(``--mode force``) one "step" is one force call at the built configuration, replayed as a captured graph: side
b's wrong forces would heat an MD run and change its edges (the 27 October smoke in ``--mode md`` ran 2.46x
slower). The table op itself also evaluates per-edge Chebyshev series from L2-resident coefficients (about 1,700
FMAs per edge, forward and derivative); that arithmetic is not in the stub, so the stub is an upper bound.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402

RADIAL_SECTION_START = "    source_embedding = self.source_embedding(node_attrs)\n"
RADIAL_SECTION_END = "        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)\n"
STUB_SECTION = '''\
    reassociation = getattr(self, "first_reassociation", None)
    chunked = reassociation is None and getattr(self, "chunked_backward_edges", 0) > 0
    tp_weights = None
    radial = self.conv_tp_weights
    # OPT6-T04 stub: the table op's [h c, c] (the cutoff c carries the r-dependence) and edge density.
    stub_input = torch.cat([cutoff * radial.folded_weight()[0, :64], cutoff], dim=1)
    if reassociation is None and not chunked:
        tp_weights = (folded_linear(stub_input, radial.folded_weight(), tile_rule=radial.folded_tiles,
                                    parts=radial.folded_parts) if radial.fused_backward
                      else torch.nn.functional.linear(stub_input, radial.folded_weight()))
    augmented = stub_input if (chunked or reassociation is not None) else None
    edge_density = cutoff * 0.5
'''
REASSOCIATED_AUGMENTED = (
    "        augmented = (self.conv_tp_weights.augmented_after_first(conv_hidden, cutoff) if joint is not None\n"
    "                     else self.conv_tp_weights.augmented(edge_feats, cutoff))\n")


def stub_forward():
    """``cutoff_folded_forward`` with the radial section replaced by the stub, in the module's namespace."""
    from fastmace.rewrites.mh1 import modules

    source = textwrap.dedent(inspect.getsource(modules.cutoff_folded_forward))
    start, end = source.index(RADIAL_SECTION_START), source.index(RADIAL_SECTION_END)
    source = source[:start] + STUB_SECTION + source[end + len(RADIAL_SECTION_END):]
    if source.count(REASSOCIATED_AUGMENTED) != 1:
        raise RuntimeError("cutoff_folded_forward changed: the reassociated branch's augmented line not found")
    source = source.replace(REASSOCIATED_AUGMENTED, "")
    namespace = dict(vars(modules))
    exec(compile(source, "opt6_radial_stub:cutoff_folded_forward", "exec"), namespace)
    function = namespace["cutoff_folded_forward"]
    for name, value in vars(modules.cutoff_folded_forward).items():  # READS_RESIDUAL_LINEAR, READS_GATED_ROWS
        setattr(function, name, value)
    return function


@contextlib.contextmanager
def stubbed():
    from fastmace.rewrites.mh1 import modules

    original = modules.cutoff_folded_forward
    modules.cutoff_folded_forward = stub_forward()
    try:
        yield
    finally:
        modules.cutoff_folded_forward = original


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("force", "md"), default="force",
                   help="force: time force calls at the built configuration (default); md: time MD steps (side b "
                        "heats, since its forces are wrong: the 27 October smoke ran 2.46x slower)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "NaNbO3"])
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE", help="Both sides' RewriteConfig")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--order", choices=("ab", "ba"), default="ab", help="Build order")
    return p.parse_args(argv)


class ForceOnlySide:
    """One force call per "step" (the engine's force stage on its current edges, as one captured CUDA graph) at
    the built configuration: side b's forces are wrong, so MD steps would heat it and change its edges."""

    def __init__(self, engine, torch):
        self.engine, self.torch = engine, torch
        self.replay = engine.captured(("opt6.force",), engine._force_stage)

    def step(self):
        self.replay()

    def prepare_block(self):
        pass

    def finish_block(self):
        pass

    def synchronize(self):
        self.torch.cuda.synchronize()


def main(argv=None):
    args = parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import ab_rewrites
    import torch
    from torch._dynamo.utils import counters

    from fastmace.bench import ab, step_time
    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    stub_forward()  # fail before any GPU work if the forward changed
    overrides = ab_rewrites.rewrite_overrides(args.rewrites)
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    order = list(args.order)
    meta = dict(tool="opt6_radial_stub", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                rewrites=overrides, sides={"a": "default plan", "b": "default plan, radial chain stubbed"},
                protocol=protocol.as_dict(), pattern=args.pattern, order=order, load=list(os.getloadavg()),
                structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        sides, setup = {}, {}
        for side in order:
            started = time.perf_counter()
            field = ForceField(loaded, base, precision="float32",
                               rewrites=RewriteConfig(**overrides) if overrides else "auto")
            integrator = IntegratorConfig(temperature_K=args.temperature_K)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            with stubbed() if side == "b" else contextlib.nullcontext():
                engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K, check=False)
            sides[side] = (ForceOnlySide(engine, torch) if args.mode == "force"
                           else ab_step_time.FastMACESide(engine, torch))
            setup[side] = time.perf_counter() - started
        for side in order:
            step_time.warm_up(sides[side], protocol.warmup)
        counted = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        entry = dict(natoms=len(base), mode=args.mode, ab=result.as_dict(), ratio_b_over_a=medians["b"] / medians["a"],
                     saving_ms=medians["a"] - medians["b"],
                     pair_ratios=[b / a for a, b in zip(blocks["a"], blocks["b"])], setup_s=setup,
                     compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: ab_step_time.FastMACESide(sides[side].engine, torch).observables()
                                  for side in sides},
                     load_after=list(os.getloadavg()),
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = not recompiled and not entry["other_gpu_processes"]
        meta["structures"][name] = entry
        print(name, " ".join(f"{s} {medians[s]:.4f} ms" for s in sides), f"b/a {entry['ratio_b_over_a']:.4f}",
              "valid" if entry["valid"] else "INVALID", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sides = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
