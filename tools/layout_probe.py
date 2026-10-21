"""Where the step's pure data-movement kernels come from (plan task OPT3-T05, layout audit).

Builds the default engine as ``tools/step_breakdown.py`` does (minimal planner, hybrid float32), makes three
eager force calls, then profiles ``--calls`` more with shapes and Python stacks. For every copy-only kernel
(``aten::copy_``, ``aten::cat``, ``aten::clone``, and Inductor kernels whose fused name has only layout parts) it
writes its CPU op, input shapes and strides, the Python stack that launched it (eager ops; Inductor kernels
show the generated wrapper), and its time. One short GPU job; use the local reservation.

    python tools/layout_probe.py --checkpoint ~/models/mace-mh-1.model --structure NaCl --output DIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import step_breakdown  # noqa: E402

LAYOUT = {"clone", "cat", "view", "slice", "transpose", "permute", "unsafe", "to", "copy", "unsqueeze", "squeeze",
          "t", "expand", "constant", "pad", "nd", "select", "backward"}
COPY_OPS = {"aten::copy_", "aten::cat", "aten::clone", "aten::contiguous", "aten::_to_copy"}


def layout_only(name):
    m = re.match(r"triton_(poi|per|red)_fused_(.*)_\d+$", name)
    return bool(m) and all(t in LAYOUT for t in m.group(2).split("_") if t)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--calls", type=int, default=3)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.rewrites = step_breakdown.rewrite_overrides(args.rewrites)
    if args.backend == "hybrid":
        ab_step_time.prepare_fastmace_hybrid()  # before anything imports MACE
    import torch
    from torch.profiler import ProfilerActivity, profile

    step_breakdown.label_triton_kernels()
    _, atoms = ab_step_time.read_structure(args.structure, None)
    engine, info = step_breakdown.build_engine(args, atoms)
    field = engine.force_field
    positions = engine.state.positions.detach().clone()
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 with_stack=True) as prof:
        for _ in range(args.calls):
            field(positions)
        torch.cuda.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(lambda: {"us": 0.0, "launches": 0})
    for event in prof.events():
        name = event.name
        if not (name in COPY_OPS or layout_only(name)):
            continue
        cuda_us = getattr(event, "device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(event, "cuda_time_total", 0.0)
        if not cuda_us:
            continue
        stack = [frame for frame in (event.stack or []) if "fastmace" in frame or "mace" in frame][:6]
        key = (name, json.dumps(event.input_shapes), json.dumps(getattr(event, "input_strides", None) or []),
               " <- ".join(stack))
        rows[key]["us"] += cuda_us
        rows[key]["launches"] += 1
    out = []
    for (name, shapes, strides, stack), row in sorted(rows.items(), key=lambda item: -item[1]["us"]):
        out.append({"name": name, "us_per_call": row["us"] / args.calls,
                    "launches_per_call": row["launches"] / args.calls, "input_shapes": json.loads(shapes),
                    "input_strides": json.loads(strides), "stack": stack})
    result = {"structure": args.structure, "gpu": torch.cuda.get_device_name(), "plan": info["describe"],
              "rows": out}
    (args.output / f"layout-{args.structure}.json").write_text(json.dumps(result, indent=1) + "\n")
    for row in out:
        print(f"{row['us_per_call']:7.1f} us x{row['launches_per_call']:.0f} {row['name'][:50]:50} "
              f"{str(row['input_shapes'])[:50]:50} {str(row['input_strides'])[:50]:50} | {row['stack'][:300]}")
    return result


if __name__ == "__main__":
    main()
