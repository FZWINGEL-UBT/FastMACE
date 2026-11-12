"""Render POLAR-1-M's production w' backward source on each GPU offline (CPU, no GPU; fastmace/wprime-off-polar).

The OEq problem is read from the checkpoint's second interaction (interactions[1].conv_tp: irreps and instructions;
the first one is reassociated and has no w' kernel). ``WPrimeBackward.__init__`` itself runs with the device name
patched (torch.cuda.get_device_name / capability) and compile_cubin replaced by a capture, so every switch is
decided by the production code for the model label (MH-1 class: generic=False, so no TABLE_W, HORNER or
FOLD_ALPHA). For each GPU it records the decided flags and writes sources/<name>.cu:

  a100-default     A100-SXM4-80GB defaults for polar1-m (WEIGHTED_VECTOR on by WEIGHTED_VECTOR_GPUS)
  a100-noweighted  the same with WEIGHTED_VECTOR = False
  a100-all-on      SKIP_OUTSIDE / LATE_PULLBACK forced True (HOIST_DX default): no change without TABLE_W
  h200-default     H200 defaults (not used by any plan: polar1-m runs no w' on the H200; for the record)

    wsl.exe bash run_wsl.sh render_polar.py [--checkpoint ~/models/MACE-POLAR-1-M.model] [--terms 6]
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

GPUS = {"a100": ("NVIDIA A100-SXM4-80GB", (8, 0)), "h200": ("NVIDIA H200", (9, 0))}
FLAGS = ("table_w", "factored", "horner", "fold_alpha", "scalar_paths", "butterfly", "redux", "stage_dout",
         "stage_sh", "stage_edges", "skip_outside", "hoist_dx", "late_pullback", "weighted_vector", "channel_vector")


def problem(checkpoint, interaction=1):
    import openequivariance as oeq

    model = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    tp = model.interactions[interaction].conv_tp
    instructions = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]
    return (oeq.TPProblem(oeq.Irreps(str(tp.irreps_in1)), oeq.Irreps(str(tp.irreps_in2)),
                          oeq.Irreps(str(tp.irreps_out)), instructions, shared_weights=False, internal_weights=False),
            dict(irreps_in1=str(tp.irreps_in1), irreps_in2=str(tp.irreps_in2), irreps_out=str(tp.irreps_out),
                 instructions=[list(map(str, i)) for i in instructions]))


class _Captured(Exception):
    pass


def render(prob, layout, gpu, terms, label="polar1-m", overrides=None):
    from fastmace.kernels.oeq_streaming import runtime
    from fastmace.kernels.oeq_streaming import wprime_backward as wp

    name, cc = GPUS[gpu]
    schedule, kernel = runtime.render_kernels(prob, np.float32, runtime.NUM_THREADS, layout)
    attrs = kernel.ffi_attributes
    kernels = types.SimpleNamespace(source=kernel.jit_kernel, schedule=schedule, num_threads=attrs["num_threads"],
                                    cohort=attrs["logical_cohort_width"], shared_memory=attrs["shared_memory_bytes"],
                                    output_dim=attrs["output_dim"])
    saved = {k: getattr(wp, k) for k in (overrides or {})}
    patches = [(torch.cuda, "get_device_name", lambda *a, **k: name),
               (torch.cuda, "get_device_capability", lambda *a, **k: cc),
               (torch.cuda, "current_device", lambda: 0)]
    old = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
    captured = {}

    def capture(source, capability, options=()):
        captured["source"], captured["options"] = source, tuple(options)
        raise _Captured

    old_compile = runtime.compile_cubin
    try:
        for obj, attr, fn in patches:
            setattr(obj, attr, fn)
        for k, v in (overrides or {}).items():
            setattr(wp, k, v)
        runtime.compile_cubin = capture
        self = wp.WPrimeBackward.__new__(wp.WPrimeBackward)
        try:
            wp.WPrimeBackward.__init__(self, kernels, terms, torch.device("cuda", 0), vector=wp.SLOPE_VECTOR,
                                       table_w=False, horner=False, table_pairs=False, factored=None,
                                       fold_alpha=False, model_label=label)
        except _Captured:
            pass
    finally:
        runtime.compile_cubin = old_compile
        for obj, attr, fn in old:
            setattr(obj, attr, fn)
        for k, v in saved.items():
            setattr(wp, k, v)
    flags = {f: getattr(self, f, None) for f in FLAGS}
    return captured["source"], flags, schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path.home() / "models/MACE-POLAR-1-M.model")
    p.add_argument("--layout", default="ir_mul")
    p.add_argument("--terms", type=int, default=6)
    args = p.parse_args()
    from fastmace.kernels.oeq_streaming import wprime_backward as wp

    prob, info = problem(args.checkpoint)
    out = {"problem": info, "layout": args.layout, "terms": args.terms, "variants": {}}
    (HERE / "sources").mkdir(exist_ok=True)
    variants = {"a100-default": ("a100", None), "a100-noweighted": ("a100", {"WEIGHTED_VECTOR": False}),
                "a100-all-on": ("a100", {"SKIP_OUTSIDE": True, "LATE_PULLBACK": True}),
                "h200-default": ("h200", None)}
    for vname, (gpu, overrides) in variants.items():
        source, flags, schedule = render(prob, args.layout, gpu, args.terms, overrides=overrides)
        (HERE / "sources" / f"{vname}.cu").write_text(source)
        weighted = [i for i, f in enumerate(wp.factored_paths(schedule)) if len(f["edges"]) > 3]
        out["variants"][vname] = dict(flags=flags, lines=len(source.splitlines()), weighted_paths=weighted)
        print(vname, flags, "paths with >3 edge comps:", weighted, flush=True)
    same = (HERE / "sources/a100-all-on.cu").read_text() == (HERE / "sources/a100-default.cu").read_text()
    out["a100_all_on_equals_default"] = same
    print("a100: SKIP_OUTSIDE/LATE_PULLBACK forced on leaves the source unchanged:", same)
    (HERE / "render_polar.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
