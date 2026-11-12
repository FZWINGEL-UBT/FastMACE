"""Render MH-1's w' backward source per GPU and variant offline (CPU only; fastmace/mh1-catchup), for the SASS census.

Adapted from results/wprime-off-polar-20261112/render_polar.py (branch fastmace/wprime-off-polar). The OEq problem
is MH-1's second interaction (interactions[1].conv_tp; the first is reassociated and has no w' kernel).
``WPrimeBackward.__init__`` runs with the device name patched and compile_cubin captured, so every flag is the
production decision for label mh1-r2scan, except the variant's explicit arguments:

  <gpu>-slope    the stored-w kernel MH-1 runs today (A100) / would run with wprime_backward on (H200): derivative
                 table of DEGREE terms (5), FACTORED etc. by GPU set, no TABLE_W (MH-1 is generic=False)
  <gpu>-tablew   TABLE_W=True as the A/B constant gives it: value table of DEGREE + 1 terms (6), HORNER and FOLD_ALPHA
                 by GPU set, so SKIP_OUTSIDE + HOIST_DX (A100, H200) and LATE_PULLBACK (H200) follow

    wsl.exe bash run_wsl.sh render_mh1.py [--checkpoint ~/models/mace-mh-1.model]
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
LABEL = "mh1-r2scan"


def problem(checkpoint, interaction=1):
    import openequivariance as oeq

    model = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    tp = model.interactions[interaction].conv_tp
    instructions = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]
    return (oeq.TPProblem(oeq.Irreps(str(tp.irreps_in1)), oeq.Irreps(str(tp.irreps_in2)),
                          oeq.Irreps(str(tp.irreps_out)), instructions, shared_weights=False, internal_weights=False),
            dict(irreps_in1=str(tp.irreps_in1), irreps_in2=str(tp.irreps_in2), irreps_out=str(tp.irreps_out),
                 instructions=[list(map(str, i)) for i in instructions], interactions=len(model.interactions)))


class _Captured(Exception):
    pass


def render(prob, layout, gpu, table_w):
    from fastmace.kernels.oeq_streaming import runtime
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    from fastmace.rewrites import radial

    name, cc = GPUS[gpu]
    schedule, kernel = runtime.render_kernels(prob, np.float32, runtime.NUM_THREADS, layout)
    attrs = kernel.ffi_attributes
    kernels = types.SimpleNamespace(source=kernel.jit_kernel, schedule=schedule, num_threads=attrs["num_threads"],
                                    cohort=attrs["logical_cohort_width"], shared_memory=attrs["shared_memory_bytes"],
                                    output_dim=attrs["output_dim"])
    patches = [(torch.cuda, "get_device_name", lambda *a, **k: name),
               (torch.cuda, "get_device_capability", lambda *a, **k: cc),
               (torch.cuda, "current_device", lambda: 0)]
    old = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
    captured = {}

    def capture(source, capability, options=(), *rest):
        captured["source"], captured["options"] = source, tuple(options)
        raise _Captured

    old_compile = runtime.compile_cubin
    try:
        for obj, attr, fn in patches:
            setattr(obj, attr, fn)
        runtime.compile_cubin = capture
        dev = torch.device("cuda", 0)
        if table_w:
            horner = wp.horner_default(dev)
            fold_alpha = wp.fold_alpha_default(dev, LABEL) and wp.factored_default(dev) and not wp.FACTORED_SCALE
            terms, vector = radial.DEGREE + 1, wp.TABLE_W_VECTOR
        else:
            horner, fold_alpha, terms, vector = False, False, radial.DEGREE, wp.SLOPE_VECTOR
        self = wp.WPrimeBackward.__new__(wp.WPrimeBackward)
        try:
            wp.WPrimeBackward.__init__(self, kernels, terms, dev, vector=vector, table_w=table_w, horner=horner,
                                       table_pairs=False, factored=True if fold_alpha else None,
                                       fold_alpha=fold_alpha, model_label=LABEL)
        except _Captured:
            pass
    finally:
        runtime.compile_cubin = old_compile
        for obj, attr, fn in old:
            setattr(obj, attr, fn)
    flags = {f: getattr(self, f, None) for f in FLAGS}
    return captured["source"], flags, schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path.home() / "models/mace-mh-1.model")
    p.add_argument("--layout", default="ir_mul")
    args = p.parse_args()
    from fastmace.kernels.oeq_streaming import wprime_backward as wp

    prob, info = problem(args.checkpoint)
    out = {"problem": info, "layout": args.layout, "variants": {}}
    (HERE / "sources").mkdir(exist_ok=True)
    for gpu in GPUS:
        for table_w in (False, True):
            vname = f"{gpu}-{'tablew' if table_w else 'slope'}"
            source, flags, schedule = render(prob, args.layout, gpu, table_w)
            (HERE / "sources" / f"{vname}.cu").write_text(source)
            weighted = [i for i, f in enumerate(wp.factored_paths(schedule)) if len(f["edges"]) > 3]
            out["variants"][vname] = dict(flags=flags, lines=len(source.splitlines()), weighted_paths=weighted)
            print(vname, flags, "paths with >3 edge comps:", weighted, flush=True)
    (HERE / "render_mh1.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(info)


if __name__ == "__main__":
    main()
