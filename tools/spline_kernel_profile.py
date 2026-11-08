"""Nsight Compute driver for the SPLINE_W kernels: one launch each of w′ and the spline forward, on the real
arguments and on the one-slab arguments, inside a cudaProfilerStart/Stop window.

Run under ``ncu --profile-from-start off`` so the equilibration and capture (tools/spline_table_bound.py's
``capture``) stay unprofiled. The launch order inside the window is fixed:

1. wprime real
2. wprime one-slab
3. forward real
4. forward one-slab

Example (local, through tools/ci/local_gpu.py run):

    ncu --profile-from-start off --set full -o <out>/spline python tools/spline_kernel_profile.py \
        --checkpoint <mace-omat-0-medium.model> --constant MODULE:NAME=VALUE ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_step_time  # noqa: E402
import spline_table_bound as stb  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--equilibrate-steps", type=int, default=200)
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from ab_rewrites import constant_overrides

    for (module, name), value in constant_overrides(args.constant).items():
        setattr(sys.modules[module], name, value)
    captured, _, _ = stb.capture(args)
    r_max = float(captured["wprime"][1][16])
    launches = []
    for name in ("wprime", "forward"):
        kernel, real = captured[name]
        slab, inside = stb.one_slab(name, real, r_max)
        for label, a in (("real", real), ("one-slab", slab)):
            launches.append((name, label, kernel, a))
            print(f"{len(launches)}. {name} {label} (edges {real[1].shape[0]}, inside {inside})", flush=True)

    def launch(name, kernel, a):
        if name == "wprime":
            a[6].zero_()
        kernel(*a)

    for name, _, kernel, a in launches:  # compiled and warm before the window
        launch(name, kernel, a)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, _, kernel, a in launches:
        launch(name, kernel, a)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("profiled", flush=True)


if __name__ == "__main__":
    main()
