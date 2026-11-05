"""Record the (M, K, N, B layout) of every 3xTF32 GEMM (``tf32x3_gemm._launch``) in the first force calls of a
structure, with the plan the GPU (or ``tools/bench/gpu_alias``) chooses: the shapes a GEMM rewrite must cover.

    python tools/bench/gemm_shapes.py --structure results/loup-a100-20261103/SrTiO3.cif@10x10x10 \
        --checkpoint ~/models/mace-omat-0-medium.model --output shapes.json
"""

import argparse
import collections
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True, help="PATH@NXxNYxNZ")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", default="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    import torch
    from ase.io import read

    import fastmace as fm
    from fastmace.config import NeighbourConfig
    from fastmace.kernels.triton import tf32x3_gemm

    seen = collections.Counter()
    original = tf32x3_gemm._launch

    def launch(a, b, out):
        layout = "rc" if b.stride(0) == 1 and b.shape[1] > 1 else "rr"
        seen[(int(a.shape[0]), int(a.shape[1]), int(b.shape[1]), layout, tf32x3_gemm.uses_tf32x3_tile(
            int(a.shape[1]), int(b.shape[1])))] += 1
        return original(a, b, out)

    tf32x3_gemm._launch = launch
    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    field = fm.ForceField(fm.load(str(args.checkpoint), sha256=args.sha256), atoms, precision="float32")
    sim = fm.Simulation(atoms, field, integrator=fm.integrators.NVE(timestep_fs=1.0),
                        workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                        neighbours=NeighbourConfig(skin=0.5))
    sim.run(steps=args.steps)
    torch.cuda.synchronize()
    rows = [dict(m=m, k=k, n=n, layout=lay, tf32x3_tile=t, calls=c) for (m, k, n, lay, t), c in sorted(seen.items())]
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "structure": args.structure,
                                       "gemms": rows}, indent=1))
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
