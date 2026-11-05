"""Compile check of the fused forward's wgmma product (``fused_forward.WGMMA``) without a Hopper GPU: render the
kernel for a uvu problem shaped like OMAT-0's interaction (128 channels, l <= 3 edge irreps, ``--paths`` paths), with
the H200's launch numbers, compile it with NVRTC for sm_90a and print the launch rule and the cubin's resource usage
(``cuobjdump``, when on PATH).

    python tools/bench/wgmma_compile_check.py [--channels 128] [--k 65 --used 64] [--out DIR]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import fastmace  # noqa: F401  (soft file locks before torch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--k", type=int, default=65)
    parser.add_argument("--used", type=int, default=64)
    parser.add_argument("--optin", type=int, default=232448)  # H200 opt-in shared memory per block
    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--store-w", action="store_true")
    parser.add_argument("--layout", default="ir_mul")
    parser.add_argument("--out", type=Path, default=Path("wgmma-check"))
    args = parser.parse_args()

    import numpy as np
    import openequivariance as oeq

    from fastmace.kernels.oeq_streaming import fused_forward as ff
    from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, compile_cubin, render_kernels

    c = args.channels
    irreps_in1, irreps_in2 = oeq.Irreps(f"{c}x0e+{c}x1o+{c}x2e+{c}x3o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o")
    outs, instructions = [], []
    for i1, l1 in enumerate((0, 1, 2, 3)):
        for i2, l2 in enumerate((0, 1, 2, 3)):
            for lo in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                if (l1 + l2 + lo) % 2:
                    continue
                if len(outs) >= 10:
                    break
                instructions.append((i1, i2, len(outs), "uvu", True))
                outs.append(f"{c}x{lo}{'e' if lo % 2 == 0 else 'o'}")
    problem = oeq.TPProblem(irreps_in1, irreps_in2, oeq.Irreps("+".join(outs)), instructions, shared_weights=False,
                            internal_weights=False)
    schedule, _ = render_kernels(problem, np.float32, NUM_THREADS, args.layout)
    rule = ff.launch_rule_wgmma(len(schedule.paths), args.used, schedule.edge_dim, schedule.channels, args.optin,
                                args.sms)
    print("paths", len(schedule.paths), "edge_dim", schedule.edge_dim, "rule", rule, flush=True)
    source = ff.render(schedule, args.k, rule, store_w=args.store_w, used=args.used, tc=True, vec_sh=True,
                       vec_a=args.used % 32 == 0, wgmma=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fused_wgmma.cu").write_text(source)
    cubin = compile_cubin(source, (9, 0), suffix="a")
    (args.out / "fused_wgmma.cubin").write_bytes(cubin)
    print("cubin bytes", len(cubin), flush=True)
    tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    if Path(tool).exists():
        result = subprocess.run([tool, "--dump-resource-usage", str(args.out / "fused_wgmma.cubin")],
                                capture_output=True, text=True, check=False)
        print(result.stdout[-2000:], result.stderr[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
