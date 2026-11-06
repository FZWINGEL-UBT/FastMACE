"""G1's ceiling (H200 ideas plan, "G1. The CUTLASS 3xTF32 GEMMs with the weights split at plan time"): every recorded
3xTF32 GEMM shape (``tools/bench/gemm_shapes.py`` output) timed under CUDA graphs with each of several cubins of the
same source, alternating rounds in one process. The cubins: the shipped one, a rebuild of it ("plain"), and the
ceiling build whose B operand is not split (``mma_tensor_op_fast_f32.h`` patched to put B's raw bits into both
fragments; wrong outputs, so only its time counts). Reports per shape and in sum over a force call (``calls``) the
plain and ceiling times; the plan's stop rule: build the pre-split mainloop only if the ceiling is <= 0.95 of plain.

    python tools/bench/g1_ceiling_bench.py --shapes shapes.json --cubin plain=PATH --cubin ceiling=PATH
"""

import argparse
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (soft file locks before torch)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_rule_bench import graph_of, replay_us  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--cubin", action="append", default=[], metavar="NAME=PATH",
                        help="a cubin of tf32x3_sm80.cu for this GPU's capability (repeatable)")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch

    from fastmace.kernels.cutlass import tf32x3

    cap = tuple(torch.cuda.get_device_capability(0))
    kernels = {"shipped": tf32x3._kernels(0)}  # held, so no module is unloaded while graphs use it
    shipped_path = tf32x3.CUBINS.get(cap)
    for spec in args.cubin:
        name, path = spec.split("=", 1)
        tf32x3.CUBINS[cap] = Path(path)
        tf32x3._KERNELS.pop(0, None)
        kernels[name] = tf32x3._kernels(0)
    tf32x3.CUBINS[cap] = shipped_path
    tf32x3._KERNELS[0] = kernels["shipped"]

    gemms = json.loads(args.shapes.read_text())["gemms"]
    torch.manual_seed(0)
    rows, total = [], {n: 0.0 for n in kernels}
    for g in gemms:
        m, k, n, layout, calls = g["m"], g["k"], g["n"], g["layout"], g["calls"]
        if not tf32x3.takes(m, k, n):
            continue
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda") if layout == "rr" else torch.randn(n, k, device="cuda").t()
        outs = {name: torch.empty(m, n, device="cuda") for name in kernels}

        def run(name):
            def fn():
                tf32x3._KERNELS[0] = kernels[name]
                tf32x3.mm_into(a, b, outs[name])
            return fn

        graphs = {name: graph_of(run(name)) for name in kernels}
        tf32x3._KERNELS[0] = kernels["shipped"]
        times = {name: [] for name in graphs}
        for r in range(args.rounds):
            order = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
            for name in order:
                times[name].append(replay_us(graphs[name], args.reps))
        med = {name: sorted(v)[len(v) // 2] for name, v in times.items()}
        for name in med:
            total[name] += med[name] * calls
        same = {name: bool(torch.equal(outs[name], outs["shipped"])) for name in outs}
        row = dict(m=m, k=k, n=n, layout=layout, calls=calls, us={x: round(v, 2) for x, v in med.items()},
                   bitwise_to_shipped=same)
        rows.append(row)
        print(json.dumps(row), flush=True)
    ref = total.get("plain", total["shipped"])
    summary = dict(gpu=torch.cuda.get_device_name(), us_per_force_call={x: round(v, 1) for x, v in total.items()},
                   ratio_to_plain={x: round(v / ref, 4) for x, v in total.items()}, rows=rows)
    print(json.dumps({x: v for x, v in summary.items() if x != "rows"}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
