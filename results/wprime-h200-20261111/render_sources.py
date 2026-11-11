"""Render today's w' backward sources for OMAT-0 medium's interaction 2 offline (CPU, no GPU).

Builds the OEq problem of interaction 2 (input 128x0e+128x1o, edge 0e+1o+2e+3o, 10 uvu paths in MACE's
instruction order), the ir_mul schedule, linear_1's block layout (one block per output degree) and the spline grid
of the captured production kernel, then renders ``wprime_backward.render`` with each GPU's defaults:

  a100  FACTORED, FOLD_ALPHA, HORNER, SCALAR_PATHS, BUTTERFLY, STAGE_DOUT, STAGE_SH, STAGE_EDGES, SPLINE_W,
        SPLINE_VECTOR 4, BLOCK_MESSAGE, TABLE_W
  h200  the same without STAGE_EDGES (STAGE_EDGES_GPUS has no H200; omat0-medium has no label entry)

Check: the a100 render must equal the source captured by the A100 production run (batch2/ghoist-prod, cd7575bb);
that pins the schedule, block order and grid. Writes sources/ beside this file. Run in WSL mace_cueq_oeq with
FastMACE on PYTHONPATH and CUDA_VISIBLE_DEVICES=''.
"""
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
CAPTURED = ROOT / "results/a100-colab-ideas-20261111/batch2/ghoist-prod/sources/wprime-real.cu"


def problem(channels=128):
    import openequivariance as oeq

    ins = [(0, 1), (1, -1)]
    edge = [(l, (-1) ** l) for l in range(4)]
    outs, instructions = [], []
    for i1, (l1, p1) in enumerate(ins):
        for i2, (l2, p2) in enumerate(edge):
            for lo in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                if p1 * p2 != (-1) ** lo:
                    continue
                instructions.append((i1, i2, len(outs), "uvu", True))
                outs.append(lo)
    # MACE (tp_out_irreps_with_instructions) sorts the output irreps and renumbers the instructions' outputs
    order = sorted(range(len(outs)), key=lambda k: outs[k])
    position = {k: n for n, k in enumerate(order)}
    # ... and the paths (weights) follow the sorted outputs (the captured kernel's weight order)
    instructions = sorted(((a, b, position[o], m, t) for a, b, o, m, t in instructions), key=lambda i: i[2])
    irreps_out = "+".join(f"{channels}x{outs[k]}{'e' if outs[k] % 2 == 0 else 'o'}" for k in order)
    return oeq.TPProblem(oeq.Irreps(f"{channels}x0e+{channels}x1o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o"),
                         oeq.Irreps(irreps_out), instructions, shared_weights=False, internal_weights=False)


def grid_from(source):
    """The spline grid [(start, end, intervals)] from a captured source's interval lookup."""
    rows = re.findall(r"\(clamped - (\S+?)f\) \* sx_inv; .*?il = il < (\d+) \? il", source)
    ends = re.findall(r"if \(clamped < (\S+?)f\)", source)
    starts = [float.fromhex(s) for s, _ in rows]
    counts = [int(n) + 1 for _, n in rows]
    last = starts[-1] + counts[-1] / float.fromhex(re.findall(r"sx_inv = (\S+?)f;", source)[-1])
    stops = [float.fromhex(e) for e in ends] + [round(last, 12)]
    return [(a, b, n) for a, b, n in zip(starts, stops, counts)]


def blocks_of(schedule):
    by_dim = {}
    for slot in sorted(schedule.output_slots, key=lambda s: int(s.start)):
        by_dim.setdefault(int(slot.irrep_dim), [])
        if int(slot.start) not in by_dim[int(slot.irrep_dim)]:
            by_dim[int(slot.irrep_dim)].append(int(slot.start))
    return tuple((d, tuple(by_dim[d])) for d in sorted(by_dim))


def sources():
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, render_kernels

    schedule, _ = render_kernels(problem(), np.float32, NUM_THREADS, "ir_mul")
    captured = CAPTURED.read_text()
    grid = grid_from(captured)
    common = dict(vector=4, table_w=True, blocks=blocks_of(schedule), factored=True, scale=True, horner=True,
                  scalar_paths=True, butterfly=True, stage_dout=True, fold_alpha=True, spline=grid,
                  stage_sh=NUM_THREADS // 32)
    out = {"a100": wp.render(schedule, "float", 4, stage_edges=True, **common),
           "h200": wp.render(schedule, "float", 4, stage_edges=False, **common)}
    return schedule, grid, captured, out


def main():
    schedule, grid, captured, out = sources()
    (HERE / "sources").mkdir(exist_ok=True)
    for name, text in out.items():
        (HERE / "sources" / f"wprime-{name}.cu").write_text(text)
    print("grid", grid)
    print("blocks", blocks_of(schedule))
    def lines(text):  # the template lost one blank line since cd7575bb (EDGE_FORCE)
        return [line for line in text.splitlines() if line.strip()]

    same = lines(out["a100"]) == lines(captured)
    print("a100 render == captured production source (blank lines aside):", same)
    if not same:
        import difflib
        diff = list(difflib.unified_diff(lines(captured), lines(out["a100"]), lineterm="", n=1))
        print("\n".join(diff[:80]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
