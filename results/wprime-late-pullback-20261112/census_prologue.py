"""The STAGE_DOUT prologue's cost, which census_sass.py leaves out (it counts the channel pass x 4 + the edge loop).

Per run of equal receivers each of the block's 128 threads executes the prologue once; a block holds 32 edges, one warp
per edge. In census units (warp instructions per edge): prologue_per_edge = P x runs_per_block / 8, P = the per-thread
instruction count of one prologue pass. P: the stage loop's instructions outside the edge loop, with every loop inside
the prologue weighted by its trip count for thread 0..127 (averaged over threads; the copy loops' bounds are read from
the source in order). Prints the loop structure so the attribution can be checked.

    python census_prologue.py --arch 90 sass/skip+hoist.sm90.sass sources/skip+hoist.cu ...
"""
import argparse
import json
import re
import sys
from pathlib import Path

from common import H200, HERE  # noqa: F401  (sys.path)
from census_sass import count, loops, parse


def structure(sass_path):
    rows, labels = parse(Path(sass_path).read_text())
    found = loops(rows, labels)
    stats = []
    for lo, hi in found:
        ops, _ = count(rows, lo, hi)
        stats.append(dict(start=lo, end=hi, total=sum(ops.values()), ffma=ops.get("FFMA", 0)))
    top = max(s["ffma"] for s in stats)
    channel = min((s for s in stats if s["ffma"] >= 0.8 * top), key=lambda s: s["end"] - s["start"])
    around = [s for s in stats if s["start"] <= channel["start"] and s["end"] >= channel["end"] and s is not channel]
    edge = min(around, key=lambda s: s["end"])
    outer = [s for s in stats if s["start"] <= edge["start"] and s["end"] >= edge["end"] and s is not edge]
    stage = min(outer, key=lambda s: s["end"] - s["start"])
    inner = [s for s in stats if stage["start"] <= s["start"] and s["end"] <= stage["end"]
             and not (s["start"] >= edge["start"] and s["end"] <= edge["end"]) and s is not stage]
    return rows, stage, edge, inner


def copy_trips(source):
    """Average trips per thread (128 threads) of the plain copy's float4 loops, in source order."""
    bounds = [int(b) for b in re.findall(r"for \(int i = threadIdx\.x; i < (\d+); i \+= blockDim\.x\)\n\s*"
                                         r"reinterpret_cast<float4\*>", source)]
    return [sum(len(range(t, b, 128)) for t in range(128)) / 128 for b in bounds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=int, default=90)
    ap.add_argument("--runs-per-block", type=float, required=True)
    ap.add_argument("pairs", nargs="+", help="SASS SOURCE pairs")
    args = ap.parse_args()
    out = []
    for sass, source in zip(args.pairs[::2], args.pairs[1::2]):
        rows, stage, edge, inner = structure(sass)
        ops, _ = count(rows, stage["start"], stage["end"], exclude=[(edge["start"], edge["end"])])
        static = sum(ops.values())
        # inner loops: the copy loops (float4 path) get their trip counts; nested loops counted once (outermost)
        top_inner = [s for s in inner if not any(o is not s and o["start"] <= s["start"] and s["end"] <= o["end"]
                                                 for o in inner)]
        trips = copy_trips(Path(source).read_text())
        weighted = static
        detail = []
        if top_inner:
            # the vectorised copy loops are the first len(trips) loops (the scalar fallbacks follow and do not run)
            for k, s in enumerate(top_inner):
                body, _ = count(rows, s["start"], s["end"])
                n = sum(body.values())
                trip = (trips[k] if k < len(trips) else 0.0) if trips else 1.0
                weighted += n * (trip - 1)
                detail.append(dict(start=hex(s["start"]), instructions=n, trips=trip))
        per_edge = weighted * args.runs_per_block / 8
        r = dict(file=Path(source).name, arch=args.arch, static=static, per_thread_pass=round(weighted, 1),
                 inner_loops=detail, runs_per_block=args.runs_per_block, per_edge=round(per_edge, 1),
                 ops=dict(ops.most_common(12)))
        out.append(r)
        print(json.dumps(r))
    Path(HERE / f"census_prologue_sm{args.arch}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
