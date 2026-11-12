"""Owner's question (12 November 2026, via the conductor): do LATE_PULLBACK's 56-64 registers rescue a switch that lost
to registers before? Offline census only (sm_90 and sm_80, census_sass edge loop + channel passes, plus the STAGE_DOUT
prologue amortised as in census_masks):

- H200 plan: lp (= mask-DDDD: skip render, LATE_PULLBACK, HOIST_DX) against lp + CHANNEL_VECTOR 2 / 4 (skip render,
  LATE_PULLBACK, then wprime_tuning.vec; vec subsumes the hoist).
- A100 plan: a100-lp (STAGE_EDGES render, LATE_PULLBACK) against a100-lp+skip+hoist (wprime_tuning.skip, then
  LATE_PULLBACK, then wprime_tuning.hoist).
- Every source runs through emulate2's float64 check where it can (vec: not emulated, see the README).

Blocks per SM from registers (allocation per warp in units of 256), shared memory (+1 KiB reserved per block;
sm_90 228 KiB, sm_80 164 KiB per SM), 128 threads per block, at most 16 such blocks (2,048 threads).

    python census_followup.py   -> census_followup.json, sources/fu-*.cu
"""
import json
import math

from common import H200, HERE, setup
from pathlib import Path

import census_prologue as cp
from census_masks import prologue
from census_sass import census, count, loops, parse

SM = {90: 228 * 1024, 80: 164 * 1024}


def blocks_per_sm(regs, shared, arch, threads=128):
    warps = threads // 32
    per_warp = math.ceil(regs * 32 / 256) * 256
    by_regs = 65536 // (per_warp * warps)
    by_smem = SM[arch] // (shared + 1024) if shared else 32
    return min(by_regs, by_smem, 2048 // threads, 32)


def single_pass(sass_path, source_path, runs):
    """vec4 (128 channels in one pass): the compiler drops the channel loop, so census_sass's attribution does not
    apply. The edge loop = of the loops holding >= 80 % of the most FFMAs, apart from the stage loop (the largest),
    the one that ends first; per edge = its whole body. The prologue as census_masks.prologue, with that edge loop."""
    rows, labels = parse(Path(sass_path).read_text())
    stats = []
    for lo, hi in loops(rows, labels):
        ops, _ = count(rows, lo, hi)
        stats.append(dict(start=lo, end=hi, total=sum(ops.values()), ffma=ops.get("FFMA", 0)))
    top = max(s["ffma"] for s in stats)
    heavy = [s for s in stats if s["ffma"] >= 0.8 * top]
    stage = max(heavy, key=lambda s: s["end"] - s["start"])
    edge = min((s for s in heavy if s is not stage), key=lambda s: s["end"])
    ops, _ = count(rows, stage["start"], stage["end"], exclude=[(edge["start"], edge["end"])])
    weighted = sum(ops.values())
    inner = [s for s in stats if stage["start"] <= s["start"] and s["end"] <= stage["end"] and s is not stage
             and not (s["start"] >= edge["start"] and s["end"] <= edge["end"])
             and not (s["start"] <= edge["start"] and s["end"] >= edge["end"])]
    trips = cp.copy_trips(Path(source_path).read_text())
    for k, s in enumerate(inner):
        body, _ = count(rows, s["start"], s["end"])
        trip = (trips[k] if k < len(trips) else 0.0) if trips else 1.0
        weighted += sum(body.values()) * (trip - 1)
    return edge["total"], weighted * runs / 8


def sources():
    from fastmace.kernels.oeq_streaming import wprime_pullback as wpb
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt
    s = setup()
    skip_src = (H200 / "sources" / "wprime-h200-skip.cu").read_text()
    a100_src = (H200 / "sources" / "wprime-a100.cu").read_text()
    lp = wpb.apply(skip_src, s.schedule, s.blocks, True)
    out = {"fu-h200-lp.cu": wt.hoist(lp),
           "fu-h200-lp-vec2.cu": wt.vec(lp, 2),
           "fu-h200-lp-vec4.cu": wt.vec(lp, 4),
           "fu-a100-lp.cu": wpb.apply(a100_src, s.schedule, s.blocks, True),
           "fu-a100-lp-nohoist-skip.cu": wpb.apply(wt.skip(a100_src), s.schedule, s.blocks, True)}
    out["fu-a100-lp-skip-hoist.cu"] = wt.hoist(out["fu-a100-lp-nohoist-skip.cu"])
    for name, text in out.items():
        (HERE / "sources" / name).write_text(text)
    return out


def main():
    runs = json.loads((HERE / "runs_per_block.json").read_text())["runs_per_block"]
    names = sources()
    rows = []
    for arch in (90, 80):
        for name in names:
            if "nohoist" in name:
                continue
            f = HERE / "sources" / name
            n = 4 if "vec4" in name else 2 if "vec2" in name else 1
            r = census(str(f), arch, (), str(HERE / "sass"), trips=4 / n)
            sass = HERE / "sass" / f"{f.stem}.sm{arch}.sass"
            if n == 4:
                loop, pe = single_pass(sass, f, runs)
                channel = None
            else:
                _, pe = prologue(sass, f, runs)
                loop, channel = r["per_edge"]["instructions"], r["channel"]["instructions"]
            row = dict(file=name, arch=arch, registers=r["registers"], local=r["local"], shared=r["shared"],
                       channel_pass=channel, passes=4 / n, edge_loop=loop, prologue_per_edge=round(pe, 1),
                       total=round(loop + pe, 1), blocks_per_sm=blocks_per_sm(r["registers"], r["shared"], arch))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (HERE / "census_followup.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
