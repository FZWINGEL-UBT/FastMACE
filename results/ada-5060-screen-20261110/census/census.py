"""Offline kernel census of the ada-5060-screen candidates (10 November 2026): NVRTC builds on the CPU of the
spline forward and the w' backward for an OMAT-0-shaped problem (128 channels, 0e+1o+2e+3o inputs and edges, 10 uvu
paths, ir_mul; a synthetic three-piece spline grid, so the piece chain is not the model's), each candidate switch
on and off, for sm_89 (RTX 4000 Ada) and sm_120 (RTX 5060 Ti, w' capped at 80 registers as there). Prints registers,
spills, shared memory and SASS instruction counts per variant.
    CUDA_VISIBLE_DEVICES= PYTHONPATH=<src> python -I census.py OUTDIR"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import openequivariance as oeq
import triton

from fastmace.kernels.oeq_streaming import spline_forward as sf
from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, compile_cubin, render_kernels

BIN = Path(triton.__file__).parent / "backends/nvidia/bin"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
c = 128
irreps = "+".join(f"{c}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(4))
outs, instructions = [], []
for i1, l1 in enumerate(range(4)):
    for i2, l2 in enumerate(range(4)):
        for lo in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
            if (l1 + l2 + lo) % 2 or len(outs) >= 10:
                continue
            instructions.append((i1, i2, len(outs), "uvu", True))
            outs.append(f"{c}x{lo}{'e' if lo % 2 == 0 else 'o'}")
problem = oeq.TPProblem(oeq.Irreps(irreps), oeq.Irreps("1x0e+1x1o+1x2e+1x3o"), oeq.Irreps("+".join(outs)),
                        instructions, shared_weights=False, internal_weights=False)
schedule, _ = render_kernels(problem, np.float32, NUM_THREADS, "ir_mul")
by_dim = collections.defaultdict(list)
for slot in schedule.output_slots:
    by_dim[int(slot.irrep_dim)].append(int(slot.start))
blocks = [(d, sorted(set(s))) for d, s in sorted(by_dim.items())]
GRID = [(0.3, 1.0, 70), (1.0, 2.5, 50), (2.5, 6.0, 100)]
R_MAX = 6.0
paths = len(schedule.paths)
print("paths", paths, "edge_dim", schedule.edge_dim, "blocks", [(d, len(s)) for d, s in blocks], flush=True)

stage = dict(threads=c, min_blocks=6, stage_edges=True)
base_sf = sf.render(schedule, blocks, GRID, R_MAX, [1.0] * paths, **stage)
SF = {"sf-base": base_sf,
      "sf-packed8": sf.packed8_source(base_sf, 20),
      "sf-block_order": sf.render(schedule, blocks, GRID, R_MAX, [1.0] * paths, block_order=True, **stage),
      "sf-prefetch2": sf.render(schedule, blocks, GRID, R_MAX, [1.0] * paths, prefetch=2, **stage)}
common = dict(vector=2, table_w=True, blocks=blocks, factored=True, horner=True, spline=GRID)
sh = NUM_THREADS // 32
WP = {"wp-base": {}, "wp-butterfly": dict(butterfly=True), "wp-redux": dict(redux=True),
      "wp-stage_dout": dict(stage_dout=True), "wp-stage_sh": dict(stage_sh=sh),
      "wp-fold_alpha": dict(scale=True, fold_alpha=True),
      "wp-stage_edges": dict(stage_dout=True, stage_sh=sh, stage_edges=True)}


def census(name, source, capability, options=()):
    tag = f"{name}-sm{capability[0]}{capability[1]}" + ("-r80" if options else "")
    try:
        cubin = compile_cubin(source, capability, options)
    except Exception as error:  # noqa: BLE001
        return tag, dict(error=str(error)[-300:])
    (out / f"{tag}.cubin").write_bytes(cubin)
    sass = subprocess.check_output([str(BIN / "cuobjdump"), "-sass", str(out / f"{tag}.cubin")], text=True)
    res = subprocess.check_output([str(BIN / "cuobjdump"), "-res-usage", str(out / f"{tag}.cubin")], text=True)
    inst = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9.]+)\s", sass)
    ops = collections.Counter(x.split(".")[0] for x in inst)
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    return tag, dict(reg=int(m[1]), stack=int(m[2]), shared=int(m[3]), local=int(m[4]), instructions=len(inst),
                     **{k: ops[k] for k in ("FFMA", "LDG", "LDS", "STS", "STL", "LDL", "SHFL", "BAR", "RED", "REDUX",
                                           "ATOMG", "CCTL")})


results = {}
for capability in ((8, 9), (12, 0)):
    for name, source in SF.items():
        tag, r = census(name, source, capability)
        results[tag] = r
        print(tag, json.dumps(r), flush=True)
    for name, extra in WP.items():
        source = wp.render(schedule, "float", 4, **{**common, **extra})
        opts = ("--maxrregcount=80",) if capability == (12, 0) else ()
        tag, r = census(name, source, capability, opts)
        results[tag] = r
        print(tag, json.dumps(r), flush=True)
(out / "census.json").write_text(json.dumps(results, indent=2) + "\n")
