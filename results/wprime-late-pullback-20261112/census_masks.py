"""Design 2 census: every mask (sources/mask-<bits>.cu, bits l = 0..3, D direct / s spherical; HOIST_DX applied) and
the extra sources given on the command line, on sm_90 and sm_80: census_sass's edge loop + 4 channel passes, plus
the STAGE_DOUT prologue amortised (census_prologue; runs per block from runs_per_block.json).

    python census_masks.py [EXTRA.cu ...]   -> census_masks.json
"""
import json
import sys
from pathlib import Path

from common import HERE  # noqa: F401 (sys.path)
import census_prologue as cp
from census_sass import census, count

MASKS = [("".join("D" if m >> l & 1 else "s" for l in range(4))) for m in range(16)]


def prologue(sass_text_path, source_path, runs):
    rows, stage, edge, inner = cp.structure(sass_text_path)
    ops, _ = count(rows, stage["start"], stage["end"], exclude=[(edge["start"], edge["end"])])
    weighted = sum(ops.values())
    top = [s for s in inner if not any(o is not s and o["start"] <= s["start"] and s["end"] <= o["end"] for o in inner)]
    trips = cp.copy_trips(Path(source_path).read_text())
    for k, s in enumerate(top):
        body, _ = count(rows, s["start"], s["end"])
        trip = (trips[k] if k < len(trips) else 0.0) if trips else 1.0
        weighted += sum(body.values()) * (trip - 1)
    return weighted, weighted * runs / 8


def main(extra):
    runs = json.loads((HERE / "runs_per_block.json").read_text())["runs_per_block"]
    files = [HERE / "sources" / f"mask-{m}.cu" for m in MASKS] + [Path(e) for e in extra]
    out = []
    for arch in (90, 80):
        for f in files:
            r = census(str(f), arch, (), str(HERE / "sass"))
            sass = HERE / "sass" / f"{f.stem}.sm{arch}.sass"
            per_thread, per_edge = prologue(sass, f, runs)
            row = dict(file=f.name, arch=arch, registers=r["registers"], local=r["local"], shared=r["shared"],
                       channel_pass=r["channel"]["instructions"], edge_loop=r["per_edge"]["instructions"],
                       fp=r["per_edge"]["ffma_fmul_fadd"], prologue_per_thread=round(per_thread, 1),
                       prologue_per_edge=round(per_edge, 1),
                       total=round(r["per_edge"]["instructions"] + per_edge, 1))
            out.append(row)
            print(json.dumps(row), flush=True)
    (HERE / "census_masks.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
