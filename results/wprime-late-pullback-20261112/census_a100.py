"""sm_80 census of the A100 plan's w' (STAGE_EDGES) with and without LATE_PULLBACK -> census_a100.json."""
import json
from common import HERE
from census_masks import prologue
from census_sass import census
runs = json.loads((HERE / "runs_per_block.json").read_text())["runs_per_block"]
out = []
for name in ("a100-base.cu", "a100-lp.cu"):
    f = HERE / "sources" / name
    r = census(str(f), 80, (), str(HERE / "sass"))
    pt, pe = prologue(HERE / "sass" / f"{f.stem}.sm80.sass", f, runs)
    row = dict(file=name, registers=r["registers"], edge_loop=r["per_edge"]["instructions"], prologue_per_thread=pt,
               prologue_per_edge=round(pe, 1), total=round(r["per_edge"]["instructions"] + pe, 1))
    out.append(row)
    print(json.dumps(row))
(HERE / "census_a100.json").write_text(json.dumps(out, indent=1) + "\n")
