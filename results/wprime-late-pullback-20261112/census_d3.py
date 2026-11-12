"""Census of design 3's split channel loops (census_sass counts one channel loop; here every loop inside the edge
loop that holds FFMAs is a channel loop of 4 trips): per edge = sum of channel-loop bodies x 4 + the rest of the
edge loop, plus the prologue as census_prologue. Also re-counts skip+hoist the same way as a control.

    python census_d3.py   -> census_d3.json
"""
import json

from common import HERE  # noqa: F401
import census_prologue as cp
from census_sass import count, disassemble, loops, parse


def run(path, arch, runs):
    sass, res = disassemble(path.read_text(), arch)
    (HERE / "sass" / f"{path.stem}.sm{arch}.sass").write_text(sass)
    rows, labels = parse(sass)
    stats = []
    for lo, hi in loops(rows, labels):
        ops, _ = count(rows, lo, hi)
        stats.append(dict(start=lo, end=hi, total=sum(ops.values()), ffma=ops.get("FFMA", 0)))
    heavy = [s for s in stats if s["ffma"] > 20]
    heavy = [s for s in heavy if not any(o is not s and s["start"] <= o["start"] and o["end"] <= s["end"]
                                         for o in heavy)]  # innermost FFMA loops: the channel loops
    lo, hi = min(s["start"] for s in heavy), max(s["end"] for s in heavy)
    edge = min((s for s in stats if s["start"] <= lo and s["end"] >= hi and s not in heavy),
               key=lambda s: s["end"] - s["start"])
    inside = heavy
    rest, _ = count(rows, edge["start"], edge["end"], exclude=[(s["start"], s["end"]) for s in inside])
    channel = sum(s["total"] for s in inside)
    import re
    regs = int(re.search(r"REG:(\d+)", res).group(1))
    local = int(re.search(r"LOCAL:(\d+)", res).group(1))
    return dict(file=path.name, arch=arch, registers=regs, local=local, channel_loops=[s["total"] for s in inside],
                per_edge=channel * 4 + sum(rest.values()))


def main():
    runs = json.loads((HERE / "runs_per_block.json").read_text())["runs_per_block"]
    out = []
    for arch in (90, 80):
        for name in ("skip+hoist.cu", "d3-irrepfirst.cu"):
            r = run(HERE / "sources" / name, arch, runs)
            out.append(r)
            print(json.dumps(r))
    (HERE / "census_d3.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
