"""Upper bound on running edge-walk-independent kernels beside the two big edge walks (round-6 H200 breakdown).

Reads ../h200-fastpath-20261108/round6/breakdown/{kernel_sequence,ops,stages}.csv. The independent sets are
by dataflow (see README.md): beside the interaction-2 spline forward (sequence 43), the readout-1 GEMV (37) and
the skip GEMM (38); beside w' (73), the readout-1 backward (74) and the skip backward (75, 76) and w''s output
zeroing (72). The bound hides every listed kernel completely and removes its graph launch gap.
"""
import csv
import json
from pathlib import Path

here = Path(__file__).resolve().parent
src = here.parent / "h200-fastpath-20261108" / "round6" / "breakdown"
seq = list(csv.DictReader(open(src / "kernel_sequence.csv")))
ops = list(csv.DictReader(open(src / "ops.csv")))
stages = {r["stage"]: r for r in csv.DictReader(open(src / "stages.csv"))}


def us(i):
    """Time per call of sequence kernel i: its (kernel, op, input dims) row in ops.csv, split evenly over the
    row's launches."""
    k = seq[i]
    rows = [o for o in ops if o["kernel"].startswith(k["kernel"][:60]) and o["op"] == k["op"]
            and o["input_dims"] == k["input_dims"]]
    assert len(rows) == 1, (i, len(rows))
    return float(rows[0]["us_per_call"]) / float(rows[0]["launches_per_call"])


sets = {"beside spline forward (43)": [37, 38], "beside w' (73)": [72, 74, 75, 76]}
kernel_sum = sum(float(o["us_per_call"]) for o in ops)
step = float(stages["B: forces + second half kick"]["ms_per_call"]) * 1e3
launches = len(seq)
gap = (step - kernel_sum) / launches  # over-estimate: the stage also holds the kick kernels
out = {"step_us": step, "kernel_sum_us": round(kernel_sum, 1), "launches": launches, "gap_per_launch_us": round(gap, 2),
       "edge_walks_us": {"spline_forward": us(43), "wprime": us(73)}, "sets": {}}
total = 0.0
for name, idx in sets.items():
    t = sum(us(i) for i in idx)
    out["sets"][name] = {"kernels": {seq[i]["kernel"][:60]: us(i) for i in idx}, "kernel_us": round(t, 1),
                         "with_gaps_us": round(t + gap * len(idx), 1)}
    total += t + gap * len(idx)
out["ceiling_us"] = round(total, 1)
out["ceiling_share_of_step"] = round(total / step, 4)
print(json.dumps(out, indent=1))
json.dump(out, open(here / "bound.json", "w"), indent=1)
