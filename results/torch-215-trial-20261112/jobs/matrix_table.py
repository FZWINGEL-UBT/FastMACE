"""§5 table from */matrix/run.log: ms/step per variant (mean of the forward and reversed passes) and change vs a.

  python matrix_table.py RUN_LOG [RUN_LOG ...]
"""
import re
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    t = defaultdict(list)
    last = None
    for line in open(path, encoding="utf-8"):
        m = re.search(r"end ab-(\w+)-(f1|r1)-(\w+) exit", line)
        if m:
            last = (m.group(1), m.group(3))
            continue
        m = re.match(r"point \S+ .*?: (ok, ([\d.]+) ms/step|error.*)", line)
        if m and last:
            t[last].append(float(m.group(2)) if m.group(2) else None)
            last = None
    print(path)
    for s in sorted({k[0] for k in t}):
        base = [x for x in t[(s, "a")] if x]
        a = sum(base) / len(base)
        row = []
        for v in ("a", "c4", "c8", "t9", "n2", "n4", "n5", "b", "b0"):
            xs = [x for x in t.get((s, v), []) if x]
            row.append(f"{v} failed" if not xs else f"{v} {sum(xs)/len(xs):.2f} ({100*(sum(xs)/len(xs)/a-1):+.1f}%) [{', '.join(f'{x:.2f}' for x in xs)}]")
        print(" ", s, " | ".join(row))
