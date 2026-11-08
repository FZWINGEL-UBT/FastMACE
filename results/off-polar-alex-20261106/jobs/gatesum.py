"""Prints gate ratios: gatesum.py <gate log>..."""
import json
import sys

for path in sys.argv[1:]:
    print("==", path)
    for line in open(path):
        name, sep, rest = line.partition(" summary ")
        if not sep:
            continue
        d = json.loads(rest)
        cells = " ".join("%s %.2f" % (k, d[k]["ratio"]) for k in ("energy", "forces", "stress") if k in d)
        print("  ", name, cells, "PASS" if d["passed"] else "FAIL")
