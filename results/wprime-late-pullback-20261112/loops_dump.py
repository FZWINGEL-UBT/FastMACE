"""The loops of a SASS dump (census_sass.loops): address range, instructions, FFMAs, atomics. Checks census_followup's
loop attribution (vec4 runs its 128 channels in one pass).

    python loops_dump.py sass/X.sm90.sass ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wprime-h200-20261111"))
from census_sass import count, loops, parse  # noqa: E402

for p in sys.argv[1:]:
    rows, labels = parse(Path(p).read_text())
    print(p)
    for lo, hi in loops(rows, labels):
        ops, _ = count(rows, lo, hi)
        atoms = sum(v for k, v in ops.items() if k.startswith(("RED", "ATOM")))
        print(f"  {lo:6x}-{hi:6x} total {sum(ops.values()):5d} ffma {ops.get('FFMA', 0):4d} atomics {atoms}")
