"""Idea 13 step 1: per-edge epilogue SASS of the census variants.

    python tools/bench/sh_recompute_epilogue.py ARCH   (after sh_recompute_census.py wrote /tmp/sh13/*.cu)

Counts the instructions from the last dx atomic (RED/ATOM) to the next BAR or EXIT: the end-of-edge reduction and
store, which runs once per edge.
"""
import collections
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cu_census  # noqa: E402

arch = sys.argv[1]
for name in ("base", "gather", "split"):
    cubin = cu_census.compile_cubin(Path(f"/tmp/sh13/{name}.cu").read_text(), arch)
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "k.cubin")
        Path(f).write_bytes(cubin)
        sass = subprocess.run([cu_census.tool("cuobjdump"), "-sass", f], capture_output=True, text=True).stdout
    Path(f"/tmp/sh13/{name}-{arch}.sass").write_text(sass)
    ins = []
    for line in sass.splitlines():
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            ins.append(m.group(3).split(".")[0])
    reds = [i for i, op in enumerate(ins) if op.startswith("RED") or op.startswith("ATOM")]
    last = reds[-1]
    end = next(i for i in range(last, len(ins)) if ins[i] in ("BAR", "EXIT"))
    # the epilogue straight-line region, plus everything after the main EXIT (subroutines: slow paths)
    epi = collections.Counter(ins[last + 1:end])
    print(f"sm_{arch} {name}: total {len(ins)}, epilogue {end - last - 1} "
          f"{dict(epi.most_common(12))}")
