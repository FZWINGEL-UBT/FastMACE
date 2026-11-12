"""Offline NVRTC census (PyTorch 2.15 trial, conductor 12 Nov 2026): one NVRTC per process (runtime.NVRTC), each source
compiled for sm_80 and sm_90 with the production options; per kernel: registers, local memory (spills), static SASS
instructions, FFMA, LDL/STL, and, where census_sass.py finds the w' loop nest, the per-edge instruction count.

    python census_nvrtc.py LIBNVRTC OUT.json SOURCE.cu ...   (PYTHONPATH=<checkout>, PATH with triton's cuobjdump first)

CENSUS_ARCHS (default "80 90") picks the targets, e.g. "89 120" for the sources an Ada or a 5060 Ti dumped.
"""
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

lib, out, sources = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
from fastmace.kernels.oeq_streaming import runtime  # noqa: E402

runtime.NVRTC = lib
sys.path.insert(0, str(Path(runtime.__file__).parents[3] / "results" / "wprime-h200-20261111"))
import census_sass  # noqa: E402

rows = []
for src in sources:
    text = Path(src).read_text()
    for arch in [int(a) for a in os.environ.get("CENSUS_ARCHS", "80 90").split()]:
        row = {"source": Path(src).name, "arch": arch}
        try:
            cubin = runtime.compile_cubin(text, (arch // 10, arch % 10))
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "k.cubin"
                f.write_bytes(cubin)
                sass = subprocess.run(["cuobjdump", "-sass", str(f)], capture_output=True, text=True).stdout
                res = subprocess.run(["cuobjdump", "-res-usage", str(f)], capture_output=True, text=True).stdout
            ops = collections.Counter(m.group(1).split(".")[0] for m in
                                      re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", sass))
            row.update(registers=int(re.search(r"REG:(\d+)", res).group(1)),
                       local=int(re.search(r"LOCAL:(\d+)", res).group(1)),
                       shared=int(re.search(r"SHARED:(\d+)", res).group(1)),
                       instructions=sum(ops.values()), ffma=ops.get("FFMA", 0),
                       ldl_stl=ops.get("LDL", 0) + ops.get("STL", 0), cubin_bytes=len(cubin))
            try:
                c = census_sass.census(src, arch)
                row["per_edge"] = c["per_edge"]["instructions"]
                row["per_edge_non_fma"] = c["per_edge"]["non_fma"]
                row["channel_pass"] = c["channel"]["instructions"]
            except Exception as e:  # noqa: BLE001 (no w' loop nest: the forward)
                row["per_edge"] = None
                row["loop_note"] = f"{type(e).__name__}: {e}"[:120]
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:600]
        rows.append(row)
        print(json.dumps(row), flush=True)
out.write_text(json.dumps({"nvrtc_lib": lib, "rows": rows}, indent=1) + "\n")
