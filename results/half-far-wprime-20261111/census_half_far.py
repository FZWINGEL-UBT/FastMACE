"""Offline sm_80 / sm_90 census of HALF_FAR's generated kernels (from the captured A100 w' source) beside the float32
w' kernel with the skip transform. CPU only (NVRTC + cuobjdump)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "bench")]
from cu_census import census  # noqa: E402
from fastmace.kernels.oeq_streaming import half_far  # noqa: E402
from fastmace.kernels.oeq_streaming.spline_forward import lookup_code  # noqa: E402

src = (ROOT / "results/a100-colab-ideas-20261111/screen1/ghoist/sources/wprime-real.cu").read_text()
grid = [(0.3, 0.5, 9), (0.5, 2.5, 90), (2.5, 6.0, 157)]  # a grid of the spline's form (the far band at the base spacing)
lookup = lookup_code(grid, "clamped", "float", "            ")
out = HERE / "census"
files = {"half_far": out / "half_far.cu", "fp32_skip": out / "wprime_skip.cu", "fp32": out / "wprime.cu"}
files["half_far"].write_text(half_far.kernel_source(src, lookup), newline="\n")
files["fp32_skip"].write_text(half_far.with_skip(src), newline="\n")
files["fp32"].write_text(src, newline="\n")
files["half_far_b2"] = out / "half_far_b2.cu"
files["half_far_b2"].write_text(half_far.kernel_source(src, lookup, min_blocks=2), newline=chr(10))
rows = {}
for arch in ("80", "90"):
    for name, f in files.items():
        r = census(f, arch)
        rows[f"{name}_sm{arch}"] = r
        print(name, arch, {k: v for k, v in r.items() if k not in ("longest_block_ops", "file")})
        print("   loop", r["longest_block_ops"])
(HERE / "census_half_far.json").write_text(json.dumps(rows, indent=1) + "\n")
