"""Dump the SASS (cuobjdump) and resource use of NVRTC sources compiled by one libnvrtc (torch 2.15 trial, 12 Nov 2026).

    python sass_dump.py LIBNVRTC ARCH OUTDIR SOURCE.cu [--opt OPTION ...]   (PATH with a CUDA 13 cuobjdump first)
Writes OUTDIR/<source>.sm<ARCH>.sass and .res; extra NVRTC options (--opt) go to runtime.compile_cubin.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

lib, arch, outdir, src = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
opts = tuple(sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--opt")
from fastmace.kernels.oeq_streaming import runtime  # noqa: E402

runtime.NVRTC = lib
outdir.mkdir(parents=True, exist_ok=True)
cubin = runtime.compile_cubin(src.read_text(), (arch // 10, arch % 10), opts) if opts else \
    runtime.compile_cubin(src.read_text(), (arch // 10, arch % 10))
with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "k.cubin"
    f.write_bytes(cubin)
    for kind, flag in (("sass", "-sass"), ("res", "-res-usage")):
        (outdir / f"{src.stem}.sm{arch}.{kind}").write_text(
            subprocess.run(["cuobjdump", flag, str(f)], capture_output=True, text=True).stdout)
print(src.name, arch, len(cubin), "bytes", opts)
