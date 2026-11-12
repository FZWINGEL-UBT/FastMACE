"""Shared setup for the late-pullback screen (12 Nov 2026): OMAT-0 medium interaction 2's w' schedule, the
paths with their dout addresses and coupling tensors, the sh frame and the N1 receiver-adjoint conversions.
CPU only (WSL mace_cueq_oeq, CUDA_VISIBLE_DEVICES='')."""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H200 = ROOT / "results" / "wprime-h200-20261111"
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "bench", H200):
    sys.path.insert(0, str(p))


def setup():
    import numpy as np
    import render_sources as rs
    from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, render_kernels
    from lab_cartesian import fit_conversion, path_table, sh_frame

    schedule, _ = render_kernels(rs.problem(), np.float32, NUM_THREADS, "ir_mul")
    blocks = rs.blocks_of(schedule)
    kernel = SimpleNamespace(schedule=schedule, blocks=blocks, channels=int(schedule.channels),
                             output_dim=int(schedule.output_dim))
    paths = path_table(kernel)
    frame = sh_frame()
    for p in paths:
        m, res = fit_conversion(p["a"], p["l"], p["L"], p["coef"], frame)
        p["M"], p["fit_residual"] = m, res
    return SimpleNamespace(schedule=schedule, blocks=blocks, kernel=kernel, paths=paths, frame=frame)
