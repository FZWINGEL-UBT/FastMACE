"""RMS and largest force errors against float64 of the TF32 and IEEE gate runs (TF32 screen, 29 October 2026).
   python gate_rms.py RUN_DIR   (reads RUN_DIR/gate-{tf32,ieee}/values/*.npz)"""
import sys

import numpy as np

root = sys.argv[1]
for gate in ("gate-tf32", "gate-ieee"):
    production = np.load(f"{root}/{gate}/values/production-fastmace.npz")
    for structure in ("NaCl", "NbCl5", "NaNbO3"):
        f64 = np.load(f"{root}/{gate}/values/reference-{structure}-f64.npz")
        f32 = np.load(f"{root}/{gate}/values/reference-{structure}-ase_f32.npz")
        keys = sorted(k for k in f64.files if "forces" in k)
        rms_p, rms_y, largest = [], [], []
        for key in keys:
            d, e = production[key] - f64[key], f32[key] - f64[key]
            rms_p.append(np.sqrt((d ** 2).mean()))
            rms_y.append(np.sqrt((e ** 2).mean()))
            largest.append(np.abs(d).max())
        p, y = np.mean(rms_p), np.mean(rms_y)
        print(gate, structure, "n", len(keys), f"RMS prod {p:.3g} f32 {y:.3g} ratio {p / y:.0f} max {max(largest):.3g}")
