import json
from pathlib import Path

import numpy as np

O = Path("pkg-results")
for s in ("NaCl", "NbCl5", "NaNbO3"):
    try:
        a, b = np.load(f"f64pkg/{s}-stock.npz"), np.load(f"f64pkg/{s}-pkg.npz")
    except FileNotFoundError:
        continue
    n = len(a["forces"])
    df = np.abs(b["forces"] - a["forces"])
    ds = np.abs(b["stress"] - a["stress"])
    print(f"{s}: max|dF| {df.max():.2e} eV/A (max|F| {np.abs(a['forces']).max():.2f}), rms {np.sqrt((df**2).mean()):.1e}; "
          f"max|dS| {ds.max():.1e} eV/A^3 (max|S| {np.abs(a['stress']).max():.1e}); dE/atom {(float(b['energy']) - float(a['energy'])) / n:.1e} eV")
    for side in ("stock", "pkg"):
        log = O / f"sb-{s}-{side}.log"
        if log.exists():
            text = log.read_text()
            wall = [l for l in text.splitlines() if "wall_ms_per_step" in l]
            print(f"   {side}: {wall[-1].strip() if wall else 'no timing'}")
for s in ("NaCl", "NbCl5"):
    for side in ("stock", "pkg"):
        files = list((O / f"nve-{s}-{side}").glob("*.json")) if (O / f"nve-{s}-{side}").exists() else []
        if files:
            d = json.loads(files[0].read_text())
            keys = [k for k in d if "drift" in k or "rms" in k or "range" in k or k == "ms_per_step"]
            print(f"nve {s} {side}:", {k: d[k] for k in keys})
