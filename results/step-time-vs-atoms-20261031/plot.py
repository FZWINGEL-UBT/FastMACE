"""Time per MD step (whole-system path) or per force call (tiled path) against atom count, MH-1 float32."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).parent
rows = list(csv.DictReader(open(here / "step_times.csv", encoding="utf-8")))

series = [
    ("whole", "NaCl", "5060Ti", "NaCl, whole system, RTX 5060 Ti 16 GB", "#1f77b4", "o", "-"),
    ("whole", "NbCl5", "5060Ti", "NbCl5, whole system, RTX 5060 Ti 16 GB", "#d62728", "o", "-"),
    ("whole", "NaCl", "Ada", "NaCl, whole system, RTX 4000 Ada 20 GB", "#1f77b4", "s", "--"),
    ("whole", "NbCl5", "Ada", "NbCl5, whole system, RTX 4000 Ada 20 GB", "#d62728", "s", "--"),
    ("tiled", "NaCl", "Ada", "NaCl, tiled (one eager force call, older code), RTX 4000 Ada 20 GB", "#17becf", "^",
     ":"),
    ("tiled", "NbCl5", "Ada", "NbCl5, tiled compiled (MD step / repeated call), RTX 4000 Ada 20 GB", "#ff7f0e", "^",
     ":"),
]

fig, ax = plt.subplots(figsize=(10, 6.5))
for path, crystal, gpu, label, color, marker, ls in series:
    pts = sorted(
        (int(r["atoms"]), float(r["seconds_per_step"]))
        for r in rows
        if (r["path"], r["crystal"], r["gpu"]) == (path, crystal, gpu)
    )
    x, y = zip(*pts)
    ax.plot(x, y, marker=marker, ls=ls, color=color, label=label, ms=5, lw=1.4)

# Constant cost per atom guides.
for us, txt in [(10, "10 µs/atom"), (30, "30 µs/atom")]:
    xs = [100, 1.2e8]
    ax.plot(xs, [v * us * 1e-6 for v in xs], color="grey", lw=0.8, alpha=0.5)
    ax.text(2.5e2, 2.5e2 * us * 1e-6 * 1.15, txt, color="grey", fontsize=8, rotation=31)

ax.annotate("16 GB limit\n115k atoms", (115200, 1.5056), xytext=(2.2e4, 4.0), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="grey"))
ax.annotate("20 GB limit\n137k NbCl5", (137376, 1.953), xytext=(1.6e5, 0.35), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="grey"))
ax.annotate("71.0M NbCl5\n1,034 s per step\n(20 GB and host limit)", (70957944, 1034), xytext=(2.0e6, 1500),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="grey"))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(150, 1.2e8)
ax.set_ylim(1.5e-3, 4e3)
ax.set_xlabel("Atoms")
ax.set_ylabel("Seconds per MD step")
ax.set_title("FastMACE, MACE MH-1 float32: best measured step time vs system size (1 Nov 2026)")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.legend(fontsize=8, loc="upper left")
fig.text(0.01, 0.01,
         "Whole system: Langevin NVT, CUDA graphs, default plan (memory-rescue rungs above ~32k atoms on 16 GB). "
         "Tiled NbCl5: compiled tiles, MD steps (884k, 9.26M) or repeated calls (40M, 71M).\nTiled NaCl and some "
         "mid-size whole-system points predate later speed-ups. "
         "Sources per point: step_times.csv.", fontsize=7, color="dimgrey")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(here / "step_time_vs_atoms.png", dpi=150)
print("saved")
