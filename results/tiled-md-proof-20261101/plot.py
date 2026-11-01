"""MD at 70,957,944 NbCl5 atoms on one 20 GB GPU: the recorded NVE runs, plotted (owner request, 1 November 2026).

Reads the ``tiled_md.py`` records listed in ``RUNS`` (paths relative to the repository root) and writes:

- ``md_proof.csv``: one row per run and step, with every value plotted and its source file. A value the run did not
  record is left empty.
- ``md_proof.png``: energies per atom, temperature, wall time per step, GPU and host memory per step, and the
  884,736-atom 200-step NVE drift (eager against compiled) as the context for a longer run.

Temperature is 2 E_kin / (3 N k_B), with N the atom count (the zero total momentum removes 3 of the 3N degrees of
freedom, a 4e-8 relative difference here). Run as ``python results/tiled-md-proof-20261101/plot.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KB_EV = 8.617333262e-5
GIB = 2**30

RUNS = [  # (label, record, machine)
    ("leto, compact list cache, auto mode (eager first call)",
     "results/tiled-md-proof-20261101/runs/leto-nve-proof-NbCl5-69x69x207/md-NbCl5-70957944-R4096.json", "leto"),
    ("murphy, list cache off, compiled",
     "results/tiled-md-host-20261101/runs/murphy-nve-host-NbCl5-69x69x207/md-NbCl5-70957944-R4096.json", "murphy"),
]
CONTEXT = [  # the 884,736-atom 200-step NVE runs (tiled-compiled-default.md)
    ("eager",
     "results/tiled-compiled-default-20261101/runs/murphy-nve2-eager-NbCl5-16x16x48/md-NbCl5-884736-R4096.json"),
    ("compiled",
     "results/tiled-compiled-default-20261101/runs/murphy-nve2-compiled-NbCl5-16x16x48/md-NbCl5-884736-R4096.json"),
]
COLUMNS = ["run", "machine", "source", "commit", "atoms", "step", "time_fs", "epot_eV_per_atom", "ekin_eV_per_atom",
           "etot_eV_per_atom", "temperature_K", "wall_seconds", "call_mode", "call_receivers", "gpu_peak_GiB",
           "gpu_total_GiB", "host_peak_GiB", "host_total_GiB", "list_cache_bytes"]


def load(path):
    record = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return record, record["runs"]["tiled"]


def gib(row, key):
    return row[key] / GIB if row.get(key) is not None else None


def rows_of(label, path, machine):
    record, run = load(path)
    n, dt = record["atoms"], record["timestep_fs"]
    marks = [row["seconds"] for row in run["rows"]]
    out = []
    for i, row in enumerate(run["rows"]):
        # A step's wall time: the integrator and the force call that ends it; step 0 is the first call alone.
        wall = row.get("call_seconds") if i == 0 else marks[i] - marks[i - 1]
        out.append(dict(run=label, machine=machine, source=path, commit=record["source_commit"], atoms=n,
                        step=row["step"], time_fs=row["step"] * dt, epot_eV_per_atom=row["epot"] / n,
                        ekin_eV_per_atom=row["ekin"] / n, etot_eV_per_atom=row["etot"] / n,
                        temperature_K=2 * row["ekin"] / (3 * n * KB_EV), wall_seconds=wall,
                        call_mode=row.get("mode"), call_receivers=row.get("call_receivers"),
                        gpu_peak_GiB=gib(row, "gpu_peak_bytes"), gpu_total_GiB=gib(row, "gpu_total_bytes"),
                        host_peak_GiB=gib(row, "host_peak_bytes"), host_total_GiB=gib(row, "host_total_bytes"),
                        list_cache_bytes=row.get("list_cache_bytes")))
    return record, run, out


def main():
    runs = [(label, machine) + rows_of(label, path, machine) for label, path, machine in RUNS
            if (ROOT / path).exists()]
    with open(HERE / "md_proof.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        for *_, rows in runs:
            for row in rows:
                writer.writerow({k: ("" if row[k] is None else row[k]) for k in COLUMNS})

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    (ax_e, ax_t, ax_w), (ax_g, ax_h, ax_c) = axes
    styles = ["o-", "s--"]
    colours = {"etot": "black", "epot": "#d62728", "ekin": "#1f77b4"}
    for (label, machine, record, run, rows), style in zip(runs, styles):
        steps = [r["step"] for r in rows]
        for key, name in (("etot", "total"), ("epot", "potential"), ("ekin", "kinetic")):
            base = rows[0][f"{key}_eV_per_atom"]
            ax_e.plot(steps, [1e3 * (r[f"{key}_eV_per_atom"] - base) for r in rows], style, color=colours[key],
                      ms=5, label=f"{name}, {machine}")
        drift = 1e3 * (rows[-1]["etot_eV_per_atom"] - rows[0]["etot_eV_per_atom"])
        ax_e.annotate(f"{machine}: total drift {drift:+.3f} meV/atom over {len(rows) - 1} steps",
                      (steps[-1], 1e3 * (rows[-1]["etot_eV_per_atom"] - rows[0]["etot_eV_per_atom"])),
                      xytext=(0.03, 0.06 + 0.08 * styles.index(style)), textcoords="axes fraction", fontsize=8)
        ax_t.plot(steps, [r["temperature_K"] for r in rows], style, color="#9467bd", ms=5, label=machine)
        walls = [(r["step"], r["wall_seconds"], r["call_mode"]) for r in rows if r["wall_seconds"] is not None]
        offset = -0.2 if styles.index(style) == 0 else 0.2
        for step, wall, mode in walls:
            colour = "#ff7f0e" if mode == "eager" else "#2ca02c" if mode == "compiled" else "grey"
            ax_w.bar(step + offset, wall, width=0.38, color=colour, edgecolor="black", linewidth=0.5,
                     hatch="" if styles.index(style) == 0 else "//")
            ax_w.text(step + offset, wall + 15, f"{wall:.0f}", ha="center", fontsize=7)
        memory = [(r["step"], r["gpu_peak_GiB"], r["host_peak_GiB"]) for r in rows if r["gpu_peak_GiB"] is not None]
        if memory:
            ax_g.plot([m[0] for m in memory], [m[1] for m in memory], style, color="#2ca02c", ms=6, label=machine)
            ax_h.plot([m[0] for m in memory], [m[2] for m in memory], style, color="#8c564b", ms=6, label=machine)
            gpu_total, host_total = rows[0]["gpu_total_GiB"], rows[0]["host_total_GiB"]
            ax_g.axhline(gpu_total, color="grey", ls=":")
            ax_g.text(0, gpu_total * 0.97, f"card: {gpu_total:.2f} GiB", fontsize=8, va="top")
            ax_h.axhline(host_total, color="grey", ls=":")
            ax_h.text(0, host_total * 0.98, f"host: {host_total:.0f} GiB", fontsize=8, va="top")
    ax_e.set(xlabel="MD step (2 fs)", ylabel="E(step) − E(0)  [meV/atom]", title="Energies per atom (NVE)")
    top = ax_e.secondary_xaxis("top", functions=(lambda s: 2 * s, lambda t: t / 2))
    top.set_xlabel("time [fs]")
    ax_e.legend(fontsize=7, ncol=2)
    ax_t.set(xlabel="MD step (2 fs)", ylabel="temperature [K]", title="Temperature (start: 600 K Maxwell–Boltzmann)")
    ax_t.legend(fontsize=8)
    ax_w.set(xlabel="MD step (step 0: the first force call)", ylabel="wall time [s]",
             title="Wall time per step")
    from matplotlib.patches import Patch

    ax_w.legend(handles=[Patch(color="#ff7f0e", label="eager call"), Patch(color="#2ca02c", label="compiled call"),
                         Patch(color="grey", label="mode not recorded (murphy: compiled)"),
                         Patch(facecolor="white", edgecolor="black", label="plain: leto"),
                         Patch(facecolor="white", edgecolor="black", hatch="//", label="hatched: murphy (cache off)")],
                fontsize=7, loc="upper right")
    ax_g.set(xlabel="MD step", ylabel="GiB", title="GPU peak per step (allocated by PyTorch)", ylim=(0, 21))
    ax_h.set(xlabel="MD step", ylabel="GiB", title="Host peak resident memory per step", ylim=(0, 260))
    for ax in (ax_e, ax_t, ax_w, ax_g, ax_h):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in (ax_g, ax_h):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.text(0.5, 0.5, "not recorded", ha="center", transform=ax.transAxes)
    for label, path in CONTEXT:
        record, run = load(path)
        n = record["atoms"]
        etot = [row["etot"] for row in run["rows"]]
        ax_c.plot([row["step"] for row in run["rows"]], [1e3 * (e - etot[0]) / n for e in etot],
                  label=f"{label}: drift {run['drift_meV_per_atom']:.4f} meV/atom", lw=1.2)
    ax_c.set(xlabel="MD step (2 fs)", ylabel="E_tot(step) − E_tot(0)  [meV/atom]",
             title="Context: 884,736 atoms, 200 NVE steps\n(murphy, 32a459ba)")
    ax_c.legend(fontsize=8)
    commits = ", ".join(sorted({f"{rows[0]['machine']} {rows[0]['commit']}" for *_, rows in runs}))
    fig.suptitle(f"NbCl5 70,957,944 atoms, MACE MH-1 float32, one NVIDIA RTX 4000 Ada 20 GB — NVE MD, "
                 f"tiled force field, 1 November 2026 ({commits})", fontsize=12)
    fig.text(0.01, 0.005, "Values from the tiled_md.py records; one row per point in md_proof.csv with its source. "
             "Memory per step was recorded only in the leto run.", fontsize=8, color="dimgrey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(HERE / "md_proof.png", dpi=130)
    print("saved", HERE / "md_proof.png")


if __name__ == "__main__":
    main()
