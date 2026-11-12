"""Summarise a Benchmark_loup (develop 075bf48) results tree into Markdown tables (stdlib only).

usage: python summarize.py RESULTS_DIR [RESULTS_DIR ...] > tables.md
Each RESULTS_DIR is a run_all.sh --out directory (SrTiO3 at its top, other systems below <system>/<size>).
Reads results/comparison_all.csv (written by the suite's plot_results.py) for times and errors, and the raw trial
records for the extra columns: peak GPU memory (torch max allocated, FastMACE / MACE+cuEq; nvidia-smi after the run
otherwise), LAMMPS thermo temperature at the start (step 5) and end (step 25) of the timed window, and LAMMPS
neighbour-list builds in the timed window. Values a code does not record are printed as "n/r" (not recorded).
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

# setup label in comparison_all.csv -> (raw trial dir relative to the system dir, kind)
RAW = [
    ("FastMACE (ASE", "FastMACE/results/ase_compiled_matched", "torch"),
    ("Symmetrix-XL (ASE)", "symmetrix-xl/results/ase_capacity", "smi"),
    ("Symmetrix-XL (LAMMPS, static", "symmetrix-xl/results/lammps_capacity", "lammps"),
    ("MACE + cuEquivariance", "MACE-cueq/results/ase", "torch"),
    ("Symmetrix-XL (LAMMPS, NVE", "symmetrix-xl/results/lammps_nve", "lammps"),
    ("FastMACE own MD engine (NVE", "FastMACE/results/md_nve", "torch"),
    ("Symmetrix-XL (LAMMPS, Langevin", "symmetrix-xl/results/lammps_langevin", "lammps"),
    ("FastMACE own MD engine (NVT", "FastMACE/results/md_nvt", "torch"),
]


def fmt(x, nd=2):
    if x is None or x == "":
        return "–"
    try:
        return f"{float(x):.{nd}f}"
    except ValueError:
        return str(x)


def lammps_extra(d):
    """median over trials: (T at step 5, T at step 25, neighbour builds in the timed run)."""
    t0s, t1s, nbs = [], [], []
    for log in sorted(d.glob("trial*.lammps.log")):
        txt = log.read_text(errors="replace")
        runs = txt.split("WARMUP_COMPLETE")
        if len(runs) < 2:
            continue
        timed = runs[1]
        rows = re.findall(r"^\s+(\d+)\s+\d+\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*$", timed, re.M)
        if rows:
            t0s.append(float(rows[0][3]))
            t1s.append(float(rows[-1][3]))
        m = re.search(r"Neighbor list builds = (\d+)", timed)
        if m:
            nbs.append(int(m.group(1)))
    med = lambda v: statistics.median(v) if v else None
    return med(t0s), med(t1s), med(nbs)


def trial_mem(d, kind):
    vals, after = [], []
    for j in sorted(d.glob("trial[0-9].json")):
        try:
            r = json.loads(j.read_text())
        except Exception:
            continue
        if r.get("torch_max_mem_alloc_mib") is not None:
            vals.append(r["torch_max_mem_alloc_mib"])
        if r.get("gpu_mem_mib_after") is not None:
            after.append(r["gpu_mem_mib_after"])
    if kind == "torch" and vals:
        return f"{max(vals):.0f} (torch peak)"
    if kind == "smi" and after:
        return f"{max(after):.0f} (nvidia-smi after)"
    return "n/r"


def failures(sysdir):
    log = sysdir / "logs" / "run_all.log"
    if not log.exists():
        return []
    out, block = [], None
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"\] (START|SKIP) +(.*)", line)
        if m:
            block = m.group(2)
            if m.group(1) == "SKIP":
                out.append(f"SKIP {block}")
            continue
        m = re.match(r"trial (\d+) exit=(\d+)(?: -> (.*))?", line)
        if m and m.group(2) != "0":
            msg = (m.group(3) or "").split(". ")[0][:110]
            out.append(f"{block}: trial {m.group(1)} exit {m.group(2)} ({msg})")
    return out


def table(sysdir, title):
    csvp = sysdir / "results" / "comparison_all.csv"
    print(f"\n#### {title}\n")
    fails = failures(sysdir)
    if not csvp.exists():
        print("no comparison table (run did not finish)\n")
    else:
        rows = list(csv.DictReader(csvp.open()))
        print("| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) "
              "| T start→end (K) | nbr builds | peak GPU mem (MiB) |")
        print("|---|---|---:|---:|---:|---|---:|---:|---|---:|---|")
        for r in rows:
            setup = r["setup"]
            raw = next(((p, k) for pre, p, k in RAW if setup.startswith(pre)), None)
            temp, nb, mem = "n/r", "n/r", "n/r"
            if raw:
                d = sysdir / raw[0]
                if raw[1] == "lammps":
                    t0, t1, n = lammps_extra(d)
                    if t0 is not None and "static" not in setup:
                        temp = f"{t0:.1f}→{t1:.1f}"
                    elif "static" in setup:
                        temp = "static"
                    nb = "–" if n is None else f"{n:.0f}"
                else:
                    if "MD engine" not in setup:
                        temp, nb = "static", "n/r"
                    mem = trial_mem(d, raw[1])
            ref = setup.startswith("MACE + cuEquivariance")
            med, rng, ntr = fmt(r["time_median"]), f"{fmt(r['time_min'])}–{fmt(r['time_max'])}", r["n_trials"]
            if raw and raw[1] == "lammps" and ntr == "0":
                # the suite rejects LAMMPS MD trials with a neighbour rebuild in the timed window (accepted=false);
                # show their raw loop times so the row is not empty, marked as rejected
                ts = [json.loads(j.read_text())["median_us_per_atom_step"] for j in sorted((sysdir / raw[0]).glob("trial[0-9].json"))]
                if ts:
                    med = f"({statistics.median(ts):.2f}) rejected"
                    rng = f"{min(ts):.2f}–{max(ts):.2f}"
                    ntr = f"0 of {len(ts)}"
            print(f"| {setup} | {r['boundary']} | {r['natoms'] or '–'} | {ntr} | {med} | {rng} "
                  f"| {'reference' if ref else fmt(r['force_rmse_median_meV_per_A'], 4)} "
                  f"| {'reference' if ref else fmt(r['energy_err_median_meV_per_atom'], 4)} | {temp} | {nb} | {mem} |")
    if fails:
        print("\nFailures / skips:\n")
        for f in fails:
            print(f"- {f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for root in map(Path, sys.argv[1:]):
        print(f"\n### {root.name}\n")
        if (root / "results" / "comparison_all.csv").exists() or (root / "logs" / "run_all.log").exists():
            table(root, "SrTiO3 10x10x10, 5,000 atoms")
        series = root / "cu111_h2o_omat0"
        if series.exists():
            def natoms(p):
                a, b = p.name.split("x")
                return int(a) * int(b)
            for s in sorted(series.iterdir(), key=natoms):
                table(s, f"Cu(111)-H2O, OMAT-0, {s.name} ({272 * natoms(s)} atoms)")


if __name__ == "__main__":
    main()
