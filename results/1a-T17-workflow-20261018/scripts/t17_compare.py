"""Compare the 1a-T17 shortened anneals: --engine gpu versus --engine fastmace, two seeds each.

Measures as studies/annealing/README.md: relaxation, MD potential energy relative to the relaxed start
(peak, mean per 100 K bin of the target temperature on each ramp), thermostat tracking, quench energies
(heating, cooling, final) relative to the relaxed start. Engine differences are judged against the
seed-to-seed differences of each engine.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
SEEDS = ("20261001", "20261101")
ENGINES = ("gpu", "fastmace")


def load(engine, seed):
    run = ROOT / f"anneal-{engine}-{seed}"
    crystal = run / "NaCl"
    meta = json.loads((crystal / "metadata.json").read_text())
    rows = list(csv.DictReader((crystal / "temperature.csv").open()))
    start = meta["initial_relaxation"]["energy_eV_atom"]
    quench_dirs = [d for d in crystal.glob("quench-*") if (d / "metadata.json").is_file()]
    quench = json.loads((quench_dirs[0] / "metadata.json").read_text()) if quench_dirs else None
    qrows = list(csv.DictReader((quench_dirs[0] / "quench.csv").open())) if quench_dirs else []
    return dict(meta=meta, rows=rows, start=start, quench=quench, qrows=qrows,
                run=json.loads((run / "run.json").read_text()))


def summary(d):
    rows, start = d["rows"], d["start"]
    epot = [(float(r["target_K"]), r["phase"], 1000 * (float(r["potential_eV_atom"]) - start), float(r["temperature_K"]))
            for r in rows]
    peak = max(epot, key=lambda x: x[2])
    bins = {}
    for target, phase, e, temperature in epot:
        if target < 250:
            continue
        key = (phase, int(min(target, 999.999) // 100) * 100)
        bins.setdefault(key, []).append(e)
    cooling = [(float(r["temperature_K"]) - float(r["target_K"])) for r in rows if r["phase"] == "cooling"]
    q = {}
    for phase in ("heating", "cooling", "final"):
        values = [1000 * (float(r["quenched_eV_atom"]) - start) for r in d["qrows"] if r["phase"] == phase]
        q[phase] = (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0, len(values))
    relax = d["meta"]["initial_relaxation"]
    info = d["meta"]["engine_info"]
    steps = d["meta"]["completed_steps"]
    return dict(relax=(relax["steps"], relax["energy_eV_atom"], relax["fmax_eV_A"]), peak=peak,
                bins={k: statistics.mean(v) for k, v in bins.items()},
                tracking=(statistics.mean(cooling), statistics.stdev(cooling)), quench=q,
                not_converged=d["quench"]["not_converged"] if d["quench"] else None,
                wall=d["meta"]["wall_s"], steps=steps, rebuilds=info["neighbour_rebuilds"],
                sizes=len(info["edge_sizes_used"]), peak_gb=info.get("peak_reserved_bytes", 0) / 2**30,
                quench_wall=d["quench"]["quench_wall_s"] if d["quench"] else None)


S = {(e, s): summary(load(e, s)) for e in ENGINES for s in SEEDS}
print("| Run | Relax steps, E (eV/atom), fmax | MD wall (s) | Peak (meV/atom, target K) | Cooling T - target (K) | "
      "Quench heating | Quench cooling | Final | Not conv. | Rebuilds | Edge sizes | Peak GB |")
print("|" + " --- |" * 12)
for (e, s), x in S.items():
    r, p, t, q = x["relax"], x["peak"], x["tracking"], x["quench"]
    print(f"| {e} {s} | {r[0]}, {r[1]:.6f}, {r[2]:.4f} | {x['wall']:.0f} | {p[2]:+.1f} ({p[0]:.0f}) | "
          f"{t[0]:+.1f} ± {t[1]:.1f} | {q['heating'][0]:+.4f} ± {q['heating'][1]:.4f} (n={q['heating'][2]}) | "
          f"{q['cooling'][0]:+.4f} ± {q['cooling'][1]:.4f} (n={q['cooling'][2]}) | {q['final'][0]:+.4f} | "
          f"{x['not_converged']} | {x['rebuilds']} | {x['sizes']} | {x['peak_gb']:.2f} |")


def max_bin_diff(a, b):
    keys = sorted(set(a["bins"]) & set(b["bins"]))
    return max(abs(a["bins"][k] - b["bins"][k]) for k in keys), len(keys)


print()
print("Per-bin mean MD potential energy (meV/atom), max |difference| over the 100 K bins of both ramps:")
for s in SEEDS:
    d, n = max_bin_diff(S[("gpu", s)], S[("fastmace", s)])
    print(f"  gpu vs fastmace, seed {s}: {d:.2f} ({n} bins)")
for e in ENGINES:
    d, n = max_bin_diff(S[(e, SEEDS[0])], S[(e, SEEDS[1])])
    print(f"  seed {SEEDS[0]} vs {SEEDS[1]}, {e}: {d:.2f} ({n} bins)")
print()
print("Bins (phase, K): " + "  ".join(f"{k[0][0]}{k[1]}" for k in sorted(S[("gpu", SEEDS[0])]["bins"])))
for (e, s), x in S.items():
    print(f"  {e:8s} {s}: " + "  ".join(f"{x['bins'][k]:6.1f}" for k in sorted(x["bins"])))
for e in ENGINES:
    print(f"run.json engine {e}: {S[(e, SEEDS[0])]['steps']} steps; checkout",
          load(e, SEEDS[0])["run"].get("fastmace", {}).get("checkout"))
