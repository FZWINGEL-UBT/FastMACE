"""Tables of the consistency probes: analyse_probe.py DIR (probe-a/b/c.json) -> markdown on stdout, summary.json.

Directional derivatives: per step h the mismatch e = CD - (-F.d) of each direction, as
  rel_Fd  |e| / |F.d|      (the specified measure; large where F.d happens to be near zero)
  rel_F   |e| / |F|        (|F| the 2-norm over all 3N components, the bound of |F.d| for a unit d)
  bias    mean(e) / |F|    (a systematic part survives the average over directions; noise does not)
Also the paired difference between configurations: e_a - e_c and e_b - e_c of the same direction and step (the
float32 noise of E is the same function of the positions only within one configuration, so this is not noise-free,
but a consistent sign shows a systematic gradient error)."""

import json
import sys
from pathlib import Path

import numpy as np


def load(directory, config):
    path = Path(directory) / f"probe-{config}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def derivative_table(records, repeat):
    out, lines = {}, []
    configs = [c for c in "abc" if records.get(c) and repeat in records[c]["structures"]]
    if not configs:
        return lines, out
    steps = list(records[configs[0]]["structures"][repeat]["derivative"]["rows"][0]["central"])
    lines.append(f"\n**SrTiO3 {repeat}** (mismatch e = CD - (-F.d); median / max over directions)\n")
    lines.append("| h (A) | " + " | ".join(f"{c}: med rel_F | {c}: max rel_F | {c}: med rel_Fd | {c}: bias/|F|"
                                          for c in configs) + " |")
    lines.append("|" + " --- |" * (1 + 4 * len(configs)))
    for h in steps:
        cells = []
        for c in configs:
            entry = records[c]["structures"][repeat]
            fnorm = entry["derivative"]["force_rms_eV_per_A"] * np.sqrt(3 * entry["natoms"])
            e = np.array([r["central"][h]["mismatch"] for r in entry["derivative"]["rows"]])
            fd = np.array([abs(r["minus_F_dot_d"]) for r in entry["derivative"]["rows"]])
            stats = dict(med_rel_F=float(np.median(abs(e)) / fnorm), max_rel_F=float(abs(e).max() / fnorm),
                         med_rel_Fd=float(np.median(abs(e) / fd)), bias_rel_F=float(e.mean() / fnorm),
                         bias_se_rel_F=float(e.std(ddof=1) / np.sqrt(len(e)) / fnorm), F_norm=float(fnorm))
            out.setdefault(repeat, {}).setdefault(h, {})[c] = stats
            cells.append(f"{stats['med_rel_F']:.2e} | {stats['max_rel_F']:.2e} | {stats['med_rel_Fd']:.2e} | "
                         f"{stats['bias_rel_F']:+.1e} ± {stats['bias_se_rel_F']:.0e}")
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    return lines, out


def main():
    directory = Path(sys.argv[1])
    records = {c: load(directory, c) for c in "abc"}
    lines, summary = ["## Directional derivatives"], dict(derivative={}, knots={}, loop={})
    repeats = []
    for c in "abc":
        if records[c]:
            repeats += [r for r in records[c]["structures"] if r not in repeats]
    for repeat in repeats:
        more, stats = derivative_table(records, repeat)
        lines += more
        summary["derivative"].update(stats)
    first = repeats[0]
    lines += ["\n## Knot crossings and closed loop (SrTiO3 " + first + ")\n",
              "| config | crossings | max jump about trend (eV) | rms jump (eV) | dE (eV) | -int F.dR (eV) | "
              "dE - work (eV) | max abs(E - cumulative work) (eV) | loop work (eV) | sum abs segment work (eV) |",
              "|" + " --- |" * 10]
    for c in "abc":
        entry = (records[c] or {}).get("structures", {}).get(first, {})
        if "knots" not in entry:
            continue
        k, lp = entry["knots"], entry.get("loop", {})
        summary["knots"][c] = {key: v for key, v in k.items() if not isinstance(v, list)}
        summary["loop"][c] = {key: v for key, v in lp.items() if not isinstance(v, list)}
        lines.append(f"| {c} | {entry['line_choice']['crossings']} | {k['max_jump_eV']:.2e} | {k['rms_jump_eV']:.2e} | "
                     f"{k['delta_E_eV']:.6f} | {k['work_eV']:.6f} | {k['delta_E_minus_work_eV']:+.2e} | "
                     f"{k['max_abs_E_minus_cumulative_work_eV']:.2e} | {lp.get('loop_work_eV', float('nan')):+.2e} | "
                     f"{lp.get('sum_abs_segment_work_eV', float('nan')):.3e} |")
    (directory / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
