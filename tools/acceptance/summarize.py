"""Summaries and verdicts of a phase 1a acceptance run (plan task 1a-T19), for docs/validation/phase-1a.md.

    python3 tools/acceptance/summarize.py RUN_ROOT [--output results/<run id>/summary.json]

Reads the jobs' metadata.json files under RUN_ROOT (from run.py) and derives:

- speed: FastMACE's median step time over the baseline's, per A/B process and structure, whatever
  the build order; the pass rule is |ratio - 1| <= max(3 %, noise floor), with the noise floor of
  docs/validation/1a-noise-floor.md for the number of blocks (both are reported), and a row is valid
  only if nothing compiled during its timed blocks (``compile_during_timing``; conductor's condition 1
  for sharing Inductor's custom-pass slot);
- controls: setup time and peak memory per engine from the single-engine processes (reported, not
  judged);
- forces: FastMACE's starting forces in each A/B process against those of the FastMACE-only control
  process of the same configuration (decision after wave 1b);
- gate: the verdict and worst ratios per label.

Needs NumPy for the forces only; prints the tables as markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEED_LIMIT = 0.03
# docs/validation/1a-noise-floor.md: largest |B/A - 1| of two identical engines in one process.
NOISE_FLOOR = {12: 0.0016, 4: 0.0032}
FORCE_TOLERANCE = 1e-5  # eV/A: the gate's force floor (SPEC §12.3)


def noise_floor(blocks):
    """The measured noise floor for ``blocks`` timed blocks per side (the 4-block value below 12 blocks)."""
    return NOISE_FLOOR[12] if blocks >= 12 else NOISE_FLOOR[4]


def speed_verdict(ratio, blocks, limit=SPEED_LIMIT):
    """FastMACE/baseline ``ratio`` against max(limit, noise floor)."""
    floor = noise_floor(blocks)
    criterion = max(limit, floor)
    return dict(ratio=ratio, deviation=ratio - 1, limit=limit, noise_floor=floor, criterion=criterion,
                passed=bool(abs(ratio - 1) <= criterion))


def side_names(meta):
    """(baseline side, FastMACE side) of an A/B process's metadata, whatever the build order."""
    kinds = meta.get("side_kinds") or {name: name.split("_")[0] for name in meta["sides"]}
    baseline = [name for name, kind in kinds.items() if kind == "baseline"]
    fastmace = [name for name, kind in kinds.items() if kind == "fastmace"]
    if len(baseline) != 1 or len(fastmace) != 1:
        raise ValueError(f"not a FastMACE-against-baseline process: sides {kinds}")
    return baseline[0], fastmace[0]


def ab_rows(name, meta):
    """One row per structure of an A/B process: the ratio, its verdict and the per-pair ratio range."""
    baseline, fastmace = side_names(meta)
    blocks = meta["protocol"]["blocks"]
    rows = []
    for structure, entry in meta["structures"].items():
        b, f = entry["sides"][baseline], entry["sides"][fastmace]
        pairs = [y / x for x, y in zip(b["block_ms"], f["block_ms"])]
        # Option (a) of 1a-T19: a run counts only if nothing compiled during its timed blocks.
        compiled = entry.get("compile_during_timing", {})
        verdict = speed_verdict(f["median_ms"] / b["median_ms"], blocks)
        valid = bool(compiled.get("valid", False))
        verdict["passed"] = verdict["passed"] and valid
        rows.append(dict(job=name, structure=structure, natoms=entry["natoms"], ensemble=meta.get("ensemble", "nvt"),
                         model=meta["model"], pattern=meta["pattern"],
                         build_order=meta.get("build_order", meta["sides"]),
                         baseline_ms=b["median_ms"], fastmace_ms=f["median_ms"],
                         pair_ratio_min=min(pairs), pair_ratio_max=max(pairs),
                         observables_difference=entry.get("observables_difference"),
                         valid=valid, compile_changes=compiled.get("changes"),
                         custom_pass_uuids=entry.get("custom_pass_uuids"), **verdict))
    return rows


def control_rows(name, meta):
    """Setup time, load time and peak memory of a single-engine process, per structure."""
    (side,) = meta["sides"]
    load = meta.get("calculators", {}).get(side, {}).get("load_s")
    return [dict(job=name, side=meta.get("side_kinds", {}).get(side, side), model=meta["model"], structure=structure,
                 ensemble=meta.get("ensemble", "nvt"), load_s=load, setup_s=entry["sides"][side]["setup_s"],
                 peak_memory_bytes=entry["sides"][side]["peak_memory_bytes"],
                 median_ms=entry["sides"][side]["median_ms"])
            for structure, entry in meta["structures"].items()]


def compare_forces(a, b, tolerance=FORCE_TOLERANCE):
    """Largest absolute difference of two force arrays and whether it is within ``tolerance`` (eV/A)."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return dict(max_abs=None, bitwise=False, passed=False, reason=f"shapes {a.shape} and {b.shape} differ")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(max_abs=None, bitwise=False, passed=False, reason="non-finite forces")
    delta = float(np.abs(a - b).max()) if a.size else 0.0
    return dict(max_abs=delta, bitwise=bool(np.array_equal(a, b)), passed=bool(delta <= tolerance),
                tolerance=tolerance)


def forces_rows(ab_jobs, control_jobs):
    """FastMACE's starting forces in each A/B process against the FastMACE-only control of its configuration.

    ``ab_jobs`` and ``control_jobs`` map a configuration slug to [(job name, output directory)].
    """
    import numpy as np
    rows = []
    for slug, jobs in ab_jobs.items():
        controls = control_jobs.get(slug, [])
        if not controls:
            continue
        control_name, control_dir = controls[0]
        for name, directory in jobs:
            for path in sorted(Path(directory, "forces").glob("*-fastmace.npy")):
                reference = Path(control_dir, "forces", path.name)
                row = dict(job=name, control=control_name, structure=path.name.removesuffix("-fastmace.npy"))
                if reference.is_file():
                    row.update(compare_forces(np.load(path), np.load(reference)))
                else:
                    row.update(max_abs=None, bitwise=False, passed=False, reason=f"no {reference}")
                rows.append(row)
    return rows


def gate_row(name, meta):
    """The verdict of one gate run and its worst force and stress ratios over the structures."""
    summaries = meta.get("summary", {})
    worst = {}
    for quantity in ("forces", "stress"):
        ratios = [s[quantity]["ratio"] for s in summaries.values()
                  if quantity in s and s[quantity]["ratio"] is not None]
        worst[quantity] = max(ratios) if ratios else None
    return dict(job=name, model=meta.get("model"), engine=meta.get("engine"), status=meta.get("status"),
                passed=bool(meta.get("passed")) and meta.get("status") == "complete",
                structures=sorted(summaries), worst_force_ratio=worst["forces"], worst_stress_ratio=worst["stress"],
                failed_structures=sorted(k for k, s in summaries.items() if not s.get("passed")))


def summarize_run(run_root):
    """Every derived row of a run directory (the jobs' metadata.json files), by kind."""
    run_root = Path(run_root)
    out = dict(run_root=str(run_root), speed=[], controls=[], forces=[], gate=[])
    ab_jobs, control_jobs = {}, {}
    for meta_path in sorted(run_root.glob("*/metadata.json")):
        name = meta_path.parent.name
        if "." in name:
            continue  # a stopped or invalid run kept under a suffixed name (e.g. "ab-x.invalid-species-skip")
        meta = json.loads(meta_path.read_text())
        if name.startswith("ab-"):
            out["speed"] += ab_rows(name, meta)
            ab_jobs.setdefault(_slug(name), []).append((name, meta_path.parent))
        elif name.startswith("control-"):
            out["controls"] += control_rows(name, meta)
            if name.endswith("-fastmace"):
                control_jobs.setdefault(name.removeprefix("control-").removesuffix("-fastmace"), []).append(
                    (name, meta_path.parent))
        elif name.startswith("gate-"):
            out["gate"].append(gate_row(name, meta))
    out["forces"] = forces_rows(ab_jobs, control_jobs)
    out["passed"] = dict(speed=all(r["passed"] for r in out["speed"]) if out["speed"] else None,
                         forces=all(r["passed"] for r in out["forces"]) if out["forces"] else None,
                         gate=all(r["passed"] for r in out["gate"]) if out["gate"] else None,
                         gate_labels=len(out["gate"]))
    return out


def _slug(name):
    """The configuration of an A/B job name: "ab-mh1-nvt-r2" -> "mh1-nvt", "ab-mp0-medium-nvt" -> "mp0-medium-nvt"."""
    slug = name.removeprefix("ab-")  # "ab-mh1-cueq-nvt" -> "mh1-cueq-nvt"
    head, _, tail = slug.rpartition("-")
    return head if tail.startswith("r") and tail[1:].isdigit() else slug


def markdown(summary):
    """The tables of docs/validation/phase-1a.md."""
    lines = ["| Job | Model | Ensemble | Structure | Atoms | Baseline (ms) | FastMACE (ms) | FastMACE/baseline "
             "| Pair ratios | Criterion | Pass |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for r in summary["speed"]:
        lines.append(f"| {r['job']} | {r['model']} | {r['ensemble']} | {r['structure']} | {r['natoms']} "
                     f"| {r['baseline_ms']:.3f} | {r['fastmace_ms']:.3f} | {r['ratio']:.4f} "
                     f"| {r['pair_ratio_min']:.4f}–{r['pair_ratio_max']:.4f} | ±{r['criterion']:.1%} "
                     f"(noise floor {r['noise_floor']:.2%}) | {'yes' if r['passed'] else '**no**'}"
                     f"{'' if r['valid'] else ' (compiled while timed: invalid)'} |")
    lines += ["", "| Job | Engine | Model | Structure | Load (s) | Setup (s) | Peak memory (GiB) | Step (ms) |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in summary["controls"]:
        load = "–" if r["load_s"] is None else f"{r['load_s']:.1f}"
        lines.append(f"| {r['job']} | {r['side']} | {r['model']} | {r['structure']} | {load} | {r['setup_s']:.1f} "
                     f"| {r['peak_memory_bytes'] / 2**30:.2f} | {r['median_ms']:.3f} |")
    lines += ["", "| A/B job | Control | Structure | Max \\|ΔF\\| (eV/Å) | Bitwise | Pass |",
              "| --- | --- | --- | ---: | --- | --- |"]
    for r in summary["forces"]:
        delta = "–" if r["max_abs"] is None else f"{r['max_abs']:.1e}"
        lines.append(f"| {r['job']} | {r['control']} | {r['structure']} | {delta} | {'yes' if r['bitwise'] else 'no'} "
                     f"| {'yes' if r['passed'] else '**no**'} |")
    lines += ["", "| Label | Engine | Structures | Worst force ratio | Worst stress ratio | Pass |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for r in summary["gate"]:
        force = "–" if r["worst_force_ratio"] is None else f"{r['worst_force_ratio']:.2f}"
        stress = "–" if r["worst_stress_ratio"] is None else f"{r['worst_stress_ratio']:.2f}"
        lines.append(f"| {r['model']} | {r['engine']} | {', '.join(r['structures'])} | {force} | {stress} "
                     f"| {'yes' if r['passed'] else '**no**'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_root", type=Path)
    p.add_argument("--output", type=Path, help="Write the summary JSON here")
    args = p.parse_args(argv)
    summary = summarize_run(args.run_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(markdown(summary))
    return summary


if __name__ == "__main__":
    main()
