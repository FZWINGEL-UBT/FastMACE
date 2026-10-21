"""Summaries and verdicts of a phase 2 acceptance run (plan task 2-T23), for docs/validation/phase-2.md.

    python3 tools/acceptance/summarize2.py RUN_ROOT --machine murphy [--output results/<run id>/summary.json]

Reads the jobs' outputs under RUN_ROOT (from run.py) and derives:

- regression (item 8): this checkout's median step time over fastmace-1b's, per A/B process and
  structure. The rule is |ratio - 1| <= that machine's own noise floor for 12 blocks (with --one-sided,
  ratio - 1 <= floor: the combined acceptance after optimisation round 1) (murphy 0.16 %,
  docs/validation/1a-noise-floor.md; RTX 5060 Ti 1.1 % NaCl and 0.4 % NbCl5,
  docs/validation/2-T12-local-noise-floor.md). A row is valid only if nothing compiled and no process-wide
  patch changed while timed, and the two plans' engine options are equal;
- forces: each side's starting forces in the A/B processes against its control process (equal plans
  must give the same forces);
- trajectories (item 8): ``tools/trajectory_versions.py compare`` on the trajectory2 jobs;
- tiers: passed, failed, errors and skipped from each tier's junit file.

Prints the tables as markdown. Needs NumPy for the forces and trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# 12-block same-process noise floors, by machine and structure (None: the machine's single figure).
NOISE_FLOORS = {"murphy": {None: 0.0016}, "local": {"NaCl": 0.011, "NbCl5": 0.004}}
FORCE_TOLERANCE = 1e-5  # eV/Å, the gate's force floor (SPEC §12.3)


def noise_floor(machine, structure):
    floors = NOISE_FLOORS[machine]
    return floors.get(structure, floors.get(None))


def regression_rows(name, meta, machine, one_sided=False):
    """One row per structure of an A/B process of tools/ab_fastmace.py --release 1b."""
    current, release = meta["current_side"], meta["release_side"]
    rows = []
    for structure, entry in meta["structures"].items():
        a, b = entry["sides"][release], entry["sides"][current]
        pairs = [y / x for x, y in zip(a["block_ms"], b["block_ms"])]
        ratio = entry["ratio"]
        floor = noise_floor(machine, structure)
        valid = bool(entry["valid"])
        rows.append(dict(job=name, structure=structure, natoms=entry["natoms"], pattern=meta["pattern"],
                         build_order=meta["build_order"], release_ms=a["median_ms"], current_ms=b["median_ms"],
                         ratio=ratio, deviation=ratio - 1, pair_ratio_min=min(pairs), pair_ratio_max=max(pairs),
                         noise_floor=floor, plan_equal=entry["plan_check"]["equal"],
                         plan_as_expected=entry["plan_check"].get("as_expected", entry["plan_check"]["equal"]),
                         plan_differences=entry["plan_check"].get("differences"),
                         plan_describe_equal=entry.get("plan_describe_equal"),
                         observables_difference=entry.get("observables_difference"),
                         one_sided=one_sided,
                         passed=bool(valid and (ratio - 1 <= floor if one_sided else abs(ratio - 1) <= floor))))
    return rows


def control_rows(name, meta):
    (side,) = meta["sides"]
    return [dict(job=name, side=side, structure=structure, load_s=meta["versions"][side]["load_s"],
                 setup_s=entry["sides"][side]["setup_s"], peak_memory_bytes=entry["sides"][side]["peak_memory_bytes"],
                 median_ms=entry["sides"][side]["median_ms"])
            for structure, entry in meta["structures"].items()]


def forces_rows(run_root, ab_names, control_names):
    """Each side's starting forces in each A/B process against its control's, per structure."""
    import numpy as np
    controls = {}
    for name in control_names:
        meta = json.loads((run_root / name / "metadata.json").read_text())
        (side,) = meta["sides"]
        controls[side] = run_root / name / "forces"
    rows = []
    for name in ab_names:
        meta = json.loads((run_root / name / "metadata.json").read_text())
        for structure in meta["structures"]:
            for side in meta["sides"]:
                if side not in controls:
                    continue
                a = np.load(run_root / name / "forces" / f"{structure}-{side}.npy")
                b = np.load(controls[side] / f"{structure}-{side}.npy")
                worst = float(np.abs(a - b).max())
                rows.append(dict(job=name, structure=structure, side=side, max_abs_eV_A=worst,
                                 passed=bool(worst <= FORCE_TOLERANCE)))
        if len(meta["sides"]) == 2:
            for structure in meta["structures"]:
                a, b = (np.load(run_root / name / "forces" / f"{structure}-{side}.npy") for side in meta["sides"])
                rows.append(dict(job=name, structure=structure, side="between versions",
                                 max_abs_eV_A=float(np.abs(a - b).max()), passed=None))
    return rows


def junit_totals(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors",
                                                                               "skipped")}
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["failed_ids"] = [f"{case.get('classname')}::{case.get('name')}" for case in root.iter("testcase")
                            if case.find("failure") is not None or case.find("error") is not None]
    totals["time_s"] = sum(float(suite.get("time", 0)) for suite in suites)
    return totals


def summarize_run(run_root, machine, allowed=(), one_sided=False):
    run_root = Path(run_root)
    summary = dict(run_root=str(run_root), machine=machine, regression=[], controls=[], forces=[], trajectories={},
                   tiers={})
    ab, controls = [], []
    for directory in sorted(run_root.glob("regression2-*")):
        path = directory / "metadata.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text())
        if meta.get("status") != "complete":
            summary["regression"].append(dict(job=directory.name, status=meta.get("status"), error=meta.get("error")))
            continue
        if len(meta["sides"]) == 2:
            summary["regression"] += regression_rows(directory.name, meta, machine, one_sided)
            ab.append(directory.name)
        else:
            summary["controls"] += control_rows(directory.name, meta)
            controls.append(directory.name)
    if ab and controls:
        summary["forces"] = forces_rows(run_root, ab, controls)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for backend in ("cueq", "hybrid"):
        release, current = run_root / f"trajectory2-{backend}-1b", run_root / f"trajectory2-{backend}-2"
        if (release / "trajectory.json").is_file() and (current / "trajectory.json").is_file():
            import trajectory_versions
            repeat = run_root / f"trajectory2-{backend}-2-repeat"
            try:
                result = trajectory_versions.compare(release, current,
                                                     repeat if (repeat / "trajectory.json").is_file() else None,
                                                     allowed)
                result.pop("statistics", None)
            except ValueError as exc:
                result = dict(error=str(exc), passed=False)
            summary["trajectories"][backend] = result
    for tier in ("acceptance-tier", "full-tier"):
        path = run_root / tier / "junit.xml"
        if path.is_file():
            summary["tiers"][tier] = junit_totals(path)
    return summary


def markdown(summary):
    lines = []
    if summary["regression"]:
        lines += ["| Process | Structure | 1b (ms) | 2 (ms) | 2/1b | pair range | floor | plan ok | valid | pass |",
                  "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |"]
        for row in summary["regression"]:
            if "ratio" not in row:
                lines.append(f"| {row['job']} | {row.get('status')} | | | | | | | | |")
                continue
            lines.append(f"| {row['job']} | {row['structure']} | {row['release_ms']:.3f} | {row['current_ms']:.3f} | "
                         f"{row['ratio']:.4f} | {row['pair_ratio_min']:.4f}–{row['pair_ratio_max']:.4f} | "
                         f"{row['noise_floor'] * 100:.2f} % | {row['plan_as_expected']} | {row['valid']} | "
                         f"{'yes' if row['passed'] else 'NO'} |")
    if summary["controls"]:
        lines += ["", "| Control | Side | Structure | load (s) | setup (s) | median (ms) |",
                  "| --- | --- | --- | ---: | ---: | ---: |"]
        lines += [f"| {r['job']} | {r['side']} | {r['structure']} | {r['load_s']:.1f} | {r['setup_s']:.1f} | "
                  f"{r['median_ms']:.3f} |" for r in summary["controls"]]
    if summary["forces"]:
        lines += ["", "| Forces | Structure | Side | max abs (eV/Å) | pass |", "| --- | --- | --- | ---: | --- |"]
        lines += [f"| {r['job']} | {r['structure']} | {r['side']} | {r['max_abs_eV_A']:.2e} | {r['passed']} |"
                  for r in summary["forces"]]
    for backend, result in summary["trajectories"].items():
        lines += ["", f"Trajectories, {backend}: {json.dumps(result, default=str)}"]
    for tier, totals in summary["tiers"].items():
        lines += ["", f"{tier}: {totals['passed']} passed, {totals['failures']} failed, {totals['errors']} errors, "
                      f"{totals['skipped']} skipped, {totals['time_s'] / 60:.1f} min; failed: {totals['failed_ids']}"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_root", type=Path)
    p.add_argument("--machine", required=True, choices=sorted(NOISE_FLOORS))
    p.add_argument("--output", type=Path, help="Write the summary JSON here")
    p.add_argument("--one-sided", action="store_true",
                   help="Step times: only slower than the floor fails (the combined acceptance, where main is "
                        "expected to be faster than the release)")
    p.add_argument("--allow-option-changes", nargs="*", default=[], metavar="OPTION",
                   help="Plan options whose difference between the versions is expected (the trajectory checks)")
    args = p.parse_args(argv)
    summary = summarize_run(args.run_root, args.machine, args.allow_option_changes, args.one_sided)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(markdown(summary))
    return summary


if __name__ == "__main__":
    main()
