"""Summarise a ``relax_reuse.py`` run: wall, setup and loop per mode, frames per hour, and the per-frame
equality of every block against the first fresh block (the 4-T07 limits: cell lengths 1e-3 Å, volume 0.01 %,
energy 0.025 meV/atom; and the step counts). Usage: ``relax_reuse_report.py RUN_DIRECTORY``; prints markdown.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

LIMITS = {"energy_meV_atom": 0.025, "cell_length_A": 1e-3, "volume_rel": 1e-4}


def main(argv=None):
    directory = Path((argv or sys.argv[1:])[0])
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    arrays = np.load(directory / "frames.npz")
    blocks = sorted({row["block"] for row in rows})
    print("| Block | Mode | Frames | Engine builds | Wall s | Setup s | Loop s | Steps | Frames/hour |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for block in blocks:
        mine = [row for row in rows if row["block"] == block]
        wall = sum(row["wall_s"] for row in mine)
        loop = sum(row["seconds"] for row in mine)
        print(f"| {block} | {mine[0]['mode']} | {len(mine)} | {sum(bool(row['built']) for row in mine)} | "
              f"{wall:.0f} | {wall - loop:.0f} | {loop:.0f} | {sum(row['steps'] for row in mine)} | "
              f"{3600 * len(mine) / wall:.1f} |")
    print()
    fresh = [row for row in rows if row["mode"] == "fresh"]
    if fresh:
        parts = {}
        for row in fresh:
            for name, value in (row.get("setup_parts_s") or {}).items():
                parts.setdefault(name, []).append(value)
        print("Fresh setup per frame, median s: forcefield "
              f"{statistics.median(row['forcefield_s'] for row in fresh):.1f}, simulation "
              f"{statistics.median(row['simulation_s'] for row in fresh):.1f} ("
              + ", ".join(f"{name} {statistics.median(values):.1f}" for name, values in parts.items()) + ")")
        print()
    reference = blocks[0]
    first = {row["name"]: row for row in rows if row["block"] == reference}
    print(f"Against block {reference} ({first[next(iter(first))]['mode']}):")
    print()
    print("| Block | Mode | Same steps | max abs dE meV/atom | max d cell length A | max dV/V | max atom d A | "
          "within limits |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for block in blocks[1:]:
        mine = [row for row in rows if row["block"] == block]
        same = de = dl = dv = dx = 0.0
        passed = 0
        for row in mine:
            base = first[row["name"]]
            cell_a = arrays[f"{reference}/{row['name']}/cell"]
            cell_b = arrays[f"{block}/{row['name']}/cell"]
            energy = abs(row["energy_eV_atom"] - base["energy_eV_atom"]) * 1000
            lengths = np.abs(np.linalg.norm(cell_a, axis=1) - np.linalg.norm(cell_b, axis=1)).max()
            volume = abs(row["volume_A3"] / base["volume_A3"] - 1)
            atoms = np.linalg.norm(arrays[f"{reference}/{row['name']}/positions"]
                                   - arrays[f"{block}/{row['name']}/positions"], axis=1).max()
            same += row["steps"] == base["steps"]
            de, dl, dv, dx = max(de, energy), max(dl, lengths), max(dv, volume), max(dx, atoms)
            passed += (energy <= LIMITS["energy_meV_atom"] and lengths <= LIMITS["cell_length_A"]
                       and volume <= LIMITS["volume_rel"])
        print(f"| {block} | {mine[0]['mode']} | {int(same)}/{len(mine)} | {de:.2g} | {dl:.2g} | {dv:.2g} | "
              f"{dx:.2g} | {passed}/{len(mine)} |")


if __name__ == "__main__":
    main()
