"""How much of a padded tiled call is padding: each tile's real receivers, halo and edges against its slot's.

One padded call (``mode="padded"``, no compilation: the same slots as the compiled modes) at a given receiver cap R
(``--receivers``, no budget), tile lists, the host state aliased, the species index. ``_slot_keys`` is wrapped:
each tile's (R, H, E) as the slots see it, and the slot it lands in. Reported per slot and in total: the tiles, the
real and padded node rows (R + H, the rows interaction 0 runs on) and edges, and the remainder pieces (a block's
last piece below the cap, when the block holds more than R atoms).

Run as ``run.sh NAME results/tiled-pieces-20261102/padding.py --repeat 35x35x105 --receivers 16384``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import tiled_capacity  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import dataclasses

    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                      dynamic_edges=False))
        field = planner.build_evaluation(plan, user, atoms, call_rows=args.receivers, release_node_attrs=True)
        tiled = TiledForceField(field, TileConfig(args.receivers, mode="padded", neighbours="tiles", state="host",
                                                  alias_state=True, species="index", budget_receivers=()))
        seen = {}
        original_keys, original_graph = TiledForceField._slot_keys, TiledForceField._tile_graph

        def slot_keys(self, tiles, source, local_of, n):
            sizes = []

            def graph(*a, **k):
                piece = original_graph(self, *a, **k)
                sizes.append((int(piece[4]), int(piece[5]), int(piece[1].shape[1])))
                return piece

            self._tile_graph = graph
            try:
                keys = original_keys(self, tiles, source, local_of, n)
            finally:
                del self._tile_graph
            seen.update(sizes=sizes, keys=keys)
            return keys

        TiledForceField._slot_keys = slot_keys
        positions = field._prepare_one_off(atoms)
        tiled(positions)
        TiledForceField._slot_keys = original_keys
    sizes, keys, slots = seen["sizes"], seen["keys"], tiled._slots
    per = collections.defaultdict(lambda: collections.Counter())
    for (r, h, e), key in zip(sizes, keys):
        rows, halo, edges = slots[key]["sizes"]
        c = per[key]
        c["tiles"] += 1
        c["real_rows"] += r + h
        c["padded_rows"] += rows + halo  # the slot's receiver rows (with the padding receiver) and halo
        c["real_receivers"] += r
        c["real_edges"] += e
        c["padded_edges"] += edges
        c["remainder"] += r < args.receivers // 2  # a block's last piece (blocks hold about R atoms)
    total = sum(per.values(), collections.Counter())
    record = dict(structure=args.structure, atoms=n, receivers=args.receivers, device=torch.cuda.get_device_name(),
                  host=os.uname().nodename, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  grid=tiled.last.get("grid"), tiles=len(sizes),
                  slots={str(k): dict(sizes=list(slots[k]["sizes"]), **per[k]) for k in sorted(per)},
                  total=dict(total),
                  padding=dict(rows=1 - total["real_rows"] / total["padded_rows"],
                               edges=1 - total["real_edges"] / total["padded_edges"],
                               halo_share=1 - total["real_receivers"] / total["real_rows"],
                               remainder_tiles=total["remainder"] / len(sizes),
                               remainder_edges=sum(slots[k]["sizes"][2] for (r, _, _), k in zip(sizes, keys)
                                                   if r < args.receivers // 2) / total["padded_edges"]),
                  tile_sizes=sizes, tile_keys=keys)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"padding-{args.structure}-{n}-R{args.receivers}.json").write_text(json.dumps(record))
    print(json.dumps({k: v for k, v in record.items() if k not in ("tile_sizes", "tile_keys")}))


if __name__ == "__main__":
    main()
