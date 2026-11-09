"""Where the tiled capacity run's host memory goes: ``tiled_capacity.py`` with its arguments, under an RSS sampler.

Each wrapped function (the build's steps and the tiled call's host-heavy steps) records the process's resident set
at entry, its peak while inside (a thread samples ``/proc/self/statm`` every few milliseconds) and at exit, nested
calls charged to the innermost one as well as their callers. The table (``host-profile.json`` beside the
``tiled_capacity.py`` record) lists, per function, the calls, the largest rise of the peak over the entry and the
largest net growth, in bytes and in bytes per atom. Usage: ``tiled_host_profile.py <tiled_capacity.py arguments>``.
Linux only.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiled_capacity  # noqa: E402

PAGE = os.sysconf("SC_PAGE_SIZE")
_peak = [0]
_stack = []
table = {}


def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE


def sampler(stop):
    while not stop.is_set():
        now = rss()
        if now > _peak[0]:
            _peak[0] = now
        for frame in _stack:
            if now > frame[1]:
                frame[1] = now
        time.sleep(0.005)


def wrap(owner, name, label):
    fn = owner.__dict__.get(name)
    if fn is None:
        return
    raw = fn.__func__ if isinstance(fn, (staticmethod, classmethod)) else fn

    @functools.wraps(raw)
    def wrapped(*args, **kwargs):
        start = rss()
        frame = [start, start]
        _stack.append(frame)
        try:
            return raw(*args, **kwargs)
        finally:
            _stack.remove(frame)
            end = rss()
            entry = table.setdefault(label, dict(calls=0, peak_rise=0, net=0, peak=0))
            entry["calls"] += 1
            entry["peak_rise"] = max(entry["peak_rise"], max(frame[1], end) - start)
            entry["net"] = max(entry["net"], end - start)
            entry["peak"] = max(entry["peak"], frame[1])

    if isinstance(fn, staticmethod):
        setattr(owner, name, staticmethod(wrapped))
    elif isinstance(fn, classmethod):
        setattr(owner, name, classmethod(wrapped))
    else:
        setattr(owner, name, wrapped)


def install():
    import fastmace as fm
    from fastmace.model import forces, tiled
    from fastmace.planner import minimal

    wrap(forces, "host_batch", "forces.host_batch")
    wrap(forces.MACEForceField, "__init__", "MACEForceField.__init__")
    for name in ("rebuild", "compact", "release_node_attrs", "release_whole_system", "_prepare_one_off"):
        wrap(forces.MACEForceField, name, f"MACEForceField.{name}")
    wrap(fm.ForceField, "__init__", "fm.ForceField.__init__")
    planner = minimal.MinimalPlanner
    for name in ("plan_evaluation", "build_evaluation"):
        wrap(planner, name, f"MinimalPlanner.{name}")
    wrap(tiled_capacity, "supercell", "tiled_capacity.supercell")
    wrap(tiled.TiledForceField, "__init__", "TiledForceField.__init__")
    for name in ("_call", "_wavefront", "_slot_keys", "_keep_lists", "_globals", "_plan", "_tile_graph", "_piece"):
        wrap(tiled.TiledForceField, name, f"TiledForceField.{name}")
    wrap(tiled, "plan_tiles", "tiled.plan_tiles")
    for name in ("allocate", "plan", "pool", "touched"):
        wrap(tiled._State, name, f"_State.{name}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    install()
    stop = threading.Event()
    thread = threading.Thread(target=sampler, args=(stop,), daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        tiled_capacity.main(argv)
    finally:
        stop.set()
        thread.join()
        out = Path(argv[argv.index("--output") + 1])
        records = sorted(out.glob("tiled-*.json"), key=lambda p: p.stat().st_mtime)
        atoms = json.loads(records[-1].read_text()).get("atoms") if records else None
        rows = {label: dict(entry, peak_rise_per_atom=entry["peak_rise"] / atoms if atoms else None,
                            net_per_atom=entry["net"] / atoms if atoms else None)
                for label, entry in sorted(table.items(), key=lambda kv: -kv[1]["peak_rise"])}
        report = dict(atoms=atoms, process_peak=_peak[0], process_peak_per_atom=_peak[0] / atoms if atoms else None,
                      seconds=time.perf_counter() - started, functions=rows)
        (out / "host-profile.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(dict(atoms=atoms, process_peak=_peak[0],
                              top={k: (v["peak_rise"], round(v["peak_rise_per_atom"] or 0)) for k, v in
                                   list(rows.items())[:12]})))


if __name__ == "__main__":
    main()
