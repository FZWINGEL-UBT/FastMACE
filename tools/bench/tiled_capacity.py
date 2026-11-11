"""Tiled capacity path: memory falsifier and accuracy check (docs/design/tiled-backend.md, steps 2-3).

``point``: one supercell in this process. It builds the one-off evaluation plan of ``fm.ForceField`` for the
cell (the default rewrite set, float32), then the same plan eager (``compile_mode="none"``), and wraps the eager
force field in ``fastmace.model.tiled.TiledForceField`` with ``--receivers``. It records:

- the neighbour build (rebuild, compact) and its peak, apart from the force call;
- two tiled force calls (the first one warms the kernels up), each with its peak allocated memory above the
  memory allocated before it, its time, and the global buffers it holds (h1 and its adjoint, forces);
- with ``--compare eager``, the whole-system eager call of the same force field (the same modules on one graph);
  with ``--compare compiled``, the default compiled evaluation plan (``fm.ForceField.evaluate``'s path) built in
  the same process after the tiled calls. Energy per atom, the largest force-component and stress-component
  differences, and the whole call's peak.

``--rattle A`` displaces every atom by a uniform random vector of up to A Angstrom per component (seed 0) and
``--strain S`` applies a random symmetric strain of up to S, so the comparison is off equilibrium.

Writes ``<output>/tiled-<name>-<atoms>-R<receivers><tag>.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_scan  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--receivers", type=int, default=8192)
    p.add_argument("--edges", type=int, default=None)
    p.add_argument("--mode", choices=("eager", "padded", "compiled", "auto"), default="eager",
                   help="TileConfig.mode; 'compiled' and 'auto' build the default compiled evaluation plan (its "
                        "traceable model)")
    p.add_argument("--compare", choices=("none", "eager", "compiled"), default="none")
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--strain", type=float, default=0.0)
    p.add_argument("--calls", type=int, default=2, help="tiled force calls (the first one warms up)")
    p.add_argument("--host-state", action="store_true",
                   help="TileConfig.state='host': x, sc and their adjoints in pinned host memory")
    p.add_argument("--alias-state", action="store_true",
                   help="TileConfig.alias_state (with --host-state): dE/dsc in sc's rows, dE/dx in x's rows")
    p.add_argument("--state-dtype", choices=("auto", "none", "bfloat16", "float16"), default="auto",
                   help="TileConfig.state_dtype (with --alias-state): x, sc and their aliased adjoints stored on the "
                        "host in 16 bits; auto (TileConfig's default): float16 only when the state would not fit "
                        "above the floor even without the kept lists; none: the call's dtype")
    p.add_argument("--species-index", action="store_true",
                   help="TileConfig.species='index': the tiles' one-hot rows from the species index; the field's "
                        "device one-hot is released")
    p.add_argument("--budget-receivers", type=int, nargs="*", default=[],
                   help="TileConfig.budget_receivers: larger caps the call may take after its first call (none "
                        "given: TileConfig's default)")
    p.add_argument("--no-budget", action="store_true",
                   help="budget_receivers=(): R stays --receivers (several compiled tiles at small sizes)")
    p.add_argument("--wavefront", choices=("auto", "on", "off"), default="auto",
                   help="TileConfig.wavefront (with --alias-state): the tile sweeps interleaved, x and sc in a row "
                        "pool")
    p.add_argument("--lanes", choices=("auto", "on", "off"), default="auto",
                   help="TileConfig.lanes: sweep 2's tiles in pairs on two green-context SM halves (needs "
                        "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync)")
    p.add_argument("--wavefront-group", type=int, default=None,
                   help="TileConfig.wavefront_group: sweep-2 tiles per run of the wavefront (none: the default)")
    p.add_argument("--boundary-skin", choices=("on", "off"), default="on",
                   help="TileConfig.boundary_skin: the wavefront plan's first and last layer split at the reach")
    p.add_argument("--compiled-graphs", nargs="?", const="on", choices=("auto", "on", "off"), default="auto",
                   help="TileConfig.compiled_graphs: compiled tile bodies captured in CUDA graphs (auto: None, on for "
                        "COMPILED_GRAPH_GPUS where the memory guard fits)")
    p.add_argument("--prefetch", choices=("on", "off"), default="on",
                   help="TileConfig.prefetch: the next tile's graph and host rows prepared on a side stream")
    p.add_argument("--backward-chunk-scale", type=int, default=None,
                   help="TileConfig.backward_chunk_scale: the chunked streaming backward's edges per chunk in the "
                        "tiles times this (none: by device name)")
    p.add_argument("--grid-fit", choices=("on", "off"), default="off",
                   help="TileConfig.grid_fit: the plan's grid with the fewest receiver-cap pieces (off: the rule's)")
    p.add_argument("--list-reuse", choices=("on", "off"), default="on",
                   help="TileConfig.list_reuse: the tiles' lists within cutoff + skin kept on the host across calls")
    p.add_argument("--list-cache", choices=("on", "off"), default="on",
                   help="TileConfig.list_cache: a tile's lists kept on the host within the call (with kept lists: "
                        "its filtered ones in the wavefront's band)")
    p.add_argument("--backward-single-stream", default=None, metavar="FRACTION|off",
                   help="this GPU's entry in oeq_streaming SINGLE_STREAM_DEVICES: the chunked backward on one slot "
                        "and the current stream where a chunk's slot is at least this fraction of the L2; off: two "
                        "slots (default: the table's entry for the device name)")
    p.add_argument("--sweep-profile", action="store_true",
                   help="the last call with torch.profiler around three tiles of each sweep: kernel classes and top "
                        "kernels per sweep (tiled_corun.sweep_profile)")
    p.add_argument("--sweep-times", action="store_true",
                   help="the last call with CUDA events around each tile's sweep body, fill and scatter: device "
                        "seconds per sweep (tiled_corun.sweep_times)")
    p.add_argument("--profile-last", action="store_true",
                   help="the last call under torch.profiler: kernel classes, the device's busy fraction and the "
                        "host operations with the most self time (tiled_corun.profile_call)")
    p.add_argument("--tag", default="")
    p.add_argument("--ledger", action="store_true",
                   help="after the build and after every call, record every live tensor and NumPy array by device, "
                        "dtype and shape, named where reachable from the user's force field, the evaluation field "
                        "or the tiled call (ledgers in the record)")
    p.add_argument("--memory-history", action="store_true",
                   help="record the CUDA allocator's trace during the build and keep the allocations live at the "
                        "build's device peak, by the source line that made them (build_peak in the record)")
    p.add_argument("--host-positions", action="store_true",
                   help="keep the call's float64 positions on the host (--tile-list; TILE_HOST_ROWS: 24 bytes per "
                        "atom off the device)")
    p.add_argument("--first-call-probe", choices=("default", "on", "off"), default="default",
                   help="tiled.TILE_FIRST_CALL_PROBE: mode auto's measuring first call on its largest tiles only "
                        "(the probe's report in the first call's record, 'probe')")
    p.add_argument("--call-history", type=int, default=None, metavar="CALL",
                   help="record the CUDA allocator's trace during tiled call CALL (0-based) and keep the allocations "
                        "made in it that are live at its peak, by source line (call_peak in the record; the peak "
                        "above what was allocated before the call)")
    p.add_argument("--history-entries", type=int, default=4_000_000,
                   help="the trace's ring buffer for --call-history (call_peak reports whether it overflowed)")
    p.add_argument("--tile-list", action="store_true",
                   help="per-tile neighbour lists (TileConfig.neighbours='tiles'); the force field's own list is "
                        "left empty, so it costs no memory")
    p.add_argument("--neighbours", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="NeighbourConfig overrides of the plan, e.g. gpu_neighbour_list=off (the host build: O(N) "
                        "against the GPU screen's O(N^2) at millions of atoms)")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.species_index and args.compare == "eager":
        p.error("--species-index releases the field's one-hot, which --compare eager's whole-system call needs")
    return args


# Cell copies per chunk of ``supercell`` (about 6 MB of float64 positions per chunk at five atoms per copy).
SUPERCELL_CHUNK_COPIES = 1 << 17


def supercell_ase(structure, repeat, rattle, strain):
    """The ASE form of ``supercell`` (``Atoms.repeat``, ``set_cell(scale_atoms=True)``, ``wrap``): the reference
    of its tests. It loops in Python over every cell copy and wraps through whole-system temporaries."""
    import numpy as np

    _, cell = capacity_scan.base_cell(structure)
    atoms = cell.repeat(repeat)
    rng = np.random.default_rng(0)
    if strain:
        e = rng.uniform(-strain, strain, (3, 3))
        e = 0.5 * (e + e.T)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + e), scale_atoms=True)
    if rattle:
        atoms.positions = atoms.positions + rng.uniform(-rattle, rattle, atoms.positions.shape)
    atoms.wrap()
    return atoms


def supercell(structure, repeat, rattle, strain, chunk=SUPERCELL_CHUNK_COPIES):
    """``supercell_ase``'s atoms, built a chunk of cell copies at a time: the float64 positions are allocated once
    and each chunk is placed (ASE's copy order, ``m2`` fastest), strained, rattled (the same seed-0 stream, drawn
    in atom order) and wrapped (``ase.geometry.wrap_positions``) before the next. No Python loop over the copies
    (about 2e8 at 585^3) and no whole-system temporaries: the 519M-atom segmentation fault of 9 November 2026 came
    out of ASE's per-copy loop (plans/billion-atoms-20261111.md, "Before step 4")."""
    import numpy as np
    from ase import Atoms
    from ase.cell import Cell
    from ase.geometry import wrap_positions

    _, unit = capacity_scan.base_cell(structure)
    if unit.constraints:
        return supercell_ase(structure, repeat, rattle, strain)
    m = (repeat,) * 3 if isinstance(repeat, int) else tuple(int(x) for x in repeat)
    base_cell = unit.cell.array.copy()
    for x, vec in zip(m, base_cell):
        if x != 1 and not vec.any():
            raise ValueError("Cannot repeat along undefined lattice vector")
    copies = int(np.prod(m))
    n = len(unit)
    final = Cell(np.array([m[c] * base_cell[c] for c in range(3)]))
    rng = np.random.default_rng(0)
    scale = None
    if strain:
        e = rng.uniform(-strain, strain, (3, 3))
        e = 0.5 * (e + e.T)
        strained = Cell(final.array @ (np.eye(3) + e))
        scale = np.linalg.solve(final.complete(), strained.complete())
        final = strained
    base = unit.arrays["positions"]
    positions = np.empty((copies * n, 3), dtype=np.float64)
    for q0 in range(0, copies, chunk):
        q1 = min(copies, q0 + chunk)
        index = np.stack(np.unravel_index(np.arange(q0, q1), m), axis=1)
        offsets = np.dot(index, base_cell)  # ASE: np.dot((m0, m1, m2), cell) per copy
        block = (base[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        if scale is not None:
            block = np.dot(block, scale)
        if rattle:
            block = block + rng.uniform(-rattle, rattle, block.shape)
        positions[q0 * n:q1 * n] = wrap_positions(block, final, pbc=unit.pbc)
    atoms = Atoms(cell=final, pbc=unit.pbc)
    atoms.info = dict(unit.info)
    for name, a in unit.arrays.items():
        if name != "positions":
            atoms.arrays[name] = np.tile(a, (copies,) + (1,) * (a.ndim - 1))
    atoms.arrays["positions"] = positions
    return atoms


@contextlib.contextmanager
def no_global_list(active):
    """With ``active``, ``SkinNeighbourList.build`` and ``compact`` leave the force field's list empty (one slot,
    no pairs): the tiled call's per-tile lists replace it, and its build would be the largest memory term."""
    if not active:
        yield
        return
    from fastmace.neighbors.skin import SkinNeighbourList

    build, compact = SkinNeighbourList.build, SkinNeighbourList.compact

    def empty_build(self, positions):
        grew = self.capacity == 0
        if grew:
            self._allocate(1, exact=True)
        self.num_candidates = 0
        return grew

    def empty_compact(self, positions):
        self.count.zero_()

    SkinNeighbourList.build, SkinNeighbourList.compact = empty_build, empty_compact
    try:
        yield
    finally:
        SkinNeighbourList.build, SkinNeighbourList.compact = build, compact


def host_memory():
    """The process's host memory now and at its peak so far (``VmRSS``, ``VmHWM``; pinned pages included), in
    bytes; empty where /proc is missing."""
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return {}
    values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in lines if line.startswith(("VmRSS", "VmHWM"))}
    return dict(host_rss=values.get("VmRSS"), host_peak=values.get("VmHWM"))


def memory_ledger(roots, depth=4):
    """Every live torch tensor and NumPy array (by storage, counted once), grouped by device, dtype and shape, with
    the attribute paths from ``roots`` ({name: object}) that reach it, largest first."""
    import gc

    import numpy as np
    import torch

    names = {}

    def key(value):
        if isinstance(value, torch.Tensor):
            return ("torch", value.untyped_storage().data_ptr()) if value.numel() else None
        if isinstance(value, np.ndarray):
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            return ("numpy", base.__array_interface__["data"][0]) if value.size else None
        return None

    seen = set()

    def walk(value, path, level):
        k = key(value)
        if k is not None:
            names.setdefault(k, []).append(path)
            return
        if level > depth or id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        seen.add(id(value))
        if isinstance(value, dict):
            items = list(value.items())[:64]
        elif isinstance(value, (list, tuple)):
            items = list(enumerate(value))[:64]
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            items = list(vars(value).items())
            if isinstance(value, torch.nn.Module):
                items += list(value.named_buffers(recurse=False)) + list(value.named_parameters(recurse=False))
        else:
            return
        for name, item in items:
            walk(item, f"{path}.{name}", level + 1)

    for name, root in roots.items():
        walk(root, name, 0)
    groups = {}
    counted = set()
    for obj in gc.get_objects():
        try:
            k = key(obj)
        except Exception:  # noqa: BLE001  (a tensor subclass or a freed storage)
            continue
        if k is None or k in counted:
            continue
        counted.add(k)
        if isinstance(obj, torch.Tensor):
            nbytes = obj.untyped_storage().nbytes()
            where, dtype, shape = str(obj.device), str(obj.dtype), list(obj.shape)
        else:
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            nbytes, where, dtype, shape = base.nbytes, "cpu (numpy)", str(obj.dtype), list(obj.shape)
        group = groups.setdefault((where, dtype, str(shape)), dict(device=where, dtype=dtype, shape=shape, count=0,
                                                                     bytes=0, names=[]))
        group["count"] += 1
        group["bytes"] += nbytes
        group["names"] = (group["names"] + names.get(k, [])[:3])[:6]
        if not names.get(k) and nbytes >= 2**20 and "owners" not in group:
            group["owners"] = owners(obj)
    return sorted(groups.values(), key=lambda g: -g["bytes"])


def peak_live_set(snapshot, top=30):
    """The device allocations live at the trace's peak, grouped by the first FastMACE or MACE source line of their
    stack (else the first frame), largest first: ``{peak, groups: [{site, count, bytes}]}``."""
    live, total, peak, at_peak = {}, 0, 0, {}
    for trace in snapshot.get("device_traces", []):
        for entry in trace:
            action, address, size = entry.get("action"), entry.get("addr"), int(entry.get("size", 0))
            if action == "alloc":
                live[address] = (size, entry.get("frames") or [])
                total += size
                if total > peak:
                    peak, at_peak = total, dict(live)
            elif action in ("free_completed",) and address in live:
                total -= live.pop(address)[0]

    def site(frames):
        for frame in frames:
            name = frame.get("filename", "")
            if ("fastmace" in name or "/mace/" in name) and "torch" not in name:
                short = name.split("site-packages/")[-1].split("src-all-")[-1]
                return f"{short}:{frame.get('line')} {frame.get('name')}"
        return f"{frames[0].get('filename')}:{frames[0].get('line')}" if frames else "?"

    groups = {}
    for size, frames in at_peak.values():
        group = groups.setdefault(site(frames), dict(site=site(frames), count=0, bytes=0))
        group["count"] += 1
        group["bytes"] += size
    return dict(peak=peak, groups=sorted(groups.values(), key=lambda g: -g["bytes"])[:top])


def owners(obj, levels=3):
    """What holds an unnamed array: the types (and dict keys) of its referrers, ``levels`` deep."""
    import gc
    import inspect

    chain, frontier = [], [obj]
    for _ in range(levels):
        found = []
        for item in frontier:
            for ref in gc.get_referrers(item):
                if ref is frontier or ref is chain or inspect.isframe(ref):
                    continue
                label = type(ref).__qualname__
                if isinstance(ref, dict):
                    label += "[" + ",".join(str(key) for key, value in ref.items() if value is item)[:80] + "]"
                elif hasattr(ref, "__name__"):
                    label += f" {ref.__name__}"
                found.append((label, ref))
        chain.append([label for label, _ in found][:6])
        frontier = [ref for _, ref in found][:6]
    return chain


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    if args.backward_single_stream is not None:
        import torch

        from fastmace.kernels.oeq_streaming import runtime as oeq_runtime
        name = torch.cuda.get_device_name()
        oeq_runtime.SINGLE_STREAM_DEVICES = {k: v for k, v in oeq_runtime.SINGLE_STREAM_DEVICES.items() if k != name}
        if args.backward_single_stream != "off":
            oeq_runtime.SINGLE_STREAM_DEVICES[name] = float(args.backward_single_stream)
    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = supercell(args.structure, repeat, args.rattle, args.strain)
    n = len(atoms)
    record = dict(structure=args.structure, repeat=list(repeat), atoms=n, receivers=args.receivers,
                  edges=args.edges, rattle=args.rattle, strain=args.strain, model=args.model,
                  backward_single_stream=args.backward_single_stream,
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), phases={})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"tiled-{args.structure}-{n}-R{args.receivers}{args.tag}.json"

    def save():
        path.write_text(json.dumps(record, indent=1, default=str))

    def mem(before=0):
        torch.cuda.synchronize()
        return dict(allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()),
                    peak_above_before=int(torch.cuda.max_memory_allocated() - before), **host_memory())

    lists = contextlib.ExitStack()  # (the force field's own list: global_list=False with --tile-list)
    try:
        record["phase"] = "build"
        if args.memory_history:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")
        started = time.perf_counter()
        user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
        record["phases"]["user"] = mem()
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        record["phases"]["plan"] = mem()
        eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                       dynamic_edges=False))
        if args.neighbours:
            overrides = {key: capacity_scan._value(value) for key, value in
                         (item.split("=", 1) for item in args.neighbours)}
            record["neighbours"] = overrides
            plan = dataclasses.replace(plan, neighbours=dataclasses.replace(plan.neighbours, **overrides))
            eager = dataclasses.replace(eager, neighbours=dataclasses.replace(eager.neighbours, **overrides))
        # The species tables at the tiles' row count (the tiled call binds its own per tile): the construction
        # never evaluates N rows. An eager comparison keeps N, since it calls this field on the whole system.
        call_rows = None if args.compare == "eager" else args.receivers
        # --species-index: the one-hot on the host from the construction on (it never reaches the device).
        built = plan if args.mode in ("compiled", "auto") else eager
        field = planner.build_evaluation(built, user, atoms, call_rows=call_rows,
                                         release_node_attrs=args.species_index,
                                         release_whole_system=args.species_index,
                                         global_list=not args.tile_list)
        record["call_rows"] = field.call_rows
        record["rewrites"] = list(getattr(field, "applied", ()))
        record["plan_memory"] = list(plan.decision("memory")) if any(d[0] == "memory" for d in plan.decisions) else None
        record["mode"] = args.mode
        if args.first_call_probe != "default":
            from fastmace.model import tiled as tiled_module

            tiled_module.TILE_FIRST_CALL_PROBE = args.first_call_probe == "on"
        record["first_call_probe"] = args.first_call_probe
        tiled = TiledForceField(field, TileConfig(args.receivers, args.edges, mode=args.mode,
                                                  neighbours="tiles" if args.tile_list else "global",
                                                  budget_receivers=() if args.no_budget else
                                                  tuple(args.budget_receivers) or None,
                                                  state="host" if args.host_state else "device",
                                                  alias_state=args.alias_state,
                                                  state_dtype=None if args.state_dtype == "none" else args.state_dtype,
                                                  wavefront=args.wavefront, lanes=args.lanes,
                                                  compiled_graphs={"auto": None, "on": True, "off": False}[args.compiled_graphs],
                                                  prefetch=args.prefetch == "on",
                                                  boundary_skin=args.boundary_skin == "on",
                                                  backward_chunk_scale=args.backward_chunk_scale,
                                                  grid_fit=args.grid_fit == "on",
                                                  list_cache=args.list_cache == "on",
                                                  list_reuse=args.list_reuse == "on",
                                                  **({} if args.wavefront_group is None else
                                                     dict(wavefront_group=args.wavefront_group)),
                                                  species="index" if args.species_index else "one_hot"))
        record["backward_chunk_edges"] = [module.chunked_backward_edges for module in tiled.model.modules()
                                          if getattr(module, "chunked_backward_edges", 0) > 0]
        record["phases"]["build"] = dict(seconds=time.perf_counter() - started, **mem())
        if args.memory_history:
            record["build_peak"] = peak_live_set(torch.cuda.memory._snapshot())
            torch.cuda.memory._record_memory_history(enabled=None)
        ledgers = record["ledgers"] = {}

        def ledger(name):
            if args.ledger:
                ledgers[name] = memory_ledger(dict(user=user, field=field, tiled=tiled, atoms=atoms))[:25]

        ledger("build")
        save()

        record["phase"] = "neighbours"
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        positions = field._prepare_one_off(atoms, host=args.host_positions)
        record["host_positions"] = bool(args.host_positions)
        record["edges_within_cutoff"] = int(field.neighbours.count)
        record["phases"]["neighbours"] = mem(before)
        save()

        calls = []
        out = None
        for call in range(args.calls):
            record["phase"] = f"tiled call {call}"
            del out
            out = None
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            if args.sweep_profile and call == args.calls - 1:
                import tiled_corun

                started = time.perf_counter()
                record["sweep_profile"], out = tiled_corun.sweep_profile(tiled, positions)
            elif args.sweep_times and call == args.calls - 1:
                import tiled_corun

                started = time.perf_counter()
                record["sweep_times"], out = tiled_corun.sweep_times(tiled, positions)
            elif args.profile_last and call == args.calls - 1:
                import tiled_corun

                started = time.perf_counter()
                record["profile"], out = tiled_corun.profile_call(tiled, positions, host_ops=25)
            else:
                history = args.call_history == call
                if history:
                    torch.cuda.memory._record_memory_history(max_entries=args.history_entries, stacks="python")
                started = time.perf_counter()
                out = tiled(positions)
                if history:
                    torch.cuda.synchronize()
                    snapshot = torch.cuda.memory._snapshot()
                    torch.cuda.memory._record_memory_history(enabled=None)
                    entries = sum(len(trace) for trace in snapshot.get("device_traces", []))
                    record["call_peak"] = dict(call=call, entries=entries,
                                               overflowed=entries >= args.history_entries,
                                               **peak_live_set(snapshot, top=40))
                    del snapshot
            torch.cuda.synchronize()
            entry = dict(seconds=time.perf_counter() - started, **mem(before), **tiled.last)
            # A call that measures its tiles' workspace resets the peak at its first tile: its peak is the larger of
            # the one before the tiles and the one since.
            entry["peak_above_before"] = max(entry["peak_above_before"],
                                             int(tiled.last.get("peak_before_tiles", 0)) - int(before))
            # ``allocated`` is after the call (the tiled field keeps its global state between calls); the call's
            # true peak is what was allocated before it plus its peak above that.
            entry.update(allocated_before=int(before), total_peak=int(before) + entry["peak_above_before"])
            calls.append(entry)
            record["calls"] = calls
            ledger(f"after call {call}")
            save()
        if args.ledger:
            record["ledger"] = memory_ledger(dict(user=user, field=field, tiled=tiled, atoms=atoms, out=out,
                                                  positions=positions))
            save()
        energy = float(out.energy.reshape(-1)[0])
        # The largest force from chunks of the output itself: a whole float64 copy (24 bytes per atom) does not
        # fit next to a 149M-atom call's residency on a 16 GB GPU, and |f| max is exact in the output's dtype.
        forces = out.forces.detach()
        max_force = max(float(chunk.abs().max()) for chunk in forces.reshape(-1).split(1 << 24))
        stress = out.stress.detach().double() if out.stress is not None else None
        if args.tile_list:  # the force field's list is empty: the edges the tiles built
            record["edges_within_cutoff"] = int(tiled.last["edges"])
        record["tiled"] = dict(energy_eV=energy, energy_per_atom_eV=energy / n,
                               max_abs_force=max_force,
                               stress=None if stress is None else stress.cpu().tolist())
        save()

        if args.compare != "none":
            record["phase"] = f"compare {args.compare}"
            # --tile-list left the force field's own list empty: the whole-system reference needs it, so it is
            # built here, at the same positions (the tiled calls are done).
            lists.close()
            ref_field = field
            if args.tile_list and args.compare == "eager":
                positions = field._prepare_one_off(atoms)
                record["reference_edges_within_cutoff"] = int(field.neighbours.count)
            if args.compare == "compiled":
                ref_field = planner.build_evaluation(plan, user, atoms)
                record["reference_rewrites"] = list(getattr(ref_field, "applied", ()))
                positions = ref_field._prepare_one_off(atoms)
            runs = []
            for _ in range(2):
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                ref = ref_field(positions)
                torch.cuda.synchronize()
                runs.append(dict(seconds=time.perf_counter() - started, **mem(before)))
            r_energy = float(ref.energy.detach().reshape(-1)[0])
            r_forces = ref.forces.detach().double()
            r_stress = ref.stress.detach().double().reshape(3, 3) if ref.stress is not None else None
            record["reference"] = dict(
                kind=args.compare, calls=runs, energy_eV=r_energy,
                energy_diff_meV_per_atom=1e3 * (energy - r_energy) / n,
                max_force_diff=float((forces - r_forces).abs().max()),
                rms_force_diff=float((forces - r_forces).pow(2).mean().sqrt()),
                max_abs_force=float(r_forces.abs().max()),
                max_stress_diff=None if stress is None else float((stress - r_stress).abs().max()),
                stress=None if r_stress is None else r_stress.cpu().tolist())
        # A list without pairs (or zero forces) is a failed build, not a capacity point (31 October 2026: the
        # pair screen silently did not launch at 1.73M and 2.0M atoms). One force call, no MD: part (a) only.
        valid, detail = capacity_scan.check_point(args.structure, float(field.neighbours.cutoff), n,
                                                  int(record.get("edges_within_cutoff") or 0),
                                                  record["tiled"]["max_abs_force"])
        record["valid"], record["check"] = valid, detail
        record["phase"] = "done" if record["valid"] else "invalid"
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        # cuEquivariance allocates outside PyTorch's allocator: its out-of-memory is a RuntimeError
        # (cudaErrorMemoryAllocation). Any other RuntimeError is recorded, then raised.
        message = str(error)
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message
        record["error"] = f"{type(error).__name__}: {message.splitlines()[0] if message else ''}"[:600]
        if oom:
            record["oom_at"] = record["phase"]
        import traceback

        record["traceback"] = [f"{frame.filename}:{frame.lineno} {frame.name}: {frame.line}"
                               for frame in traceback.extract_tb(error.__traceback__)][-25:]
        if not oom:
            lists.close()
            save()
            raise
    lists.close()
    save()
    print(json.dumps({k: record.get(k) for k in ("atoms", "phase", "error", "calls", "tiled", "reference")},
                     default=str)[:3000])


if __name__ == "__main__":
    main()
