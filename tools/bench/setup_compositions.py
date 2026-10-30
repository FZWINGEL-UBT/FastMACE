"""Engine setup across compositions (owner request, 30 October 2026): what a new composition's build costs, stage by
stage, and what of it changes with the composition.

Builds one float32 variable-cell relaxation per structure (MH-1 matpes_r2scan, the default plan and rewrites,
gpu-LBFGS with ``FrechetCell(scalar_pressure_GPa=0)``: Energy_Hull step 2 in float32), all in one process, in the
order given, and runs ``--steps`` relaxation steps on each. Per structure it records:

- **Stages:** ``fp32_setup.py``'s nested timers (load and conversion, plan, rewrites, species tables, radial-table
  fit and certificate, self-checks, OEq render and NVRTC, the first compiled call, warm-up, capture), plus
  ``PyCodeCache`` module loads, ``CUDAGraph.capture_end`` (stream end and ``cudaGraphInstantiate``), and the
  difference of torch's ``compilation_time_metrics`` (Dynamo, AOTAutograd, Inductor codegen and compile) and of
  Dynamo's counters (FX-graph and AOT-autograd cache hits and misses) over the structure's build.
- **What was generated:** the keys of every Inductor/Triton module loaded through ``PyCodeCache`` (hashes of the
  generated source, so equal keys mean equal code), the SHA-256 of every OEq source handed to NVRTC and of every
  fused-forward source, and the files each cache directory gained (``FASTMACE_CACHE``,
  ``TORCHINDUCTOR_CACHE_DIR``, ``TRITON_CACHE_DIR``, ``MACE_COMPILE_CACHE``): a file that appears is a miss.
- **Graphs (``--graph-dump``):** every captured CUDA graph is kept (``enable_debug_mode``) and written as DOT
  (``cudaGraphDebugDotPrint``); ``--graph-summary`` reduces each to its node list (type, kernel name, grid, block,
  shared memory) for the topology comparison. Timing runs leave this off.

Measurement only; no library change. The launcher sets the cache directories (cold: fresh and empty; warm: the same
again). Writes ``setup.json`` into --output.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROCESS_T0 = time.perf_counter()
import fastmace  # noqa: E402,F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp32_setup  # noqa: E402  (its timers and records)

CACHE_VARIABLES = ("FASTMACE_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "MACE_COMPILE_CACHE")
GENERATED = []  # (kind, key or sha256, detail)
EXTRA = []  # (label, start s since PROCESS_T0, seconds)
GRAPHS = []  # torch.cuda.CUDAGraph objects captured since the last reset


def _sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def _timed_extra(owner, attribute, label, record=None):
    original = getattr(owner, attribute)
    if getattr(original, "_setup_timed", False):
        return

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        finally:
            EXTRA.append((label, started - PROCESS_T0, time.perf_counter() - started))
        if record is not None:
            record(args, kwargs, result)
        return result

    wrapper._setup_timed = True
    setattr(owner, attribute, wrapper)


def instrument_generation(graph_dump):
    import torch
    from torch._inductor import codecache

    def py_module(args, kwargs, _result):
        key = kwargs.get("key", args[0] if args else None)
        path = kwargs.get("path", args[1] if len(args) > 1 else None)
        GENERATED.append(("inductor_module", str(key), Path(str(path)).name if path else None))

    for name in ("load_by_key_path",):
        if hasattr(codecache.PyCodeCache, name):
            _timed_extra(codecache.PyCodeCache, name, "module load: PyCodeCache." + name, py_module)
    from fastmace.kernels.oeq_streaming import fused_forward, runtime

    _timed_extra(runtime, "compile_cubin", "OEq streaming: NVRTC compile_cubin (all)",
                 lambda a, k, r: GENERATED.append(("nvrtc_source", _sha(a[0]), len(a[0]))))
    _timed_extra(fused_forward, "render", "fused forward: render (all)",
                 lambda a, k, r: GENERATED.append(("fused_forward_source", _sha(r), len(r))))

    from fastmace.rewrites import registry

    original_check = registry.Registry.selfcheck

    def selfcheck(self, name, *args, **kwargs):  # per record: which checks cost what
        started = time.perf_counter()
        try:
            return original_check(self, name, *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            EXTRA.append((f"self-check: {name}", started - PROCESS_T0, time.perf_counter() - started))

    registry.Registry.selfcheck = selfcheck
    graphs = torch.cuda.graphs
    original_init = graphs.CUDAGraph.__init__
    original_end = graphs.CUDAGraph.capture_end

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if graph_dump:
            self.enable_debug_mode()
        GRAPHS.append(self)

    def capture_end(self):
        started = time.perf_counter()
        try:
            return original_end(self)
        finally:
            EXTRA.append(("graph: capture_end (end capture + instantiate)", started - PROCESS_T0,
                          time.perf_counter() - started))

    graphs.CUDAGraph.__init__ = init
    graphs.CUDAGraph.capture_end = capture_end


def cache_files():
    found = {}
    for variable in CACHE_VARIABLES:
        root = os.environ.get(variable)
        if not root or not os.path.isdir(root):
            continue
        for directory, _dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(directory, name)
                try:
                    found[f"{variable}:{os.path.relpath(path, root)}"] = os.path.getsize(path)
                except OSError:
                    pass
    return found


def compile_metrics():
    from torch._dynamo import utils

    metrics = getattr(utils, "compilation_time_metrics", {})
    return {key: list(values) for key, values in metrics.items()}


def dynamo_counters():
    from torch._dynamo.utils import counters

    return {group: dict(values) for group, values in counters.items() if values}


def _diff_metrics(before, after):
    out = {}
    for key, values in after.items():
        new = values[len(before.get(key, [])):]
        if new:
            out[key] = dict(total_s=sum(new), calls=len(new))
    return dict(sorted(out.items(), key=lambda item: -item[1]["total_s"]))


def _diff_counters(before, after):
    out = {}
    for group, values in after.items():
        for key, value in values.items():
            delta = value - before.get(group, {}).get(key, 0)
            if delta:
                out[f"{group}.{key}"] = delta
    return out


NODE = re.compile(r'^"?(?P<id>[^"\s]+)"?\s*\[(?P<attrs>.*)\];?\s*$')


def summarise_dot(path):
    """Node list of a cudaGraphDebugDotPrint file: per node its type and, for kernels, name, grid, block, smem."""
    nodes, edges = [], 0
    text = Path(path).read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if "->" in line:
            edges += 1
            continue
        match = NODE.match(line)
        if not match or "label" not in match.group("attrs"):
            continue
        attrs = match.group("attrs")
        label = re.search(r'label="(.*)"', attrs) or re.search(r"label=<(.*)>", attrs)
        label = label.group(1) if label else attrs
        kind = re.search(r"(KERNEL|MEMCPY|MEMSET|HOST|EMPTY|EVENT_RECORD|WAIT_EVENT|MEM_ALLOC|MEM_FREE|GRAPH|"
                         r"CONDITIONAL|BATCH_MEM_OP|EXT_SEMAS_SIGNAL|EXT_SEMAS_WAIT)", label)
        node = dict(type=kind.group(1) if kind else "?")
        if node["type"] == "KERNEL":
            func = re.search(r"\\?n?([A-Za-z_][\w:<>,\s\*&]*?)\s*\\n\\{", label) or \
                re.search(r"\|\s*([^|\\]+?)\s*\\n", label) or re.search(r"fn[^\w]*=\s*([^\\|}]+)", label)
            for field, pattern in (("grid", r"gridDim[^\d]*([\d,\s]+)"), ("block", r"blockDim[^\d]*([\d,\s]+)"),
                                   ("smem", r"sharedMemBytes[^\d]*(\d+)")):
                found = re.search(pattern, label)
                node[field] = found.group(1).replace(" ", "").strip(",") if found else None
            node["name"] = func.group(1).strip() if func else None
            names = re.findall(r"([A-Za-z_]\w*(?:<[^>]*>)?(?:::\w+)*)\s*\(", label)
            if not node["name"] and names:
                node["name"] = names[0]
        nodes.append(node)
    return dict(nodes=nodes, edges=edges, counts=dict(Counter(n["type"] for n in nodes)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="a fresh directory")
    p.add_argument("--structures", nargs="+", required=True, help="NAME=path.cif[:strain], built in this order")
    p.add_argument("--repeat", default="3x3x3")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head", default="matpes_r2scan")
    p.add_argument("--precision", default="float32")
    p.add_argument("--steps", type=int, default=5, help="relaxation steps after the build")
    p.add_argument("--label", default="")
    p.add_argument("--graph-dump", action="store_true")
    p.add_argument("--dynamo-config", action="append", default=[], metavar="NAME=VALUE",
                   help="set torch._dynamo.config.NAME (a Python literal) before the first build, e.g. "
                        "caching_precompile=True (the compile-cache probe)")
    p.add_argument("--allow-busy-gpu", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model import compile as compiling

    allocator = compiling.configure_allocator()  # before anything initialises CUDA (the engine checks it)
    import ab_step_time
    import numpy as np
    import torch
    from ase.io import read

    import ast

    import torch._dynamo.config as dynamo_config

    for item in args.dynamo_config:
        name, _, value = item.partition("=")
        setattr(dynamo_config, name, ast.literal_eval(value))
    if not args.allow_busy_gpu and ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    wrapped = fp32_setup.instrument()
    instrument_generation(args.graph_dump)
    import fastmace as fm
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    repeat = tuple(int(v) for v in args.repeat.split("x"))
    out = dict(tool="setup_compositions", created_utc=datetime.now(timezone.utc).isoformat(),
               host=os.uname().nodename, label=args.label, fastmace_commit=ab_step_time.fastmace_commit(),
               python=sys.executable, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               environment={k: os.environ.get(k) for k in CACHE_VARIABLES + (
                   "TORCHINDUCTOR_COMPILE_THREADS", "CUDA_VISIBLE_DEVICES")},
               args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               wrapped=wrapped, cuda_allocator=allocator, structures=[])
    started = time.perf_counter()
    model = fm.load("mh1", head=args.head, checkpoint=args.checkpoint)
    out["model_load_s"] = time.perf_counter() - started
    out["process_to_model_s"] = time.perf_counter() - PROCESS_T0
    record = fm.optimize.LBFGS(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
    for spec in args.structures:
        name, _, path = spec.partition("=")
        path, _, strain = path.partition(":")
        atoms = read(path).repeat(repeat)
        if strain:
            atoms.set_cell(atoms.cell.array * float(strain), scale_atoms=True)
        row = dict(name=name, path=path, strain=float(strain) if strain else 1.0, natoms=len(atoms),
                   formula=atoms.get_chemical_formula(), species=sorted(set(atoms.get_chemical_symbols())))
        files_before, metrics_before, counters_before = cache_files(), compile_metrics(), dynamo_counters()
        n_records, n_extra, n_generated = len(fp32_setup.RECORDS), len(EXTRA), len(GENERATED)
        GRAPHS.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        field = ForceField(model, atoms, precision=args.precision)
        row["forcefield_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        sim = Simulation(atoms.copy(), field, optimizer=record, seed=0)
        torch.cuda.synchronize()
        row["simulation_s"] = time.perf_counter() - t1
        row["setup_parts_s"] = dict(getattr(sim, "_setup_s", {}) or {})
        t2 = time.perf_counter()
        _relaxed, summary = sim.relax(fmax=1e-9, steps=args.steps)
        torch.cuda.synchronize()
        row["relax_s"] = time.perf_counter() - t2
        row["build_to_steps_s"] = time.perf_counter() - t0
        row["relax"] = {k: summary.get(k) for k in ("steps", "seconds", "recaptures", "neighbour_rebuilds")}
        engine = getattr(sim, "_engine", None)
        info = dict(engine.info()) if engine is not None else {}
        row["engine_info"] = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, type(None)))}
        row["plan"] = str(sim.plan.describe()) if getattr(sim, "plan", None) is not None else None
        records = fp32_setup.RECORDS[n_records:]
        row["stages"] = fp32_setup.summary(records)
        extra = {}
        for label, _start, seconds in EXTRA[n_extra:]:
            extra.setdefault(label, dict(total_s=0.0, calls=0))
            extra[label]["total_s"] += seconds
            extra[label]["calls"] += 1
        row["extra_stages"] = extra
        row["timeline"] = [dict(label=r[0], parent=r[1], start_s=round(r[2], 4), seconds=round(r[3], 4),
                                depth=r[4]) for r in sorted(records, key=lambda r: r[2])]
        row["compile_metrics"] = _diff_metrics(metrics_before, compile_metrics())
        row["dynamo_counters"] = _diff_counters(counters_before, dynamo_counters())
        files_after = cache_files()
        new = {k: v for k, v in files_after.items() if k not in files_before}
        row["cache_new_files"] = new
        row["cache_new_by_variable"] = {var: dict(files=sum(1 for k in new if k.startswith(var + ":")),
                                                  bytes=sum(v for k, v in new.items() if k.startswith(var + ":")))
                                        for var in CACHE_VARIABLES}
        row["generated"] = [dict(kind=k, key=key, detail=d) for k, key, d in GENERATED[n_generated:]]
        if args.graph_dump and engine is not None:
            graph_dir = args.output / "graphs" / name
            graph_dir.mkdir(parents=True, exist_ok=True)
            keys = {id(g): str(k) for k, g in getattr(engine, "_graphs", {}).items()}
            row["graphs"] = []
            for index, graph in enumerate(GRAPHS):
                key = keys.get(id(graph), f"other-{index}")
                path = graph_dir / f"{index:02d}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', key)[:80]}.dot"
                try:
                    graph.debug_dump(str(path))
                    row["graphs"].append(dict(index=index, key=key, file=str(path.relative_to(args.output)),
                                              **summarise_dot(path)))
                except Exception as error:  # noqa: BLE001  (a measurement record)
                    row["graphs"].append(dict(index=index, key=key, error=f"{type(error).__name__}: {error}"))
        sim.close()
        del sim, field, engine
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        out["structures"].append(row)
        (args.output / "setup.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        top = sorted(row["stages"].items(), key=lambda item: -item[1]["total_s"])[:6]
        print(f"{name} {row['natoms']} atoms: forcefield {row['forcefield_s']:.1f} s, simulation "
              f"{row['simulation_s']:.1f} s, {args.steps} steps {row['relax_s']:.1f} s; new cache files "
              f"{len(new)}; " + "; ".join(f"{k} {v['total_s']:.1f}" for k, v in top), flush=True)
    out["process_total_s"] = time.perf_counter() - PROCESS_T0
    (args.output / "setup.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    main()
