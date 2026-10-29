"""Timing breakdown of one MD step of FastMACE's production default (plan task 2-T20a).

The same three measurements as the research engine's
studies/further_optimizations/probes/step_breakdown_probe.py (15 October 2026), in one process,
after equilibrating at --temperature-K, so the tables compare row by row:

1. Wall time: --steps MD steps of the planned GraphEngine (CUDA graphs, pipelined where the plan
   has it), twice; the second window counts (the first may capture graphs for new edge sizes).
2. Stages: --steps further steps with CUDA events around every stage the host launches (graph
   replays ACP or AC, B and C), the neighbour-list rebuilds (``force_field.rebuild``) and the noise
   upload (``integrator.prepare_step``). GPU time not covered by a stage is idle time between stages.
3. Force stage by kernel: --calls eager calls of the force field (the compiled model whose kernels
   the B graph replays) under the profiler. Every kernel is attributed to the CPU operator that
   launched it and to a model part by operator name and tensor shapes (edge-sized against
   atom-sized), with the research engine's categories. FastMACE's Triton and NVRTC kernels run in
   custom ops (``fastmace::...``), so their operator names identify them.
4. Replays of the force graph under the profiler: kernels per replay, kernel time against the
   replay span, the largest gaps.

Profiling columns for the optimisation round (plans/opt-1.md): per kernel the launch configuration and the
profiler's estimated occupancy, per GEMM operator the FLOPs from its shapes and the achieved FLOP/s against
the nominal FP32 peak (gemms.csv), and the gaps inside each force-graph replay. Two further processes, run
under the external profilers after the main pass and writing into its directory:

  --pass ncu    one eager force call between cudaProfilerStart/Stop, for ``ncu --profile-from-start off``
                (NCU_METRICS: DRAM and L2 bytes, FP32 instructions, grid, block, registers, achieved
                occupancy); ``--summarize-ncu`` joins ncu's kernels by launch order with the main pass's
                kernel_sequence.csv and writes ncu_kernels.csv, ncu_categories.csv and ncu_ops.csv
  --pass nsys   --calls force-graph replays in that range, for ``nsys profile --capture-range=cudaProfilerApi
                --cuda-graph-trace=node``; ``--summarize-nsys`` adds the gaps between graph nodes
  --pass nsys-steps
                --steps whole MD steps in that range, with NVTX ranges around each step, each stage
                (graph replays), the neighbour-list rebuilds and the noise upload (fp32 profile, 28 October
                2026), for the same nsys command; ``nsys stats --report nvtx_gpu_proj_sum`` then gives GPU
                time per stage

The engine comes from the minimal planner (what ``Simulation`` and ``fm.md`` build) with every
default of the plan. Velocities start from a Maxwell-Boltzmann distribution at --temperature-K with
the research probe's seed. Measurement only: the engine's methods are wrapped on the instance for
part 2 and restored afterwards. Use an otherwise idle GPU; hold the local reservation
(tools/ci/local_gpu.py run). Writes summary.json, stages.csv, categories.csv and ops.csv.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import dataclasses
import difflib
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402  (tools/: structures, commit, GPU processes)

STAGE_NAMES = {"ACP": "ACP: Langevin half kick, positions, neighbour check, edge compaction",
               "AC": "AC: Langevin half kick, positions, neighbour check, edge compaction",
               "B": "B: forces + second half kick",
               "C": "C: edge compaction after a rebuild"}
REBUILD = "Neighbour-list rebuild (GPU)"
NOISE = "Noise upload (every step a slice; a new batch every noise_batch)"
B_STAGE = STAGE_NAMES["B"]
PACKAGES = ("torch", "triton", "mace-torch", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
            "cuequivariance-ops-torch-cu13", "openequivariance", "e3nn", "ase", "numpy")


class _Labelled:
    """A Triton launcher run under a profiler label: FastMACE's Triton kernels are all named ``kernel``, and
    those launched from compiled backward code have no custom-op frame of their own in the profile. The
    label (the custom op's name) attributes them. Set on the module for this process only (the research
    probe's ``label_triton_kernels``)."""

    def __init__(self, label, kernel):
        self.label, self.kernel = label, kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            import torch
            with torch.profiler.record_function(self.label):
                return launch(*args, **kwargs)
        return run


def label_triton_kernels():
    """Wrap the launchers of fastmace.kernels.triton (skinny_linear, edge_outer_sum, fused_radial)."""
    from fastmace.kernels.triton import edge_outer_sum, fused_radial, skinny_linear

    for module, attribute, label in ((skinny_linear, "_triton_kernel", "fastmace::folded_linear_input_grad"),
                                     (edge_outer_sum, "_triton_forward", "fastmace::edge_outer_sum_forward"),
                                     (edge_outer_sum, "_triton_backward", "fastmace::edge_outer_sum_backward")):
        original = getattr(module, attribute)
        setattr(module, attribute, lambda original=original, label=label: _Labelled(f"{label} (triton)",
                                                                                    original()))
    original_kernels = fused_radial._kernels

    def labelled_kernels():
        kernels = dict(original_kernels())
        for key in ("forward", "backward"):
            kernels[key] = _Labelled(f"fastmace::fused_radial_hidden_{key} (triton)", kernels[key])
        return kernels

    fused_radial._kernels = labelled_kernels


def first_rows(dims):
    """Leading sizes of all 2-D or larger input tensors of an operator."""
    try:
        shapes = json.loads(dims) if dims else []
    except json.JSONDecodeError:
        return []
    rows = []
    for shape in shapes:
        if isinstance(shape, list) and len(shape) >= 2 and all(isinstance(v, int) for v in shape):
            rows.append(shape[0])
    return rows


def category(kernel, op, dims, edges):
    """The research probe's model part of one kernel; FastMACE's custom ops are named ``fastmace::...``."""
    name = f"{op} {kernel}"
    if "fused_radial_hidden" in name:
        return "Radial networks: fused hidden block"
    if "folded_linear_input_grad" in name:
        return "Radial networks: folded last layer, input gradient"
    if "edge_outer_sum" in name:
        return "Interaction 1: reassociated edge sums"
    rows = first_rows(dims)
    edge_sized = edges in rows
    if "jit_conv_forward" in op:
        return "OEq convolution forward"
    if "jit_conv_backward" in op:
        return "OEq convolution backward"
    if "streaming_conv_forward" in name:
        return "Interaction 2: convolution forward (streaming)"
    if "streaming_conv_backward" in name:
        return "Interaction 2: convolution backward (streaming)"
    if op in ("aten::mm", "aten::addmm", "aten::bmm"):
        if edge_sized:
            try:
                shapes = json.loads(dims)
            except json.JSONDecodeError:
                shapes = []
            weight = next((s for s in shapes if isinstance(s, list) and len(s) == 2 and s[0] != edges), None)
            if weight and max(weight) >= 1000:
                return "Radial networks: folded last layer, forward GEMM"
            return "Radial networks: other GEMMs"
        return "Atom-level linear layers (GEMM)"
    if "uniform_1d" in name or "segmented_polynomial" in name:
        return "Atom-level cuEq tensor products (symmetric contraction)"
    if kernel.startswith(("Memcpy", "Memset")) or op in ("aten::copy_", "aten::zero_", "aten::fill_"):
        return "Copies and fills"
    if edge_sized:
        if "layer_norm" in kernel or "silu" in kernel:
            return "Radial networks: elementwise (norms, activations)"
        return "Edge elementwise (geometry, radial basis, gathers, scatters)"
    if "triton" in kernel or "triton" in op:
        return "Atom-level elementwise (gates, products, residuals, readouts)"
    return "Other"


def package_versions():
    from importlib import metadata
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_engine(args, base):
    """The minimal planner's engine for ``base`` (as tools/ab_fastmace.py's 1b side), and its plan."""
    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import fixed, minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend=args.backend))
    hardware = planner.hardware()  # the allocator rule, before the first CUDA allocation
    started = time.perf_counter()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha256(args.checkpoint))
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    rewrites = RewriteConfig(**args.rewrites) if args.rewrites else "auto"
    force_field = ForceField(loaded, base, precision="float32" if args.backend == "hybrid" else "float64",
                             rewrites=rewrites)
    integrator = (IntegratorConfig(temperature_K=args.temperature_K) if args.timestep_fs is None else
                  IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=args.timestep_fs))
    plan = planner.plan(force_field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
    engine = planner.build(plan, force_field, base.copy(), temperature_K=args.temperature_K)
    options = {name: True for name in fixed.enabled_options(plan)}
    options.update({f"off:{name}": reason for name, reason in fixed.auto_options_off(plan).items()})
    return engine, dict(describe=plan.describe(full=True), options=options, load_s=load_s,
                        setup_s=time.perf_counter() - started, sha256=loaded.sha256)


def stage_name(key):
    kind = key[0] if isinstance(key, tuple) else key
    return STAGE_NAMES.get(kind, str(kind))


def read_trace(path):
    events = json.loads(path.read_text())
    path.unlink()  # large; the tables keep what is needed
    return events.get("traceEvents", events)


def gemm_flops(op, dims):
    """Floating-point operations of one GEMM operator from its input shapes (2 M K N per product), else None."""
    try:
        shapes = [s for s in json.loads(dims) if isinstance(s, list) and s] if dims else []
    except json.JSONDecodeError:
        return None
    matrices = [s for s in shapes if len(s) in (2, 3) and all(isinstance(v, int) for v in s)]
    # Inductor calls the out= variants, so an output matrix may follow the operands; addmm's bias comes
    # first and is 1-D (dropped below) or a 2-D [M, N].
    two = [s for s in matrices if len(s) == 2]
    if op in ("aten::mm", "aten::addmm") and len(two) >= 2:
        first = 0
        if (op == "aten::addmm" and len(two) >= 3 and two[1][1] == two[2][0]
                and two[0] == [two[1][0], two[2][1]]):
            first = 1  # 2-D bias
        (m, k), (k2, n) = two[first], two[first + 1]
        return 2 * m * k * n if k == k2 else None
    if op == "aten::bmm" and len(matrices) >= 2 and len(matrices[0]) == len(matrices[1]) == 3:
        (b, m, k), (b2, k2, n) = matrices[:2]
        return 2 * b * m * k * n if (b, k) == (b2, k2) else None
    return None


def fp32_peak_tflops(major, minor, sms, clock_mhz):
    """Nominal FP32 FMA peak: SMs x FP32 lanes per SM x 2 x the maximum SM clock (sm_80: 64 lanes, other
    sm_8x to sm_12x: 128). A reporting reference, not a tuning input."""
    lanes = 64 if (major, minor) == (8, 0) else 128
    return sms * lanes * 2 * clock_mhz * 1e6 / 1e12


def max_sm_clock_mhz():
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
                                      text=True)
        return float(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def new_row():
    return dict(launches=0, us=0.0, ops=set(), strides="", grid=None, block=None, registers=None, shared_bytes=None,
                occupancy_us=0.0, occupancy_weight=0.0)


def _owners(events):
    """(owner, launches, kernels): ``owner(launch)`` is the innermost CPU operator enclosing a kernel's launch,
    as (name, input dims, invocation id, input strides), or ("?", "", None, "")."""
    ops = defaultdict(list)
    launches, kernels = {}, []
    for e in events:
        cat = e.get("cat")
        if cat in ("cpu_op", "user_annotation") and "dur" in e:
            ops[e["tid"]].append(e)
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
    index = {}
    for tid, items in ops.items():
        items.sort(key=lambda e: e["ts"])
        index[tid] = ([e["ts"] for e in items], items)

    def owner(launch):
        if launch is None or launch["tid"] not in index:
            return "?", "", None, ""
        starts, items = index[launch["tid"]]
        best = None
        for e in reversed(items[:bisect.bisect_right(starts, launch["ts"])]):
            if e["ts"] + e["dur"] >= launch["ts"] + launch.get("dur", 0):
                if best is None or e["dur"] < best["dur"]:
                    best = e
                if launch["ts"] - e["ts"] > 2e5:
                    break
        if best is None:
            return "?", "", None, ""
        args = best.get("args", {})
        return (best["name"], json.dumps(args.get("Input Dims", ""))[:4000], (best["tid"], best["ts"]),
                json.dumps(args.get("Input Strides", ""))[:4000])

    return owner, launches, kernels


def kernel_table(events):
    """{(kernel, op, input dims): row} with each kernel attributed to its innermost CPU operator.

    A row holds the launches, the kernel time in µs, the operator invocations it came from, the launch
    configuration (grid, block, registers per thread, shared memory) and the profiler's estimated achieved
    occupancy, weighted by duration."""
    owner, launches, kernels = _owners(events)
    table = defaultdict(new_row)
    for k in kernels:
        args = k.get("args", {})
        op, dims, invocation, strides = owner(launches.get(args.get("correlation")))
        row = table[(k["name"][:120], op, dims)]
        row["strides"] = row["strides"] or strides
        row["launches"] += 1
        row["us"] += k["dur"]
        if invocation is not None:
            row["ops"].add(invocation)
        for key, arg in (("grid", "grid"), ("block", "block"), ("registers", "registers per thread"),
                         ("shared_bytes", "shared memory")):
            if row[key] is None and arg in args:
                row[key] = args[arg]
        estimate = args.get("est. achieved occupancy %")
        if isinstance(estimate, (int, float)):
            row["occupancy_us"] += estimate * k["dur"]
            row["occupancy_weight"] += k["dur"]
    return table, len(kernels)


def occupancy(row):
    """Duration-weighted estimated achieved occupancy (%) of a row, or None."""
    return row["occupancy_us"] / row["occupancy_weight"] if row["occupancy_weight"] else None


def op_gemm_rates(table, calls, peak_tflops):
    """Per GEMM operator (op, dims): invocations and kernel time per force call, FLOPs and achieved TFLOP/s."""
    ops = defaultdict(lambda: dict(invocations=set(), us=0.0, kernels=[], strides=""))
    for (name, op, dims), row in table.items():
        if gemm_flops(op, dims) is None:
            continue
        entry = ops[(op, dims)]
        entry["invocations"] |= row["ops"]
        entry["us"] += row["us"]
        entry["kernels"].append(name)
        entry["strides"] = entry["strides"] or row["strides"]
    rates = {}
    for (op, dims), entry in ops.items():
        flops = gemm_flops(op, dims) * len(entry["invocations"])
        tflops = flops / (entry["us"] * 1e-6) / 1e12 if entry["us"] else None
        rates[(op, dims)] = dict(invocations_per_call=len(entry["invocations"]) / calls,
                                 us_per_call=entry["us"] / calls, gflop_per_call=flops / calls / 1e9,
                                 achieved_tflops=tflops,
                                 share_of_peak=(tflops / peak_tflops if tflops and peak_tflops else None),
                                 kernels=sorted(set(entry["kernels"])), input_strides=entry["strides"])
    return rates


def split_replays(kernels, replays):
    """``kernels`` [(name, start µs, end µs)] of ``replays`` back-to-back replays of one graph, in time order,
    as one list per replay (each replay launches the same number of kernels), else None."""
    if not kernels or len(kernels) % replays:
        return None
    n = len(kernels) // replays
    return [kernels[i * n:(i + 1) * n] for i in range(replays)]


def replay_gaps(per_replay):
    """Idle gaps between consecutive kernels inside each graph replay (not between replays): totals, a
    histogram and the largest gaps with their neighbours. ``per_replay``: one [(name, start, end)] per replay."""
    buckets = {"<2us": [0, 0.0], "2-5us": [0, 0.0], "5-20us": [0, 0.0], ">=20us": [0, 0.0]}
    gaps = []
    for kernels in per_replay:
        for (name_a, _, end_a), (name_b, start_b, _) in zip(kernels, kernels[1:]):
            gap = max(start_b - end_a, 0.0)  # overlapping kernels count as no gap
            key = "<2us" if gap < 2 else "2-5us" if gap < 5 else "5-20us" if gap < 20 else ">=20us"
            buckets[key][0] += 1
            buckets[key][1] += gap
            gaps.append((gap, name_a[:80], name_b[:80]))
    gaps.sort(reverse=True)
    replays = len(per_replay)
    return dict(gaps_per_replay=len(gaps) / replays, total_gap_us_per_replay=sum(g for g, _, _ in gaps) / replays,
                histogram_per_replay={k: dict(count=v[0] / replays, us=v[1] / replays) for k, v in buckets.items()},
                largest=[dict(gap_us=round(g, 2), before=a, after=b) for g, a, b in gaps[:20]])


def write_sequence(events, edges, path):
    """The kernels of one profiled force call in launch order, with operator, category and input shapes."""
    owner, launches, kernels = _owners(events)
    rows = []
    for index, k in enumerate(sorted(kernels, key=lambda e: e["ts"])):
        name = k["name"][:120]
        op, dims, _, _ = owner(launches.get(k.get("args", {}).get("correlation")))
        rows.append(dict(index=index, kernel=name, op=op, category=category(name, op, dims, edges),
                         input_dims=dims[:300]))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEQUENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def force_graph_key(engine, edges):
    """The captured B graph the engine replays now, else one for ``edges``, else None."""
    key = ("B", int(engine._pipe_edges))
    if key in engine._graphs:
        return key
    return next((k for k in engine._graphs if isinstance(k, tuple) and k[0] == "B" and k[1] == edges), None)


NCU_METRICS = (
    # DRAM bytes: dram__bytes_read/write on most chips, dram__bytes_op_read/op_write on GB20x (ncu drops
    # the names a chip does not have, so both families are requested).
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum", "lts__t_bytes.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread", "launch__waves_per_multiprocessor",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
SEQUENCE_FIELDS = ("index", "kernel", "op", "category", "input_dims")


def _number(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def read_ncu_csv(path):
    """Rows of ``ncu --csv --page raw --print-units base`` (a header row, then a units row, then one row per
    kernel), as dicts with numbers where a cell is numeric."""
    with Path(path).open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("==")]  # ncu's own messages
    rows = list(csv.reader(lines))
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if body and not str(body[0][0]).strip().isdigit():
        body = body[1:]  # the units row
    out = []
    for cells in body:
        record = dict(zip(header, cells))
        for key in NCU_METRICS:
            if key in record:
                record[key] = _number(record[key])
        out.append(record)
    return out


def _first(record, *keys):
    """The first of ``keys`` that ``record`` has (a chip's metric name family), else None."""
    return next((record[key] for key in keys if record.get(key) is not None), None)


def _base_name(name):
    """A kernel name without return type, template arguments and parameters, lower case and cut to 80
    characters (the profiler keeps 120 of a name), for matching ncu against the profiler."""
    name = str(name).strip()
    for stop in ("(", "<"):
        name = name.split(stop)[0]
    name = name.split("::")[-1].strip()
    if name.startswith("void "):
        name = name[5:]
    # cuEquivariance's uniform_1d picks its kernel variant at run time (segmented_polynomial_algo1_4,
    # _algo2_2, _bwd_algo6, ...), and the ncu process often picks another than the profiler's: one name each.
    name = re.sub(r"^(segmented_polynomial(?:_bwd)?)_algo\d+(?:_\d+)?$", r"\1", name)
    return name.lower()[:80]


def join_ncu(ncu_rows, sequence, peak_tflops):
    """ncu's kernels joined in launch order with the profiler's kernel sequence of one force call (same compiled
    code, so the same order); each row gets the sequence's operator and category. Returns (rows, matched)."""
    kernels = [s for s in sequence if not str(s["kernel"]).startswith(("Memcpy", "Memset"))]
    # Align by base name: ncu and the profiler see the same launches, but either may miss a few (ncu does
    # not profile memory operations; the profiler may drop a kernel), so plain positions can drift.
    matcher = difflib.SequenceMatcher(a=[_base_name(r.get("Kernel Name", "")) for r in ncu_rows],
                                      b=[_base_name(k["kernel"]) for k in kernels], autojunk=False)
    unknown = dict(index="", kernel="", op="?", category="Unattributed (not aligned with the profiler)",
                   input_dims="")
    partner = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            partner[block.a + offset] = kernels[block.b + offset]
    rows, matched = [], 0
    for position, record in enumerate(ncu_rows):
        name = record.get("Kernel Name", "")
        # Unaligned (cuEq picks its kernel variant at run time, so names can differ between processes):
        # categorise by the kernel's own name.
        entry = partner.get(position) or dict(unknown, category=category(name, "?", "", -1)
                                              if category(name, "?", "", -1) != "Other" else unknown["category"])
        same = position in partner
        matched += same
        ns = record.get("gpu__time_duration.sum") or 0.0
        flops = sum(2 * (record.get(k) or 0.0) if "ffma" in k else (record.get(k) or 0.0)
                    for k in NCU_METRICS if "_op_f" in k)
        tflops = flops / (ns * 1e-9) / 1e12 if ns else None
        rows.append(dict(index=entry["index"], kernel=entry["kernel"], ncu_kernel=name, name_matches=same,
                         op=entry["op"], category=entry["category"], input_dims=entry.get("input_dims", ""),
                         duration_us=ns / 1e3, dram_read_bytes=_first(record, "dram__bytes_read.sum",
                                                                      "dram__bytes_op_read.sum"),
                         dram_write_bytes=_first(record, "dram__bytes_write.sum", "dram__bytes_op_write.sum"),
                         l2_bytes=record.get("lts__t_bytes.sum"),
                         fp32_flops=flops, achieved_tflops=tflops,
                         share_of_fp32_peak=tflops / peak_tflops if tflops is not None and peak_tflops else None,
                         grid_size=record.get("launch__grid_size"), block_size=record.get("launch__block_size"),
                         registers=record.get("launch__registers_per_thread"),
                         waves_per_sm=record.get("launch__waves_per_multiprocessor"),
                         achieved_occupancy_pct=record.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
                         theoretical_occupancy_pct=record.get("sm__maximum_warps_per_active_cycle_pct"),
                         sm_throughput_pct=record.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                         memory_throughput_pct=record.get(
                             "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")))
    return rows, matched


def group_ncu(rows, key, peak_tflops):
    """ncu rows summed by ``key`` (``"category"`` or ``"op"``): time, DRAM and L2 bytes, FLOPs, achieved FLOP/s,
    and the duration-weighted achieved occupancy."""
    groups = defaultdict(lambda: dict(kernels=0, duration_us=0.0, dram_bytes=0.0, l2_bytes=0.0, fp32_flops=0.0,
                                      occupancy_weighted=0.0, occupancy_weight=0.0))
    for row in rows:
        g = groups[row[key] if key != "op" else (row["op"], row["input_dims"])]
        g["kernels"] += 1
        g["duration_us"] += row["duration_us"]
        g["dram_bytes"] += (row["dram_read_bytes"] or 0.0) + (row["dram_write_bytes"] or 0.0)
        g["l2_bytes"] += row["l2_bytes"] or 0.0
        g["fp32_flops"] += row["fp32_flops"]
        if row["achieved_occupancy_pct"] is not None:
            g["occupancy_weighted"] += row["achieved_occupancy_pct"] * row["duration_us"]
            g["occupancy_weight"] += row["duration_us"]
    out = {}
    for name, g in groups.items():
        seconds = g["duration_us"] * 1e-6
        tflops = g["fp32_flops"] / seconds / 1e12 if seconds else None
        out[name] = dict(kernels=g["kernels"], duration_us=g["duration_us"], dram_mb=g["dram_bytes"] / 1e6,
                         l2_mb=g["l2_bytes"] / 1e6, dram_gb_per_s=g["dram_bytes"] / seconds / 1e9 if seconds else None,
                         fp32_gflop=g["fp32_flops"] / 1e9, achieved_tflops=tflops,
                         share_of_fp32_peak=tflops / peak_tflops if tflops is not None and peak_tflops else None,
                         achieved_occupancy_pct=(g["occupancy_weighted"] / g["occupancy_weight"]
                                                 if g["occupancy_weight"] else None))
    return out


def read_nsys_gpu_trace(path):
    """Kernels of ``nsys stats --report cuda_gpu_trace --format csv`` as [(name, start µs, end µs)] in time order
    (memory operations dropped)."""
    with Path(path).open(newline="") as handle:
        lines = [line for line in handle if "," in line]
    reader = csv.DictReader(lines)
    kernels = []
    for row in reader:
        name = row.get("Name", "")
        start, duration = _number(row.get("Start (ns)", "")), _number(row.get("Duration (ns)", ""))
        if start is None or duration is None or name.startswith("[CUDA mem"):
            continue
        kernels.append((name, start / 1e3, (start + duration) / 1e3))
    kernels.sort(key=lambda k: k[1])
    return kernels


def summarize_profiles(args):
    """--summarize-ncu / --summarize-nsys: tables from the external profilers' exports into --output."""
    summary = json.loads((args.output / "summary.json").read_text())
    peak = summary.get("fp32_peak_tflops")
    if args.summarize_ncu:
        with (args.output / "kernel_sequence.csv").open(newline="") as handle:
            sequence = list(csv.DictReader(handle))
        rows, matched = join_ncu(read_ncu_csv(args.summarize_ncu), sequence, peak)
        with (args.output / "ncu_kernels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["index"])
            writer.writeheader()
            writer.writerows(rows)
        for key, file in (("category", "ncu_categories.csv"), ("op", "ncu_ops.csv")):
            groups = group_ncu(rows, key, peak)
            with (args.output / file).open("w", newline="") as handle:
                writer = csv.writer(handle)
                fields = ["kernels", "duration_us", "dram_mb", "l2_mb", "dram_gb_per_s", "fp32_gflop",
                          "achieved_tflops", "share_of_fp32_peak", "achieved_occupancy_pct"]
                writer.writerow([key] + (["input_dims"] if key == "op" else []) + fields)
                for name, g in sorted(groups.items(), key=lambda item: -item[1]["duration_us"]):
                    head = list(name) if key == "op" else [name]
                    writer.writerow(head + [g[f] if not isinstance(g[f], float) else f"{g[f]:.6g}" for f in fields])
        summary["ncu"] = dict(kernels=len(rows), sequence_kernels=len(sequence), names_matched=matched,
                              source=str(args.summarize_ncu))
    if args.summarize_nsys:
        per_replay = split_replays(read_nsys_gpu_trace(args.summarize_nsys), args.calls)
        summary["nsys_force_graph_gaps"] = (replay_gaps(per_replay) if per_replay else
                                            "kernel count not a multiple of the replays")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


def profiler_pass(args, engine, torch):
    """The ncu or nsys pass: the profiled range only, between cudaProfilerStart and cudaProfilerStop."""
    field = engine.force_field
    positions = engine.state.positions.detach().clone()
    edges = int(field.data["edge_index"].shape[1])
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    record = dict(pass_=args.pass_, padded_edges=edges, created_utc=datetime.now(timezone.utc).isoformat())
    if args.pass_ == "nsys-steps":
        return nsys_steps_pass(args, engine, torch, record)
    if args.pass_ == "ncu":
        torch.cuda.profiler.start()
        field(positions)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    else:
        key = force_graph_key(engine, edges)
        if key is None:
            raise SystemExit("no captured force graph to replay")
        graph = engine._graphs[key]
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for _ in range(args.calls):
            graph.replay()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        record.update(graph=str(key), replays=args.calls)
    (args.output / f"{args.pass_}_pass.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def nsys_steps_pass(args, engine, torch, record):
    """--steps MD steps between cudaProfilerStart/Stop, every stage inside an NVTX range (wrapped on the instance
    only, as part 2 of the main pass)."""
    nvtx = torch.cuda.nvtx
    field, integrator = engine.force_field, engine.integrator
    original_run, original_rebuild, original_prepare = engine._run, field.rebuild, integrator.prepare_step

    def ranged(label, fn, *a):
        nvtx.range_push(label)
        try:
            return fn(*a)
        finally:
            nvtx.range_pop()

    engine._run = lambda key: ranged(f"stage {stage_name(key).split(':')[0]}", original_run, key)
    field.rebuild = lambda positions: ranged("neighbour-list rebuild", original_rebuild, positions)
    integrator.prepare_step = lambda state: ranged("noise upload", original_prepare, state)
    rebuilds = engine.rebuilds
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(args.steps):
        nvtx.range_push("md step")
        engine.step()
        nvtx.range_pop()
    engine.flush()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    del engine._run, field.rebuild, integrator.prepare_step
    record.update(steps=args.steps, rebuilds=engine.rebuilds - rebuilds)
    (args.output / "nsys-steps_pass.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def rewrite_overrides(items):
    """{field: value} of ``FIELD=VALUE`` items, each a RewriteConfig field; "on"/"off"/"auto" or true/false."""
    from fastmace.config import RewriteConfig

    fields = {f.name: f for f in dataclasses.fields(RewriteConfig)}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or not value:
            raise ValueError(f"--rewrites {item!r}: expected FIELD=VALUE with a RewriteConfig field")
        if isinstance(fields[name].default, bool):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"--rewrites {item!r}: {name} takes true or false")
            out[name] = value.lower() == "true"
        else:
            out[name] = value
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path,
                   help="Fresh directory (main pass), or the main pass's directory (profiler passes, summaries)")
    p.add_argument("--pass", dest="pass_", choices=("main", "ncu", "nsys", "nsys-steps"), default="main",
                   help="main: the breakdown; ncu: one eager force call between cudaProfilerStart/Stop for "
                        "`ncu --profile-from-start off`; nsys: --calls force-graph replays in that range for "
                        "`nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node`")
    p.add_argument("--summarize-ncu", type=Path, help="ncu --csv --page raw --print-units base export to join")
    p.add_argument("--summarize-nsys", type=Path, help="nsys stats --report cuda_gpu_trace --format csv export")
    p.add_argument("--ncu-metrics", action="store_true", help="Print the ncu metric list (for --metrics) and exit")
    p.add_argument("--checkpoint", type=Path, help="The label's checkpoint (passes)")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl",
                   help="Benchmark name under FASTMACE_STRUCTURES (3x3x3), or a file with PATH@NXxNYxNZ")
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--timestep-fs", type=float, default=None, help="Default: IntegratorConfig's (0.5 fs)")
    p.add_argument("--velocity-seed", type=int, default=20261007, help="The research probe's Maxwell-Boltzmann seed")
    p.add_argument("--equilibrate-steps", type=int, default=300)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="RewriteConfig fields (e.g. gated_rows=on); the others stay auto")
    p.add_argument("--tf32", action="store_true",
                   help="PyTorch's TF32 matmul mode for the whole pass (TF32 screen, 29 October 2026; never default)")
    p.add_argument("--allow-busy-gpu", action="store_true", help="CPU checks of the tool only (no timing)")
    args = p.parse_args(argv)
    try:
        args.rewrites = rewrite_overrides(args.rewrites)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.ncu_metrics:
        print(",".join(NCU_METRICS))
        return None
    if args.output is None:
        raise SystemExit("--output is required")
    if args.summarize_ncu or args.summarize_nsys:
        return summarize_profiles(args)
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for a pass")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if args.backend == "hybrid":
        oeq_runtime = ab_step_time.prepare_fastmace_hybrid()  # before anything imports MACE
    else:
        oeq_runtime = None
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from torch.profiler import ProfilerActivity, profile

    label_triton_kernels()
    ab_step_time.set_tf32(torch, args.tf32)
    if not args.allow_busy_gpu and ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    if args.pass_ == "main":
        args.output.mkdir(parents=True, exist_ok=False)
    elif not (args.output / "summary.json").is_file():
        raise SystemExit(f"--pass {args.pass_} needs the main pass's directory {args.output}")
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(args.velocity_seed))
    Stationary(atoms)
    engine, plan = build_engine(args, atoms)
    field, integrator = engine.force_field, engine.integrator
    captures = [0]
    original_capture = engine._capture

    def counted_capture(key):
        captures[0] += 1
        return original_capture(key)

    engine._capture = counted_capture
    started = time.perf_counter()
    for _ in range(args.equilibrate_steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    equilibrate_s = time.perf_counter() - started

    def counters():
        return dict(rebuilds=engine.rebuilds, redos=engine.pipelined_redos,
                    early_rebuilds=engine.pipelined_early_rebuilds, graph_captures=captures[0])

    def delta(before):
        now = counters()
        return {key: now[key] - before[key] for key in now}

    if args.pass_ != "main":
        return profiler_pass(args, engine, torch)
    properties = torch.cuda.get_device_properties(0)
    clock_mhz = max_sm_clock_mhz()
    peak_tflops = (fp32_peak_tflops(properties.major, properties.minor, properties.multi_processor_count, clock_mhz)
                   if clock_mhz else None)

    # 1. Wall time of the uninstrumented loop (twice: the first window may capture new graphs).
    windows = []
    for _ in range(2):
        before = counters()
        start = time.perf_counter()
        for _ in range(args.steps):
            engine.step()
        engine.flush()
        torch.cuda.synchronize()
        windows.append(dict(ms_per_step=(time.perf_counter() - start) * 1e3 / args.steps, **delta(before),
                            edge_sizes=sorted(engine.edge_sizes_used)))
    captures_before_stages = captures[0]

    # 2. Stage timing with CUDA events.
    marks = []  # (stage, start event, end event)
    host = defaultdict(float)
    original_run, original_rebuild, original_prepare = engine._run, field.rebuild, integrator.prepare_step

    def timed(kind, fn, *a):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        s.record()
        out = fn(*a)
        e.record()
        host[kind] += time.perf_counter() - t
        marks.append((kind, s, e))
        return out

    engine._run = lambda key: timed(stage_name(key), original_run, key)
    field.rebuild = lambda positions: timed(REBUILD, original_rebuild, positions)
    integrator.prepare_step = lambda state: timed(NOISE, original_prepare, state)
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    before = counters()
    torch.cuda.synchronize()
    t = time.perf_counter()
    first.record()
    for _ in range(args.steps):
        engine.step()
    engine.flush()
    last.record()
    torch.cuda.synchronize()
    events_wall_ms = (time.perf_counter() - t) * 1e3 / args.steps
    instrumented = delta(before)
    del engine._run, field.rebuild, integrator.prepare_step  # the class methods again
    if engine._run.__func__ is not type(engine)._run or field.rebuild != original_rebuild:
        raise RuntimeError("stage wrappers not removed")
    gpu_span_ms = first.elapsed_time(last) / args.steps
    stages = defaultdict(lambda: [0, 0.0])
    for kind, s, e in marks:
        stages[kind][0] += 1
        stages[kind][1] += s.elapsed_time(e)
    covered = sum(v[1] for v in stages.values()) / args.steps
    with (args.output / "stages.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "calls", "ms_per_call", "ms_per_step", "host_ms_per_step"])
        for kind, (count, total) in sorted(stages.items(), key=lambda item: -item[1][1]):
            writer.writerow([kind, count, f"{total / count:.4f}", f"{total / args.steps:.4f}",
                             f"{host[kind] * 1e3 / args.steps:.4f}"])
        writer.writerow(["GPU idle between stages", "", "", f"{gpu_span_ms - covered:.4f}", ""])

    # 3. Force stage by kernel (eager calls of the compiled model, on the edges B last used).
    positions = engine.state.positions.detach().clone()
    edges = int(field.data["edge_index"].shape[1])
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(args.calls):
            field(positions)
        torch.cuda.synchronize()
    trace = args.output / "trace.json"
    prof.export_chrome_trace(str(trace))
    table, kernel_count = kernel_table(read_trace(trace))
    categories = defaultdict(lambda: dict(us=0.0, launches=0.0, gemm_flops=0.0, gemm_us=0.0, occupancy_us=0.0,
                                          occupancy_weight=0.0))
    rates = op_gemm_rates(table, args.calls, peak_tflops)
    with (args.output / "ops.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["us_per_call", "launches_per_call", "category", "kernel", "op", "grid", "block",
                         "registers", "shared_bytes", "est_achieved_occupancy_pct", "input_dims"])
        for (name, op, dims), row in sorted(table.items(), key=lambda item: -item[1]["us"]):
            group = category(name, op, dims, edges)
            c = categories[group]
            c["us"] += row["us"] / args.calls
            c["launches"] += row["launches"] / args.calls
            c["occupancy_us"] += row["occupancy_us"] / args.calls
            c["occupancy_weight"] += row["occupancy_weight"] / args.calls
            occ = occupancy(row)
            writer.writerow([f"{row['us'] / args.calls:.1f}", row["launches"] / args.calls, group, name, op,
                             row["grid"], row["block"], row["registers"], row["shared_bytes"],
                             "" if occ is None else f"{occ:.1f}", dims[:300]])
    with (args.output / "gemms.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "op", "gflop_per_call", "us_per_call", "invocations_per_call",
                         "achieved_tflops", "share_of_fp32_peak", "kernels", "input_dims", "input_strides"])
        for (op, dims), rate in sorted(rates.items(), key=lambda item: -item[1]["us_per_call"]):
            group = category(rate["kernels"][0], op, dims, edges)
            categories[group]["gemm_flops"] += rate["gflop_per_call"] * 1e9
            categories[group]["gemm_us"] += rate["us_per_call"]
            writer.writerow([group, op, f"{rate['gflop_per_call']:.4f}", f"{rate['us_per_call']:.1f}",
                             rate["invocations_per_call"],
                             "" if rate["achieved_tflops"] is None else f"{rate['achieved_tflops']:.3f}",
                             "" if rate["share_of_peak"] is None else f"{rate['share_of_peak']:.3f}",
                             " | ".join(rate["kernels"]), dims[:300], rate["input_strides"][:300]])
    kernel_ms = sum(c["us"] for c in categories.values()) / 1e3
    with (args.output / "categories.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "ms_per_force_call", "share_of_kernel_time", "kernels_per_call",
                         "gemm_tflops", "gemm_share_of_fp32_peak", "est_achieved_occupancy_pct"])
        for group, c in sorted(categories.items(), key=lambda item: -item[1]["us"]):
            tflops = c["gemm_flops"] / (c["gemm_us"] * 1e-6) / 1e12 if c["gemm_us"] else None
            occ = c["occupancy_us"] / c["occupancy_weight"] if c["occupancy_weight"] else None
            writer.writerow([group, f"{c['us'] / 1e3:.4f}", f"{c['us'] / 1e3 / kernel_ms:.4f}", f"{c['launches']:g}",
                             "" if tflops is None else f"{tflops:.3f}",
                             "" if tflops is None or not peak_tflops else f"{tflops / peak_tflops:.3f}",
                             "" if occ is None else f"{occ:.1f}"])
    # One more call alone: its kernels in launch order, which the ncu pass joins by position.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        field(positions)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    write_sequence(read_trace(trace), edges, args.output / "kernel_sequence.csv")

    # 4. Replays of the force graph under the profiler: kernel time against the replay span, gaps inside.
    graph_key = force_graph_key(engine, edges)
    replay_stats = None
    if graph_key is not None:
        graph = engine._graphs[graph_key]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as replay_prof:
            for _ in range(args.calls):
                graph.replay()
            torch.cuda.synchronize()
        replay_kernels = sorted((e for e in replay_prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                                key=lambda e: e.time_range.start)
        if replay_kernels:
            busy = sum(e.time_range.elapsed_us() for e in replay_kernels)
            span = replay_kernels[-1].time_range.end - replay_kernels[0].time_range.start
            listed = [(e.name, float(e.time_range.start), float(e.time_range.end)) for e in replay_kernels]
            per_replay = split_replays(listed, args.calls)
            replay_stats = dict(graph=str(graph_key), kernels_per_replay=len(replay_kernels) / args.calls,
                                kernel_ms_per_replay=busy / 1e3 / args.calls,
                                span_ms_per_replay=span / 1e3 / args.calls,
                                gaps_inside_replays=(replay_gaps(per_replay) if per_replay else
                                                     "kernel count not a multiple of the replays"))

    summary = dict(
        tool="step_breakdown", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
        python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(), packages=package_versions(), tf32=args.tf32,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, oeq_runtime=oeq_runtime,
        other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()],
        compute_capability=f"{properties.major}.{properties.minor}", sms=properties.multi_processor_count,
        max_sm_clock_mhz=clock_mhz, fp32_peak_tflops=peak_tflops,
        model=args.model, checkpoint=str(args.checkpoint), checkpoint_sha256=plan["sha256"], backend=args.backend,
        dtype="float32" if args.backend == "hybrid" else "float64", structure=structure, natoms=len(atoms),
        temperature_K=args.temperature_K, steps=args.steps, equilibrate_steps=args.equilibrate_steps,
        calls=args.calls, padded_edges=edges, plan=plan, engine_mode=engine.mode,
        load_s=plan["load_s"], setup_s=plan["setup_s"], equilibrate_s=equilibrate_s,
        wall_ms_per_step=windows[-1]["ms_per_step"], wall_windows=windows,
        graph_captures_before_stages=captures_before_stages, force_graph_replay_profile=replay_stats,
        instrumented_wall_ms_per_step=events_wall_ms, gpu_span_ms_per_step=gpu_span_ms,
        stage_ms_per_step={k: v[1] / args.steps for k, v in stages.items()},
        stage_calls={k: v[0] for k, v in stages.items()},
        gpu_idle_ms_per_step=gpu_span_ms - covered, instrumented=instrumented,
        force_graph_ms_per_replay=stages[B_STAGE][1] / max(stages[B_STAGE][0], 1),
        eager_force_kernel_ms_per_call=kernel_ms, eager_kernels_per_call=kernel_count / args.calls,
        engine_info=dict(engine.info()), force_field_info=dict(field.info()),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("wall_ms_per_step", "gpu_span_ms_per_step", "force_graph_ms_per_replay",
                                              "eager_force_kernel_ms_per_call", "stage_ms_per_step")}, indent=2))
    return summary


if __name__ == "__main__":
    main()
