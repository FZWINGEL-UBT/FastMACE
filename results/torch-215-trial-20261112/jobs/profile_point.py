"""PyTorch 2.15 trial (12 November 2026): tools/bench/capacity_scan.py ``point`` plus a profiled window.

After the point's timed steady run, PROFILE_STEPS (20) further MD steps run under torch.profiler; the window's
GPU kernels go to ``kernels-<tag>.json`` (per kernel name: count, total and mean microseconds per step, and a
category) and the Chrome trace to ``trace-<tag>.json.gz``. The point's own timing is unaffected (the window comes
after it), but the point JSON of a profile process is not used as an A/B sample.

REFUSE_FALLBACK=1: the Simulation refuses fallbacks (FallbackConfig(fallback="refuse")), so an A/B point never
times a fallback tier. PROFILE_STEPS=0: no profiled window.

PYSTACKS=1 (2.15 only, needs cupti-python): every torch.cuda.graph capture gets enable_annotations=True and
annotation_config={"record_py_stacks": True}, autograd multithreading is off (the API requires it), and the window
is profiled with the Cuspy backend (torch.profiler.CuspyConfig: the CUPTI subscription taken by the stack
recording keeps Kineto from collecting GPU activity). The trace is exported with
cuda_graph_annotations=get_kernel_annotations(), and the recorded launch stacks go to ``pystacks-<tag>.json``
with the share of the window's replayed kernels that map to a Python frame.

Usage: the capacity_scan point arguments; env PROFILE_TAG, PROFILE_STEPS, PYSTACKS, MEMORY_FRACTION.
"""
import collections
import gzip
import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["FM_SRC"]) / "tools" / "bench"))
import capacity_scan as cs  # noqa: E402

import fastmace as fm  # noqa: E402
import fastmace.model.compile as compile_  # noqa: E402

TAG = os.environ.get("PROFILE_TAG", "x")
STEPS = int(os.environ.get("PROFILE_STEPS", "20"))
PYSTACKS = os.environ.get("PYSTACKS") == "1"
OUT = Path(os.environ["PROFILE_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
info = {"tag": TAG, "steps": STEPS, "pystacks": PYSTACKS}

if os.environ.get("DUMP_SOURCES"):  # every NVRTC source this process compiles, for the offline SASS census
    import hashlib

    from fastmace.kernels.oeq_streaming import runtime as _runtime

    _compile = _runtime.compile_cubin
    _dump = Path(os.environ["DUMP_SOURCES"])
    _dump.mkdir(parents=True, exist_ok=True)

    def _dumping_compile(source, capability, extra_options=(), suffix=""):
        name = re.search(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)', source)
        tag = hashlib.sha256(source.encode()).hexdigest()[:10]
        (_dump / f"{name.group(1) if name else 'kernel'}-sm{capability[0]}{capability[1]}{suffix}-{tag}.cu").write_text(source)
        return _compile(source, capability, extra_options, suffix)

    _runtime.compile_cubin = _dumping_compile

original_configure = compile_.configure_allocator


def configure_allocator(*args, **kwargs):
    result = original_configure(*args, **kwargs)
    import torch

    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMORY_FRACTION", "0.9")))
    return result


compile_.configure_allocator = configure_allocator

if PYSTACKS:
    import torch

    torch.autograd.grad_mode.set_multithreading_enabled(False)
    _graph = torch.cuda.graph

    class annotated_graph(_graph):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("enable_annotations", True)
            kwargs.setdefault("annotation_config", {"record_py_stacks": True})
            super().__init__(*args, **kwargs)

    torch.cuda.graph = annotated_graph
    info["captures_annotated"] = True


from kernel_category import category  # noqa: E402,F401  (kept importable without fastmace for analyze.py)


def profile_window(sim):
    import torch
    from torch.profiler import ProfilerActivity, profile

    if PYSTACKS:
        from torch.profiler import CuspyConfig, ProfilerActivityConfig

        activities = [ProfilerActivity.CPU,
                      {ProfilerActivity.CUDA: ProfilerActivityConfig(profiler_configs=[CuspyConfig()])}]
    else:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with profile(activities=activities) as prof:
        original_run(sim, steps=STEPS)
        torch.cuda.synchronize()
    trace = OUT / f"trace-{TAG}.json"
    kwargs = {}
    annotations = None
    if PYSTACKS:
        from torch.cuda import graph_annotations as ga

        annotations = ga.get_kernel_annotations()
        kwargs["cuda_graph_annotations"] = annotations
        info["annotation_entries"] = len(annotations) if annotations is not None else None
    prof.export_chrome_trace(str(trace), **kwargs)
    gz = Path(str(trace) + ".gz")
    if trace.exists():  # Kineto writes plain JSON; with Cuspy, 2.15 writes <path>.gz instead
        with open(trace, "rb") as f, gzip.open(gz, "wb") as g:
            g.write(f.read())
        trace.unlink()
    events = json.loads(gzip.open(gz).read())
    events = events["traceEvents"] if isinstance(events, dict) else events
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    table = collections.defaultdict(lambda: [0, 0.0])
    for e in kernels:
        row = table[e["name"]]
        row[0] += 1
        row[1] += float(e.get("dur", 0))
    rows = [{"name": k, "category": category(k), "count_per_step": v[0] / STEPS, "us_per_step": v[1] / STEPS,
             "mean_us": v[1] / max(v[0], 1)} for k, v in table.items()]
    rows.sort(key=lambda r: -r["us_per_step"])
    totals = collections.defaultdict(float)
    for r in rows:
        totals[r["category"]] += r["us_per_step"]
    info["kernel_events"] = len(kernels)
    info["graph_node_kernels"] = sum(1 for e in kernels if "graph node id" in e.get("args", {}))
    info["us_per_step_by_category"] = dict(totals)
    info["us_per_step_total"] = sum(totals.values())
    (OUT / f"kernels-{TAG}.json").write_text(json.dumps({"info": info, "kernels": rows}, indent=1))
    if PYSTACKS:
        try:
            from torch.cuda import graph_annotations as ga

            stacks = ga.get_kernel_py_stacks()
            info["py_stack_entries"] = len(stacks) if stacks is not None else None
            sample = {}
            cap = int(os.environ.get("PYSTACKS_MAX", "0"))  # 0: every captured node (was a 400-entry sample)
            for i, (k, v) in enumerate(stacks.items() if hasattr(stacks, "items") else []):
                if cap and i >= cap:
                    break
                sample[str(k)] = v if isinstance(v, (str, list, dict)) else repr(v)
            (OUT / f"pystacks-{TAG}.json").write_text(json.dumps(sample, indent=1, default=repr))
            keys = list(stacks.keys()) if hasattr(stacks, "keys") else []
            info["py_stack_key_examples"] = [repr(k) for k in keys[:5]]
            info["py_stack_value_example"] = repr(stacks[keys[0]])[:1500] if keys else None
            flat = set()
            for k in keys:
                flat.add(repr(k))
                if isinstance(k, tuple):
                    flat.add(repr(k[-1]))
                    flat.add(repr(tuple(k[-2:])))
            pairs = [(e["args"].get("graph id"), e["args"].get("graph node id")) for e in kernels
                     if e.get("args", {}).get("graph node id") is not None]
            info["replayed_kernels_with_node_id"] = len(pairs)
            info["replayed_kernels_with_stack_by_node_id"] = sum(1 for g, n in pairs if repr(n) in flat)
            info["replayed_kernels_with_stack_by_graph_and_node"] = sum(1 for g, n in pairs if repr((g, n)) in flat)
            annotated = [e for e in kernels if any("annot" in a.lower() or "stack" in a.lower() for a in e.get("args", {}))]
            info["kernels_with_annotation_args_in_trace"] = len(annotated)
            if annotated:
                info["annotated_example"] = {"name": annotated[0]["name"][:200], "args": annotated[0]["args"]}
        except BaseException as e:  # noqa: BLE001
            info["py_stack_error"] = f"{type(e).__name__}: {e}"[:2000]
            info["py_stack_frames"] = traceback.format_exc().splitlines()[-15:]
        (OUT / f"kernels-{TAG}.json").write_text(json.dumps({"info": info, "kernels": rows}, indent=1, default=repr))
    print("profile", TAG, json.dumps({k: v for k, v in info.items() if k != "annotated_example"}, default=repr),
          flush=True)


if os.environ.get("REFUSE_FALLBACK") == "1":  # A/B points: a broken tier raises instead of timing a fallback
    from fastmace.config import FallbackConfig

    _init = fm.Simulation.__init__

    def _refusing_init(self, *args, **kwargs):
        kwargs.setdefault("fallback", FallbackConfig(fallback="refuse"))
        _init(self, *args, **kwargs)

    fm.Simulation.__init__ = _refusing_init

original_run = fm.Simulation.run
calls = {"n": 0}


def run(self, *args, **kwargs):
    result = original_run(self, *args, **kwargs)
    calls["n"] += 1
    if calls["n"] == 2 and STEPS > 0:  # after warm-up and steady
        try:
            profile_window(self)
        except BaseException as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"[:3000]
            info["frames"] = traceback.format_exc().splitlines()[-25:]
            (OUT / f"kernels-{TAG}.json").write_text(json.dumps({"info": info}, indent=1, default=repr))
            print("profile FAILED", info["error"][:500], flush=True)
    return result


fm.Simulation.run = run

if __name__ == "__main__":
    sys.exit(cs.main())
