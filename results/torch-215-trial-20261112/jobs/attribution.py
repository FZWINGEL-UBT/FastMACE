"""Where an MD step's GPU time goes, by source line (torch 2.15 CUDA-graph annotations; conductor, 12 November 2026).

Inputs from profile_point.py with PYSTACKS=1: the Cuspy Chrome trace (trace-<tag>.json.gz), every captured node's
Python stack (pystacks-<tag>.json, keyed (graph id << 32) | graph node id) and the run's Inductor cache (generated
code with comment_origin: each fused kernel's FX source nodes, ATen ops and model source lines).

Per profiled step it reports:
  - time by kernel class (ours: NVRTC / our Triton / CUTLASS; Inductor; cuBLAS; cuEquivariance; memset / copy;
    other = eager ATen) and by attributed source site (the innermost frame outside torch / Inductor / Triton
    runtime, plus its enclosing function), ranked;
  - small kernels (< SMALL_US), copies / cat / clone / transpose / zero-fills, Inductor kernels by model operation;
  - gaps between consecutive GPU events inside each step;
  - bounds for the ledger rows species-only-kernels, i0-copies-zero-fill, launch-memsets, launch-combo-sizerule.

    python attribution.py TRACE.json.gz PYSTACKS.json INDUCTOR_CACHE_DIR STEPS OUT.json
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

trace_path, stacks_path, inductor_dir, steps, out_path = sys.argv[1:6]
STEPS = int(steps)
SMALL_US = 5.0
sys.path.insert(0, str(Path(__file__).parent))
from kernel_category import category  # noqa: E402

events = json.loads(gzip.open(trace_path).read())
events = events["traceEvents"] if isinstance(events, dict) else events
gpu = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
             key=lambda e: float(e["ts"]))
stacks = json.loads(Path(stacks_path).read_text())

# frames of the runtime, not of the model: torch, Triton's own package, Inductor's generated launchers. The paths are
# normalized to start at the package ("triton/runtime/...", but "fastmace/kernels/triton/..." is ours)
SKIP_PREFIX = ("torch/", "triton/", "functorch/", "<string>", "contextlib.py", "cuequivariance_ops_torch/")


def frames(e):
    a = e.get("args", {})
    if a.get("graph node id") is None or a.get("graph id") is None:
        return ["(eager, outside the CUDA graphs: neighbour-list rebuild steps)"]
    s = stacks.get(str((int(a["graph id"]) << 32) | int(a["graph node id"])))
    if s is None:
        s = stacks.get(str(a["graph node id"]))
    if s is None:
        return ["(graph node without stack)"]
    s = s if isinstance(s, str) else "\n".join(map(str, s))
    return [re.sub(r".*/(site-packages|src-[0-9a-f]+|src)/", "", f.strip()) for f in s.splitlines() if f.strip()]


def site(fr):
    """(innermost own/model frame, the next frame out): file:line:function of the launch and its caller."""
    own = [f for f in fr if not f.startswith(SKIP_PREFIX)]
    if not own:
        return ("(no stack)" if not fr else fr[0]), ""
    return own[0], (own[1] if len(own) > 1 else "")


def klass(e):
    n = e["name"]
    if e["cat"] == "gpu_memset" or "memset" in n.lower() or "FillFunctor" in n:
        return "memset/fill"
    if e["cat"] == "gpu_memcpy" or "memcpy" in n.lower() or "copy_kernel" in n or "direct_copy" in n:
        return "copy"
    c = category(n)
    return {"nvrtc": "ours: NVRTC", "triton": "ours: Triton", "cutlass": "ours: CUTLASS", "inductor": "Inductor",
            "cublas": "cuBLAS", "cueq": "cuEquivariance", "memory": "memset/fill"}.get(c, "other (eager ATen)")


OURS = ("ours: NVRTC", "ours: Triton", "ours: CUTLASS")
LAYOUT = {"cat", "view", "clone", "_unsafe_view", "slice", "transpose", "permute", "unsqueeze", "squeeze", "expand",
          "zeros", "full", "new_zeros", "select", "alias", "slice_backward", "slice_scatter", "_to_copy", "copy",
          "constant_pad_nd", "full_like", "zeros_like", "as_strided", "reshape", "contiguous", "t"}


def ops_of(name):
    o = origin.get(name)
    return {x.replace("aten.", "") for x in o["aten"]} if o else None


def layout_only(name):
    """An Inductor kernel whose ATen origin is only data movement (custom fastmace.* ops it feeds aside)."""
    ops = ops_of(name)
    return bool(ops) and not {x for x in ops if not x.startswith("fastmace.")} - LAYOUT


def model_line(name):
    """The first model line in an Inductor kernel's origin (not the compiled module's own call line)."""
    o = origin.get(name)
    lines = [ln for ln in sorted(o["lines"]) if not ln.startswith("fastmace/model/compile.py")] if o else []
    return lines[0] if lines else None

# Inductor origins from the generated code (comment_origin): kernel name -> ATen ops, source nodes, model lines
origin = collections.defaultdict(lambda: {"aten": set(), "nodes": set(), "lines": set()})
for py in Path(inductor_dir).rglob("*.py"):
    try:
        text = py.read_text(errors="replace")
    except OSError:
        continue
    for m in re.finditer(r"((?:^#.*\n){1,80})^(triton_\w+) = async_compile\.triton\(", text, re.M):
        head, name = m.group(1), m.group(2)
        o = origin[name]
        for a in re.findall(r"Original ATen: \[([^\]]*)\]", head):
            o["aten"].update(x.strip() for x in a.split(",") if x.strip())
        for a in re.findall(r"Source Nodes: \[([^\]]*)\]", head):
            o["nodes"].update(x.strip() for x in a.split(",") if x.strip())
        body = text[m.end():m.end() + 20000].split("''', device_str", 1)[0]
        for f in re.findall(r"((?:mace|fastmace)/[\w/]+\.py):(\d+)", head + body):
            o["lines"].add(f"{f[0]}:{f[1]}")
        for f, ln in re.findall(r'File "[^"]*?((?:mace|fastmace)/[\w/]+\.py)", line (\d+)', head + body):
            o["lines"].add(f"{f}:{ln}")

per_step = lambda us: round(us / STEPS, 2)  # noqa: E731
total = sum(float(e["dur"]) for e in gpu)
by_class, by_site, by_kernel = collections.Counter(), collections.Counter(), collections.Counter()
site_class, site_kernels, counts = {}, collections.defaultdict(collections.Counter), collections.Counter()
small = collections.Counter()
small_n = collections.Counter()
copies = collections.Counter()
inductor_ops = collections.Counter()
inductor_ops_detail = collections.defaultdict(set)
COPYISH = re.compile(r"clone|cat|transpose|permute|copy|fill|zeros|full|expand|constant_pad|index_put|new_zeros")
for e in gpu:
    d = float(e["dur"])
    k = klass(e)
    s, caller = site(frames(e))
    if k == "Inductor" and model_line(e["name"]):
        s, caller = f"Inductor: {model_line(e['name'])}", s
    key = (f"{s}  <-  {caller}" if caller else s) + f"  [{k}]"
    by_class[k] += d
    by_site[key] += d
    site_class[key] = k
    site_kernels[key][e["name"][:90]] += d
    by_kernel[(k, e["name"][:120])] += d
    counts[(k, e["name"][:120])] += 1
    if d < SMALL_US:
        small[key] += d
        small_n[key] += 1
    if k in ("memset/fill", "copy") or (k == "Inductor" and COPYISH.search(e["name"])) or \
            (k == "other (eager ATen)" and re.search(r"copy|fill|cat|index|gather|scatter", e["name"])):
        copies[(k, e["name"][:100], s)] += d
    if k == "Inductor":
        o = origin.get(e["name"])
        ops = sorted(o["aten"])[:6] if o else ["(origin not found)"]
        label = ", ".join(x.replace("aten.", "") for x in ops)
        inductor_ops[label] += d
        if o:
            inductor_ops_detail[label].update(sorted(o["lines"])[:4])

# gaps: GPU idle time between consecutive events inside a step (the window is STEPS back-to-back graph replays)
gaps, gap_rows = 0.0, collections.Counter()
for a, b in zip(gpu, gpu[1:]):
    g = float(b["ts"]) - (float(a["ts"]) + float(a["dur"]))
    if 0 < g < 1000:  # > 1 ms: between steps (host work, the integrator's own launches), counted apart
        gaps += g
        gap_rows[(a["name"][:50], b["name"][:50])] += g
wall = float(gpu[-1]["ts"]) + float(gpu[-1]["dur"]) - float(gpu[0]["ts"])
step_gaps = [float(b["ts"]) - (float(a["ts"]) + float(a["dur"])) for a, b in zip(gpu, gpu[1:])]
big_gaps = sorted(g for g in step_gaps if g >= 1000)


def rows(counter, n=25, cls=None):
    return [{"us_per_step": per_step(v), "pct": round(100 * v / total, 2), "key": k if isinstance(k, str) else list(k),
             **({"class": cls[k]} if cls else {})} for k, v in counter.most_common(n)]


def match(pattern, where=("Inductor", "other (eager ATen)", "cuBLAS", "memset/fill", "copy")):
    rx = re.compile(pattern)
    hit = collections.Counter()
    for (k, n), v in by_kernel.items():
        if k in where and rx.search(n):
            hit[n] += v
    return {"us_per_step": per_step(sum(hit.values())), "pct": round(100 * sum(hit.values()) / total, 2),
            "kernels": [[n, per_step(v)] for n, v in hit.most_common(12)]}


def by_origin(test):
    hit = collections.Counter({n: v for (k, n), v in by_kernel.items() if k == "Inductor" and test(n)})
    return {"us_per_step": per_step(sum(hit.values())), "pct": round(100 * sum(hit.values()) / total, 2),
            "kernels": [[n, per_step(v), sorted(ops_of(n) or [])] for n, v in hit.most_common(14)]}


species = by_origin(lambda n: bool((ops_of(n) or set()) & {"argmax", "cumsum"}))
gemv = sum(v for (k, n), v in by_kernel.items() if "gemv2T" in n)
species["us_per_step"] = round(species["us_per_step"] + per_step(gemv), 2)
species["pct"] = round(species["pct"] + 100 * gemv / total, 2)
species["kernels"].append(["gemv2T (cuBLAS, node_attrs x reference energies)", per_step(gemv), []])
layout = by_origin(layout_only)
ledger = {
    "species-only-kernels (Inductor kernels with argmax / cumsum in their origin, + the node_attrs gemv)": species,
    "layout-only Inductor kernels (origin only cat / view / clone / transpose / slice / zeros ...)": layout,
    "species-only-kernels (old name regex)": match(r"argmax|cumsum|arange_gather|arange_mul_unsqueeze|add_mul_unsqueeze|gemv2T|"
                                  r"triton_\w*clone_\d+$|index_5$"),
    "i0-copies-zero-fill (fills, clones, cat / transpose copies)": match(r"memset|FillFunctor|fill|zeros|clone|"
                                                                         r"transpose|_cat_|cat_|copy"),
    "launch-memsets (memsets and eager fills)": match(r"memset|FillFunctor|fill_", ("memset/fill", "other (eager ATen)",
                                                                                    "copy")),
    "launch-combo-sizerule (small Inductor kernels + gaps)": {
        "small_inductor_us_per_step": per_step(sum(v for (k, n), v in by_kernel.items() if k == "Inductor"
                                                   and v / max(counts[(k, n)], 1) < SMALL_US)),
        "small_inductor_launches_per_step": round(sum(c for (k, n), c in counts.items() if k == "Inductor"
                                                      and by_kernel[(k, n)] / c < SMALL_US) / STEPS, 1),
        "gaps_us_per_step": per_step(gaps)},
}

summary = {
    "trace": trace_path, "steps": STEPS, "gpu_events": len(gpu), "events_per_step": round(len(gpu) / STEPS, 1),
    "kernel_us_per_step": per_step(total), "window_wall_us_per_step": per_step(wall),
    "in_step_gap_us_per_step": per_step(gaps), "between_step_gaps_us": [round(g, 1) for g in big_gaps][:12],
    "graph_events_with_stack": sum(1 for e in gpu if frames(e)),
    "by_class": {k: {"us_per_step": per_step(v), "pct": round(100 * v / total, 2)} for k, v in by_class.most_common()},
    "not_ours_us_per_step": per_step(sum(v for k, v in by_class.items() if k not in OURS)),
    "by_site": [dict(r, kernels=[[n, per_step(v)] for n, v in site_kernels[r["key"]].most_common(4)])
                for r in rows(by_site, 40, site_class)],
    "not_ours_by_site": [dict(r, kernels=[[n, per_step(v)] for n, v in site_kernels[r["key"]].most_common(4)])
                         for r in rows(collections.Counter({k: v for k, v in by_site.items()
                                                            if site_class[k] not in OURS}), 30, site_class)],
    "small_kernels": {"threshold_us": SMALL_US,
                      "us_per_step": per_step(sum(small.values())),
                      "count_per_step": round(sum(small_n.values()) / STEPS, 1),
                      "by_site": [dict(r, count_per_step=round(small_n[r["key"]] / STEPS, 1))
                                  for r in rows(small, 20)]},
    "copies_fills": [{"us_per_step": per_step(v), "class": k[0], "kernel": k[1], "site": k[2]}
                     for k, v in copies.most_common(30)],
    "copies_fills_us_per_step": per_step(sum(copies.values())),
    "inductor_by_aten": [{"us_per_step": per_step(v), "aten": k, "lines": sorted(inductor_ops_detail[k])[:4]}
                         for k, v in inductor_ops.most_common(30)],
    "inductor_origins_found": sum(1 for n in {e["name"] for e in gpu if klass(e) == "Inductor"} if n in origin),
    "inductor_kernels": len({e["name"] for e in gpu if klass(e) == "Inductor"}),
    "top_gaps": [{"us_per_step": per_step(v), "after": k[0], "before": k[1]} for k, v in gap_rows.most_common(15)],
    "ledger_bounds": ledger,
}
Path(out_path).write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ("events_per_step", "kernel_us_per_step", "window_wall_us_per_step",
                                          "in_step_gap_us_per_step", "graph_events_with_stack", "by_class",
                                          "not_ours_us_per_step", "inductor_origins_found", "inductor_kernels")},
                 indent=1))
