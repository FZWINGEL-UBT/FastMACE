"""Tables for ``setup_compositions.py`` runs: per-stage setup seconds, cache misses and what changes between compositions.

Usage: ``setup_compositions_report.py ROOT`` (the launcher's output root); prints markdown. Stages are exclusive
times from the nested timers (a timer's total minus its timed children), grouped into the stages of the owner's
list; the first compiled call is split with torch's ``compilation_time_metrics``:

- Dynamo = ``_compile.compile_inner`` − ``OutputGraph.call_user_compiler`` (bytecode tracing, guards, and the
  abandoned first attempt ``compile_attempt_0``, shown on its own as well);
- AOTAutograd + Inductor = ``call_user_compiler`` − codegen − autotune − module load;
- the first execution = the first compiled call − ``compile_inner`` (the compiled graphs' first run, lazy CUDA
  and library initialisation).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = [
    ("load and conversion", ("load: ", "OEq: ", "build: template cache", "build: templates.private", "compile: private_copy",
                             "build: execution model", "build: ForceField.execution_model")),
    ("radial tables: fit + certify", ("radial tables: ",)),
    ("self-checks", ("rewrites: self-checks",)),
    ("OEq render + NVRTC", ("OEq streaming: ", "fused forward: render")),
    ("graph capture (incl. instantiate)", ("engine: graph capture", "engine: prepare")),
]
FIRST_CALL = "engine: first compiled force call"


def exclusive(stages_timeline):
    totals, children = defaultdict(float), defaultdict(float)
    for row in stages_timeline:
        totals[row["label"]] += row["seconds"]
        if row["parent"]:
            children[row["parent"]] += row["seconds"]
    return {label: totals[label] - children[label] for label in totals}


def stages(row):
    excl = exclusive(row["timeline"])
    out = defaultdict(float)
    first = 0.0
    for label, seconds in excl.items():
        if label == FIRST_CALL or label.startswith("engine: eager warm-up") or label.startswith("engine: neighbour"):
            first += seconds
            continue
        for group, prefixes in GROUPS:
            if any(label.startswith(prefix) for prefix in prefixes):
                out[group] += seconds
                break
        else:
            out["planner, rewrites, other host work"] += seconds
    metrics = {k: v["total_s"] for k, v in row["compile_metrics"].items()}
    inner = metrics.get("_compile.compile_inner", 0.0)
    user = metrics.get("OutputGraph.call_user_compiler", 0.0)
    codegen = metrics.get("GraphLowering.codegen", 0.0)
    autotune = metrics.get("CachingAutotuner.benchmark_all_configs", 0.0)
    load = row["extra_stages"].get("module load: PyCodeCache.load_by_key_path", {}).get("total_s", 0.0)
    out["Dynamo trace (incl. restart)"] = inner - user
    out["  of which abandoned attempt 0"] = metrics.get("compile_attempt_0", 0.0)
    out["Inductor codegen (source generation)"] = codegen
    out["Triton autotune benchmarks"] = autotune
    out["module loading (PyCodeCache)"] = load
    out["AOTAutograd + Inductor lowering + Triton compile"] = max(user - codegen - autotune - load, 0.0)
    out["first execution of the compiled call"] = max(first - inner, 0.0)
    return dict(out)


ORDER = ["load and conversion", "planner, rewrites, other host work", "self-checks", "radial tables: fit + certify",
         "OEq render + NVRTC", "Dynamo trace (incl. restart)", "  of which abandoned attempt 0",
         "AOTAutograd + Inductor lowering + Triton compile", "Inductor codegen (source generation)",
         "Triton autotune benchmarks", "module loading (PyCodeCache)", "first execution of the compiled call",
         "graph capture (incl. instantiate)"]


def load_runs(root):
    runs = {}
    for path in sorted(Path(root).glob("*/setup.json")):
        runs[path.parent.name] = json.loads(path.read_text())
    return runs


def own_files(row):
    """New cache files outside the Inductor/Triton/compile/xdg subdirectories of FASTMACE_CACHE (counted there)."""
    nested = ("inductor/", "triton/", "compile/", "xdg/")
    return {k: v for k, v in row["cache_new_files"].items()
            if not (k.startswith("FASTMACE_CACHE:") and k.split(":", 1)[1].startswith(nested))}


def main(argv=None):
    root = Path((argv or sys.argv[1:])[0])
    print("## Median stages per mode (s)")
    print()
    medians(root)
    print()
    runs = load_runs(root)
    table = {}  # (run, structure) -> row
    for name, run in runs.items():
        for row in run["structures"]:
            table[(name, row["name"])] = row
    print("## Totals per build (s): Simulation construction (build + capture) and process load")
    print()
    print("| Run | Structure | Atoms | Species | Simulation s | build | capture | first compiled call | new cache files "
          "(fastmace / inductor / triton) | FX hit/miss | AOT hit/miss |")
    print("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for (name, structure), row in table.items():
        parts, counters = row["setup_parts_s"], row["dynamo_counters"]
        files = own_files(row)
        fm = sum(1 for k in files if k.startswith("FASTMACE_CACHE:"))
        ind = sum(1 for k in files if k.startswith("TORCHINDUCTOR_CACHE_DIR:"))
        tri = sum(1 for k in files if k.startswith("TRITON_CACHE_DIR:"))
        first = row["stages"].get(FIRST_CALL, {}).get("total_s", 0.0)
        print(f"| {name} | {structure} | {row['natoms']} | {'-'.join(row['species'])} | {row['simulation_s']:.1f} | "
              f"{parts.get('build', 0):.1f} | {parts.get('capture', 0):.2f} | {first:.1f} | {fm} / {ind} / {tri} | "
              f"{counters.get('inductor.fxgraph_cache_hit', 0)}/{counters.get('inductor.fxgraph_cache_miss', 0)} | "
              f"{counters.get('aot_autograd.autograd_cache_hit', 0)}/{counters.get('aot_autograd.autograd_cache_miss', 0)} |")
    print()
    for name, run in runs.items():
        rows = run["structures"]
        print(f"## Stages, {name} (s; model load {run['model_load_s']:.1f} s once per process)")
        print()
        print("| Stage | " + " | ".join(r["name"] for r in rows) + " |")
        print("| --- |" + " ---: |" * len(rows))
        split = [stages(r) for r in rows]
        for stage in ORDER:
            print(f"| {stage} | " + " | ".join(f"{s.get(stage, 0):.2f}" for s in split) + " |")
        print("| **Simulation (build + capture)** | " + " | ".join(f"**{r['simulation_s']:.1f}**" for r in rows) + " |")
        print()
        checks = defaultdict(list)
        for r in rows:
            for label, v in r["extra_stages"].items():
                if label.startswith("self-check: "):
                    checks[label[12:]].append(v["total_s"])
        top = sorted(checks.items(), key=lambda item: -max(item[1]))[:8]
        print("Largest self-checks (s per structure): " + "; ".join(
            f"{k} " + "/".join(f"{x:.2f}" for x in v) for k, v in top))
        print()
    print("## Generated code: equal across compositions?")
    print()
    for name, run in runs.items():
        rows = run["structures"]
        if len(rows) < 2:
            continue
        print(f"### {name}")
        print()
        kinds = sorted({g["kind"] for r in rows for g in r["generated"]})
        for kind in kinds:
            sets = [{g["key"] for g in r["generated"] if g["kind"] == kind} for r in rows]
            common = set.intersection(*sets) if sets else set()
            union = set.union(*sets) if sets else set()
            print(f"- {kind}: {len(union)} distinct over {len(rows)} builds, {len(common)} shared by all; per build "
                  + ", ".join(f"{r['name']} {len(s)} ({len(s & set.union(*(t for t in sets[:i]), set()))} seen "
                              f"before)" for i, (r, s) in enumerate(zip(rows, sets))))
        print()
    for name in ("graphs", "graphs-edges"):
        graph_report(root, name)



# ---- CUDA graph topology (the ``graphs`` run's DOT dumps)

import re  # noqa: E402

DOT_NODE = re.compile(r'^"(?P<id>graph_\d+_node_(?P<n>\d+))"\[.*?label="\{(?P<type>[A-Z_]+)', re.S)
# The DOT text escapes the launch shape as \<\<\<grid,block,smem\>\>\> (\x5c is the backslash).
DOT_KERNEL = re.compile(r"\| (?P<name>\S+?)(?:\x5c<){3}(?P<grid>[\d,()]+?),(?P<block>[\d,()]+?),(?P<smem>\d+)"
                        r"(?:\x5c>){3}\}")
DOT_EDGE = re.compile(r'^"(graph_\d+_node_\d+)" -> "(graph_\d+_node_\d+)"', re.M)


def read_dot(path):
    """Nodes (in capture order: type, kernel name, launch shape) and edges of a cudaGraphDebugDotPrint file."""
    text = Path(path).read_text(errors="replace")
    nodes = {}
    for block in text.split('\n\n'):
        block = block.strip()
        match = DOT_NODE.match(block)
        if not match:
            continue
        node = dict(type=match.group("type"))
        kernel = DOT_KERNEL.search(block)
        if kernel:
            node.update(name=kernel.group("name"), grid=kernel.group("grid"), block=kernel.group("block"),
                        smem=int(kernel.group("smem")))
        nodes[match.group("id")] = node
    index = {key: i for i, key in enumerate(sorted(nodes, key=lambda k: int(k.rsplit("_", 1)[1])))}
    ordered = [nodes[key] for key in sorted(index, key=index.get)]
    edges = sorted((index[a], index[b]) for a, b in DOT_EDGE.findall(text) if a in index and b in index)
    return ordered, edges


def compare_graphs(a, b):
    """What cudaGraphExecUpdate would have to accept to move graph a's executable to graph b."""
    (na, ea), (nb, eb) = a, b
    out = dict(nodes=(len(na), len(nb)), edges=(len(ea), len(eb)))
    if len(na) != len(nb) or ea != eb:
        out["topology"] = "different"
        return out
    out["topology"] = "same"
    out["types_differ"] = sum(x["type"] != y["type"] for x, y in zip(na, nb))
    kernels = [(x, y) for x, y in zip(na, nb) if x["type"] == "KERNEL" and y["type"] == "KERNEL"]
    out["kernels"] = len(kernels)
    out["function_differs"] = [(x.get("name"), y.get("name")) for x, y in kernels if x.get("name") != y.get("name")]
    out["launch_differs"] = sum((x.get("grid"), x.get("block"), x.get("smem")) != (y.get("grid"), y.get("block"),
                                                                                  y.get("smem")) for x, y in kernels)
    return out


def graph_report(root, name="graphs"):
    run = Path(root) / name
    if not (run / "setup.json").exists():
        return
    data = json.loads((run / "setup.json").read_text())
    graphs = {}
    for row in data["structures"]:
        for g in row.get("graphs", []):
            kind = "force (B)" if str(g["key"]).startswith("('B'") else str(g["key"])
            graphs[(row["name"], kind)] = (read_dot(run / g["file"]), g["key"])
    print(f"## CUDA graphs ({name}): could cudaGraphExecUpdate move an executable graph to another structure?")
    print()
    print("| Graph | From | To | Keys | Nodes | Edges | Topology | Node types differ | Kernels | Kernel function "
          "differs | Launch shape differs |")
    print("| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    names = [row["name"] for row in data["structures"]]
    for kind in sorted({k for _, k in graphs}):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if (a, kind) not in graphs or (b, kind) not in graphs:
                    continue
                (ga, ka), (gb, kb) = graphs[(a, kind)], graphs[(b, kind)]
                c = compare_graphs(ga, gb)
                print(f"| {kind} | {a} | {b} | {ka} → {kb} | {c['nodes'][0]} / {c['nodes'][1]} | {c['edges'][0]} / "
                      f"{c['edges'][1]} | {c['topology']} | {c.get('types_differ', '–')} | {c.get('kernels', '–')} | "
                      f"{len(c['function_differs']) if 'function_differs' in c else '–'} | "
                      f"{c.get('launch_differs', '–')} |")
                if c.get("function_differs"):
                    print(f"|  | | | changed functions: " + "; ".join(
                        f"{x[:50]} → {y[:50]}" for x, y in c["function_differs"][:4]) + " | | | | | | | |")
    print()



def medians(root):
    """Median stage seconds per mode: cold and warm single builds, and builds 2-8 of the chains (same process)."""
    import statistics

    runs = load_runs(root)
    modes = {"cold (fresh process, empty cache)": [r for n, run in runs.items() if n.startswith("cold-")
                                                    for r in run["structures"]],
             "warm (fresh process, own warm cache)": [r for n, run in runs.items() if n.startswith("warm-")
                                                      for r in run["structures"]],
             "same process, cache cold for this composition": runs.get("chain-cold", {}).get("structures", [])[1:],
             "same process, warm cache": runs.get("chain-warm", {}).get("structures", [])[1:]}
    print("| Stage | " + " | ".join(modes) + " |")
    print("| --- |" + " ---: |" * len(modes))
    split = {mode: [stages(r) for r in rows] for mode, rows in modes.items()}
    for stage in ORDER:
        print(f"| {stage} | " + " | ".join(f"{statistics.median(s.get(stage, 0) for s in rows):.1f}" if rows else "–"
                                         for rows in split.values()) + " |")
    print("| **Simulation (build + capture), median** | " + " | ".join(
        f"**{statistics.median(r['simulation_s'] for r in rows):.1f}**" if rows else "–" for rows in modes.values())
        + " |")
    print("| range | " + " | ".join(f"{min(r['simulation_s'] for r in rows):.1f}–{max(r['simulation_s'] for r in rows):.1f}"
                                    if rows else "–" for rows in modes.values()) + " |")
    loads = {"cold": [run["model_load_s"] for n, run in runs.items() if n.startswith("cold-")],
             "warm": [run["model_load_s"] for n, run in runs.items() if n.startswith("warm-")]}
    print()
    print("Model load once per process (`fm.load`, pristine e3nn model plus the OEq runtime), median s: "
          + ", ".join(f"{k} {statistics.median(v):.1f}" for k, v in loads.items() if v))


if __name__ == "__main__":
    main()
