"""2-T13 measurements: ``fm.calc`` compiles and time per call, MH-1 on NaCl 2x2x2 (local RTX 5060 Ti).

    python calc_compile.py lbfgs OUT.json   # LBFGS as 1b-T18: per-call seconds, dynamo graphs, edge counts
    python calc_compile.py ab OUT.json      # static against dynamic edges at one edge count, same process

``lbfgs`` runs whichever FastMACE is on PYTHONPATH (main for "before", the branch for "after").

``ab`` needs the branch. One calculator on the rattled start structure; 12 blocks, each timing both
sides in alternating order. Before each side, ``torch._dynamo.reset()`` drops the compiled code, and
the force field's ``dynamic_edges`` attribute decides whether its edges are marked dynamic, so the
static side compiles static graphs and the dynamic side dynamic ones (warm Inductor caches make
these recompiles short; they are not timed). Two timings per side: ``calc`` (``calculate`` with the
neighbour-list rebuild, what an ASE optimiser pays) and ``model`` (the force field's call alone).
Measurement only: nothing here is a library option.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import fastmace as fm  # before MACE, so the planner may choose the hybrid backend
import numpy as np
import torch
from ase.calculators.calculator import all_changes
from torch._dynamo.utils import counters

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tests"))
from _support.structures import nacl  # noqa: E402

BLOCKS, CALLS, WARMUP = 12, 100, 5


def start():
    return nacl(repeat=2, rattle=0.08, seed=21)


def lbfgs():
    from ase.optimize import LBFGS

    calc = fm.calc("mh1", head="matpes_r2scan")
    calls, graphs, edges, calculate = [], [], [], calc.calculate

    def timed(*args, **kwargs):
        begin = time.perf_counter()
        calculate(*args, **kwargs)
        calls.append(time.perf_counter() - begin)
        graphs.append(counters["stats"]["unique_graphs"])
        edges.append(int(calc._evaluator.neighbours.count))

    calc.calculate = timed
    atoms = start()
    atoms.calc = calc
    optimizer = LBFGS(atoms, logfile=None)
    converged = optimizer.run(fmax=2e-3, steps=300)
    plan = calc.plan
    compiles = sum(1 for before, after in zip([0] + graphs, graphs) if after > before)
    return dict(converged=bool(converged), steps=optimizer.nsteps, calls=len(calls), call_seconds=calls,
                graphs_after_call=graphs, edges=edges, distinct_edge_counts=len(set(edges)),
                calls_that_compiled=compiles, first_call=calls[0], second_call=calls[1],
                median_rest=statistics.median(calls[2:]), backend=plan.engine.backend,
                dynamic_edges=getattr(plan.engine, "dynamic_edges", None))


def _time_side(calc, atoms, dynamic):
    torch._dynamo.reset()
    evaluator = calc._evaluator
    evaluator.dynamic_edges = dynamic
    graphs = counters["stats"]["unique_graphs"]
    begin = time.perf_counter()
    for _ in range(WARMUP):
        calc.calculate(atoms, ["energy", "forces", "stress"], all_changes)
    compile_seconds = time.perf_counter() - begin
    timings = {"calc": [], "model": []}
    for _ in range(CALLS):
        begin = time.perf_counter()
        calc.calculate(atoms, ["energy", "forces", "stress"], all_changes)
        timings["calc"].append(time.perf_counter() - begin)
    positions = evaluator._prepare_one_off(atoms)
    for _ in range(CALLS):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        evaluator(positions)
        torch.cuda.synchronize()
        timings["model"].append(time.perf_counter() - begin)
    return dict(compile_seconds=compile_seconds, graphs_added=counters["stats"]["unique_graphs"] - graphs,
                calc=statistics.median(timings["calc"]), model=statistics.median(timings["model"]))


def ab():
    calc = fm.calc("mh1", head="matpes_r2scan")
    atoms = start()
    calc.calculate(atoms, ["energy", "forces", "stress"], all_changes)  # plan and build
    blocks = []
    for block in range(BLOCKS):
        order = (False, True) if block % 2 == 0 else (True, False)
        sides = {("dynamic" if dynamic else "static"): _time_side(calc, atoms, dynamic) for dynamic in order}
        sides["order"] = ["dynamic" if dynamic else "static" for dynamic in order]
        blocks.append(sides)
        print(json.dumps(sides), flush=True)
    ratios = {kind: [b["dynamic"][kind] / b["static"][kind] for b in blocks] for kind in ("calc", "model")}
    windows = {kind: [float(np.mean(values[i:i + 4])) for i in range(0, BLOCKS, 4)]
               for kind, values in ratios.items()}
    return dict(blocks=blocks, calls_per_side=CALLS, edges=int(calc._evaluator.neighbours.count),
                backend=calc.plan.engine.backend, ratio_dynamic_over_static=ratios,
                mean_ratio={kind: float(np.mean(values)) for kind, values in ratios.items()},
                median_ratio={kind: float(np.median(values)) for kind, values in ratios.items()},
                four_block_windows=windows)


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    result = {"lbfgs": lbfgs, "ab": ab}[mode]()
    result.update(mode=mode, python=sys.executable, torch=torch.__version__, fastmace=str(Path(fm.__file__).parent),
                  gpu=torch.cuda.get_device_name(0))
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("call_seconds", "graphs_after_call", "edges", "blocks")}, indent=1))
