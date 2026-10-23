"""Rule 12's estimate against the measured GPU peak (plan task 5-T03; SPEC §9.3 rule 12, decision 51).

``run`` measures, for each label x structure x ensemble x tier, the peak of ``torch.cuda.max_memory_allocated``
over the build, the capture and ``--steps`` steps of the minimal planner's engine, each case in its own child
process (a fresh allocator and one compile family, SPEC §8.7). Every case writes ``<output>/<case>.json`` with
the plan's prediction, its parts, and the measured peak. The prediction is split into the part ``SAFETY``
multiplies (activations and rewrite terms, ``scaled``) and the rest (``fixed``) by planning twice, with
``SAFETY`` 1 and 2, so ``fit`` can choose the one global factor (SPEC §9.3 rule 12 as amended 23 October 2026).

``fit`` reads the case files and prints the markdown table of ``docs/validation/5-T03.md``: per case the
measured peak, the prediction at the library's ``SAFETY``, their ratio, and the smallest factor that keeps
that case's prediction at or above its peak; then the fitted factor (the largest of those over the judged
cases, the graph tier) and the ratios it gives. Pass: no ratio below 1, none above 1.5.

Untimed. GPU runs go through ``tools/ci/local_gpu.py run`` (local) or ``tools/colab`` (the large models).

    python tools/memory_peaks.py run --output results/5-T03-<date> --labels mh1-r2scan --structures NaCl NbCl5
    python tools/memory_peaks.py fit results/5-T03-<date>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENSEMBLES = ("nvt", "npt")
MODES = ("graph", "compiled", "eager")
JUDGED_MODE = "graph"  # the conductor's answer (23 October 2026): compiled and eager peaks are recorded only
PASS_RATIO = 1.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="measure the cases, one child process each")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--labels", nargs="+", default=None, help="default: every label of fastmace.model.labels")
    run.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    run.add_argument("--repeat", default="3x3x3")
    run.add_argument("--ensembles", nargs="+", choices=ENSEMBLES, default=list(ENSEMBLES))
    run.add_argument("--modes", nargs="+", choices=MODES, default=[JUDGED_MODE])
    run.add_argument("--precision", choices=("float32", "float64"), default=None,
                     help="the ForceField's force precision (default: its own)")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--temperature-K", type=float, default=600.0)
    run.add_argument("--timeout-s", type=int, default=3600)
    run.add_argument("--skip-done", action="store_true", help="keep case files already written")
    run.add_argument("--memory-history", action="store_true", help="passed to each case (see case --help)")
    case = sub.add_parser("case", help="one case in this process (the run command's child)")
    case.add_argument("--output", type=Path, required=True)
    case.add_argument("--label", required=True)
    case.add_argument("--structure", required=True)
    case.add_argument("--repeat", default="3x3x3")
    case.add_argument("--ensemble", choices=ENSEMBLES, required=True)
    case.add_argument("--mode", choices=MODES, required=True)
    case.add_argument("--precision", default=None)
    case.add_argument("--steps", type=int, default=100)
    case.add_argument("--temperature-K", type=float, default=600.0)
    case.add_argument("--memory-history", action="store_true",
                      help="record torch.cuda.memory's history and write the allocations live at the peak by site")
    fit = sub.add_parser("fit", help="the table and the fitted factor from a run's case files")
    fit.add_argument("directory", type=Path)
    return p.parse_args(argv)


def case_name(label, structure, ensemble, mode):
    return f"{label}__{structure}__{ensemble}__{mode}"


def read_structure(spec, repeat):
    """(name, atoms): a benchmark name from the baseline export (FASTMACE_STRUCTURES) or a file, repeated."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab_step_time

    return ab_step_time.read_structure(f"{spec}@{repeat}" if "@" not in spec else spec, None)


def integrator_for(ensemble, temperature_K):
    from fastmace.config import IntegratorConfig

    if ensemble == "npt":
        return IntegratorConfig(ensemble="npt", timestep_fs=1.0, friction=0.02, temperature_K=temperature_K,
                                cell_mode="full", pressure_GPa=0.0)
    return IntegratorConfig(timestep_fs=1.0, friction=0.02, temperature_K=temperature_K)


def plan_with_safety(planner, force_field, atoms, integrator, workload, hardware, mode, safety):
    from fastmace.planner import memory

    kept = memory.SAFETY
    memory.SAFETY = safety  # this measurement only: the split of the prediction into its parts
    try:
        return planner.plan(force_field, atoms, integrator, workload, hardware, mode=mode,
                            variable_cell=integrator.ensemble == "npt")
    finally:
        memory.SAFETY = kept


def run_case(args):
    import fastmace  # noqa: F401  (before torch: soft file locks)
    from fastmace.config import EngineConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import memory, minimal
    from fastmace.planner.workload import Workload

    record = dict(label=args.label, structure=args.structure, repeat=args.repeat, ensemble=args.ensemble,
                  mode=args.mode, steps=args.steps, temperature_K=args.temperature_K, safety=memory.SAFETY,
                  margin=memory.MARGIN, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    pristine = loading.load_pristine(args.label)
    name, atoms = read_structure(args.structure, args.repeat)
    record.update(natoms=len(atoms), volume_A3=float(atoms.get_volume()))
    options = {} if args.precision is None else {"precision": args.precision}
    force_field = ForceField(pristine, atoms, **options)
    planner = minimal.MinimalPlanner(engine=EngineConfig())
    hardware = planner.hardware()
    integrator = integrator_for(args.ensemble, args.temperature_K)
    workload = Workload(max_temperature_K=args.temperature_K)
    mode = None if args.mode == JUDGED_MODE else args.mode
    once = plan_with_safety(planner, force_field, atoms, integrator, workload, hardware, mode, 1.0)
    twice = plan_with_safety(planner, force_field, atoms, integrator, workload, hardware, mode, 2.0)
    plan = planner.plan(force_field, atoms, integrator, workload, hardware, mode=mode,
                        variable_cell=args.ensemble == "npt")
    scaled = twice.predicted_memory_bytes - once.predicted_memory_bytes
    record.update(tier=plan.tier, backend=plan.engine.backend, precision=plan.precision.dtype,
                  predicted_bytes=plan.predicted_memory_bytes, scaled_bytes=scaled,
                  fixed_bytes=once.predicted_memory_bytes - scaled,
                  memory_decision=[str(part) for part in plan.decision("memory")[1:]],
                  rewrites_off_by_rule_12=[key for key, _, reason in plan.decisions
                                           if getattr(reason, "code", None) == "rule-12-memory-drop"],
                  hardware=[hardware.name, list(hardware.compute_capability), hardware.memory_bytes])
    import torch

    torch.cuda.reset_peak_memory_stats()
    if args.memory_history:
        torch.cuda.memory._record_memory_history(stacks="python", max_entries=2_000_000)
    started = time.perf_counter()
    engine = planner.build(plan, force_field, atoms, temperature_K=args.temperature_K)
    torch.cuda.synchronize()
    # The build's peak, what stays live into the run, then the steps' own peak (the first step captures).
    record["after_build_bytes"] = int(torch.cuda.max_memory_allocated())
    record["live_after_build_bytes"] = int(torch.cuda.memory_allocated())
    build_peak = record["after_build_bytes"]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        engine.step()
    torch.cuda.synchronize()
    record["step_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    record["live_after_steps_bytes"] = int(torch.cuda.memory_allocated())
    if args.memory_history:
        record["at_peak"] = live_at_peak(torch.cuda.memory._snapshot())
        torch.cuda.memory._record_memory_history(enabled=None)
    record.update(peak_bytes=max(build_peak, record["step_peak_bytes"]),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  seconds=time.perf_counter() - started, torch=torch.__version__,
                  fastmace_commit=os.environ.get("FASTMACE_COMMIT"))
    info = engine.info()
    record["edge_sizes_used"] = list(info.get("edge_sizes_used", ()))
    record["capacity"] = int(engine.force_field.neighbours.capacity)
    record["ratio"] = record["predicted_bytes"] / record["peak_bytes"]
    path = args.output / f"{case_name(args.label, args.structure, args.ensemble, args.mode)}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: record[key] for key in ("label", "structure", "ensemble", "mode", "peak_bytes",
                                                    "predicted_bytes", "ratio")}))


def _site(frames):
    """A call site for grouping: the innermost FastMACE frame, else MACE's, else torch's, else the first one."""
    for marker in ("/fastmace/", "/mace/", "/torch/"):
        for frame in frames or ():
            if marker in frame.get("filename", ""):
                return f"{frame['filename'].split(marker)[-1]}:{frame.get('line')}:{frame.get('name')}"
    frame = (frames or [{}])[0]
    return f"{frame.get('filename', '?')}:{frame.get('line')}:{frame.get('name')}"


def _outer(frames):
    """The outermost FastMACE frame (which stage of the build or step allocated)."""
    inner = [frame for frame in frames or () if "/fastmace/" in frame.get("filename", "")]
    if not inner:
        return "?"
    frame = inner[-1]
    return f"{frame['filename'].split('/fastmace/')[-1]}:{frame.get('line')}:{frame.get('name')}"


def live_at_peak(snapshot, top=40):
    """The allocations live at the largest point of the recorded history, by site, largest first: replays the
    device trace's alloc and free events (``torch.cuda.memory._snapshot``)."""
    events = [event for event in snapshot.get("device_traces", [[]])[0]
              if event.get("action") in ("alloc", "free_completed")]
    live, current, peak, peak_index = {}, 0, 0, -1
    for index, event in enumerate(events):
        if event["action"] == "alloc":
            live[event["addr"]] = event["size"]
            current += event["size"]
        else:
            current -= live.pop(event["addr"], 0)
        if current > peak:
            peak, peak_index = current, index
    alive = {}
    for event in events[:peak_index + 1]:
        if event["action"] == "alloc":
            alive[event["addr"]] = event
        else:
            alive.pop(event["addr"], None)
    by_site, by_stage = {}, {}
    for event in alive.values():
        for table, key in ((by_site, _site(event.get("frames"))), (by_stage, _outer(event.get("frames")))):
            entry = table.setdefault(key, [0, 0])
            entry[0] += event["size"]
            entry[1] += 1
    def ranked(table):
        return [{"site": key, "bytes": value[0], "count": value[1]}
                for key, value in sorted(table.items(), key=lambda item: -item[1][0])[:top]]
    return {"traced_peak_bytes": peak, "events": len(events), "peak_event": peak_index,
            "by_site": ranked(by_site), "by_stage": ranked(by_stage)}


def run(args):
    from fastmace.model import labels

    args.output.mkdir(parents=True, exist_ok=True)
    chosen = args.labels or list(labels.LABELS)
    failures = []
    for label in chosen:
        for structure in args.structures:
            for ensemble in args.ensembles:
                for mode in args.modes:
                    name = case_name(label, structure, ensemble, mode)
                    if args.skip_done and (args.output / f"{name}.json").exists():
                        continue
                    command = [sys.executable, str(Path(__file__).resolve()), "case", "--output", str(args.output),
                               "--label", label, "--structure", structure, "--repeat", args.repeat,
                               "--ensemble", ensemble, "--mode", mode, "--steps", str(args.steps),
                               "--temperature-K", str(args.temperature_K)]
                    if args.precision:
                        command += ["--precision", args.precision]
                    if args.memory_history:
                        command.append("--memory-history")
                    log = args.output / f"{name}.log"
                    with log.open("w") as handle:
                        try:
                            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                                  timeout=args.timeout_s).returncode
                        except subprocess.TimeoutExpired:
                            code = "timeout"
                    if code != 0:
                        failures.append((name, code))
                        (args.output / f"{name}.failed.json").write_text(json.dumps(
                            dict(label=label, structure=structure, ensemble=ensemble, mode=mode, code=code,
                                 log=log.name)) + "\n")
                    print(f"{name}: {'ok' if code == 0 else f'failed ({code})'}", flush=True)
    return 1 if failures else 0


def fit(args):
    cases = [json.loads(path.read_text()) for path in sorted(args.directory.glob("*.json"))
             if not path.name.endswith(".failed.json")]
    if not cases:
        print("no case files")
        return 1

    def needed(case):  # the factor that makes this case's prediction equal its peak
        return (case["peak_bytes"] - case["fixed_bytes"]) / case["scaled_bytes"] if case["scaled_bytes"] else 0.0

    judged = [case for case in cases if case["mode"] == JUDGED_MODE]
    factor = max((needed(case) for case in judged), default=float("nan"))
    print("| label | structure | ensemble | tier | backend | precision | peak (MiB) | predicted (MiB) | ratio "
          "| factor needed | ratio at the fitted factor |")
    print("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    worst = []
    for case in sorted(cases, key=lambda c: (c["mode"] != JUDGED_MODE, c["label"], c["structure"], c["ensemble"])):
        fitted = (case["fixed_bytes"] + factor * case["scaled_bytes"]) / case["peak_bytes"]
        if case["mode"] == JUDGED_MODE:
            worst.append(fitted)
        print(f"| {case['label']} | {case['structure']} | {case['ensemble']} | {case['tier']} | {case['backend']} "
              f"| {case['precision']} | {case['peak_bytes'] / 2**20:.0f} | {case['predicted_bytes'] / 2**20:.0f} "
              f"| {case['ratio']:.2f} | {needed(case):.3f} | {fitted:.2f} |")
    print(f"\nfitted SAFETY (largest factor needed over {len(judged)} graph-tier cases): {factor:.3f}")
    if worst:
        print(f"ratios at the fitted factor: min {min(worst):.2f}, max {max(worst):.2f}; "
              f"pass (never below 1, at most {PASS_RATIO}): {min(worst) >= 1.0 - 1e-9 and max(worst) <= PASS_RATIO}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.command == "case":
        run_case(args)
        return 0
    if args.command == "run":
        return run(args)
    return fit(args)


if __name__ == "__main__":
    sys.exit(main())
