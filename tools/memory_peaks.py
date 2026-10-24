"""Rule 12's estimate against the measured GPU peak (plan task 5-T03; SPEC §9.3 rule 12, decision 51).

``run`` measures, for each label x structure x ensemble x tier, the peak of ``torch.cuda.max_memory_allocated``
over the build, the capture and ``--steps`` steps of the minimal planner's engine, each case in its own child
process (a fresh allocator and one compile family, SPEC §8.7). Every case writes ``<output>/<case>.json`` with
the plan's prediction, the estimate's run and build terms (the planner's final ``memory.estimate``, captured
while it plans), the build's peak, the steps' own peak and the overall peak.

``fit`` reads the case files and prints the markdown table of ``docs/validation/5-T03.md``. The run's estimate
is judged against the steps' peak: the smallest ``SAFETY`` (one global factor, SPEC §9.3 rule 12 as amended 23
October 2026) that keeps every graph-tier case's run at or above it. The build's estimate is judged against the
build's peak, with no factor. Then the prediction, max(build, run), at the fitted factor against the overall
peak. Pass: no ratio below 1, none above 1.5.

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
    run.add_argument("--backend", choices=("auto", "hybrid", "cueq"), default="auto",
                     help="the planner's backend (default: its own choice)")
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
    case.add_argument("--backend", choices=("auto", "hybrid", "cueq"), default="auto")
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


def with_parts(call, hardware):
    """(call(), parts): the result of ``call`` (a plan or a revision), and the run's and the build's terms of
    rule 12's final estimate while it ran (the planner's last ``memory.estimate``, after any drops), with the
    run's activations and rewrite terms unscaled (divided by ``SAFETY``), so ``fit`` can judge the run against
    the steps' peak and the build against the build's peak separately."""
    from fastmace.planner import memory

    found, original = [], memory.estimate

    def spy(*args, **kwargs):
        found.append(original(*args, **kwargs))
        return found[-1]

    memory.estimate = spy  # this measurement only
    try:
        result = call()
    finally:
        memory.estimate = original
    if not found:
        return result, None
    final = found[-1]
    scaled = {"activations", "rewrites"}
    run = dict(final.terms)
    return result, {"run_terms": run, "build_terms": dict(final.build_terms), "safety": memory.SAFETY,
                    "run_scaled_unit": sum(run[name] for name in scaled if name in run) / memory.SAFETY,
                    "run_fixed": sum(value for name, value in run.items() if name not in scaled),
                    "run": final.run, "build": final.build, "sms": getattr(hardware, "sms", None)}


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
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend=args.backend))
    hardware = planner.hardware()
    integrator = integrator_for(args.ensemble, args.temperature_K)
    workload = Workload(max_temperature_K=args.temperature_K)
    mode = None if args.mode == JUDGED_MODE else args.mode
    plan, parts = with_parts(lambda: planner.plan(force_field, atoms, integrator, workload, hardware, mode=mode,
                                                  variable_cell=integrator.ensemble == "npt"), hardware)
    record.update(tier=plan.tier, backend=plan.engine.backend, precision=plan.precision.dtype,
                  predicted_bytes=plan.predicted_memory_bytes, parts=parts,
                  memory_decision=[str(part) for part in plan.decision("memory")[1:]],
                  rewrites_off_by_rule_12=[key for key, _, reason in plan.decisions
                                           if getattr(reason, "code", None) == "rule-12-memory-drop"],
                  hardware=[hardware.name, list(hardware.compute_capability), hardware.memory_bytes])
    import torch

    torch.cuda.reset_peak_memory_stats()
    if args.memory_history:
        torch.cuda.memory._record_memory_history(stacks="python", max_entries=2_000_000)
    started = time.perf_counter()
    from fastmace.errors import SelfCheckError
    from fastmace.planner.fallback import memory_revised
    from fastmace.planner.reactions import selfcheck_revision

    # A self-check that fails turns its rewrite off and the build retries, as a Simulation does (rule 13); the
    # peak keeps the failed attempts' allocations, as production pays them too.
    record["rule_13_revisions"] = []
    while True:
        try:
            engine = planner.build(plan, force_field, atoms, temperature_K=args.temperature_K)
            break
        except SelfCheckError as error:
            if len(record["rule_13_revisions"]) >= 3:
                raise
            revised = selfcheck_revision(plan, error)
            plan, parts = with_parts(lambda: memory_revised(revised, force_field, atoms, planner), hardware)
            record["rule_13_revisions"].append(str(error).splitlines()[0])
            record.update(predicted_bytes=plan.predicted_memory_bytes, parts=parts,
                          memory_decision=[str(part) for part in plan.decision("memory")[1:]])
    torch.cuda.synchronize()
    # The build's peak, what stays live into the run, then the steps' own peak (the first step captures).
    record["after_build_bytes"] = int(torch.cuda.max_memory_allocated())
    record["live_after_build_bytes"] = int(torch.cuda.memory_allocated())
    build_peak = record["after_build_bytes"]
    torch.cuda.reset_peak_memory_stats()
    steps_begin = len(torch.cuda.memory._snapshot()["device_traces"][0]) if args.memory_history else 0
    for _ in range(args.steps):
        engine.step()
    torch.cuda.synchronize()
    record["step_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    record["live_after_steps_bytes"] = int(torch.cuda.memory_allocated())
    if args.memory_history:
        snapshot = torch.cuda.memory._snapshot()
        record["at_peak"] = live_at_peak(snapshot)
        record["at_step_peak"] = live_at_peak(snapshot, start=steps_begin)
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


def live_at_peak(snapshot, top=40, start=0):
    """The allocations live at the largest point of the recorded history, by site, largest first: replays the
    device trace's alloc and free events (``torch.cuda.memory._snapshot``). ``start``: the trace index from
    which the peak is sought (the steps' own peak: the trace's length when the steps began); the allocations
    live then still count."""
    trace = snapshot.get("device_traces", [[]])[0]
    events = [(index >= start, event) for index, event in enumerate(trace)
              if event.get("action") in ("alloc", "free_completed")]
    live, current, peak, peak_index = {}, 0, 0, -1
    for index, (after, event) in enumerate(events):
        if event["action"] == "alloc":
            live[event["addr"]] = event["size"]
            current += event["size"]
        else:
            current -= live.pop(event["addr"], 0)
        if after and current > peak:
            peak, peak_index = current, index
    events = [event for _, event in events]
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
                    if args.backend != "auto":
                        command += ["--backend", args.backend]
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
    """The judged (graph-tier) cases: the run's estimate against the steps' own peak, giving the smallest
    ``SAFETY`` that keeps every run at or above it, and the build's estimate against the build's peak (no
    factor); then the prediction, max(build, run), at the fitted factor against the overall peak."""
    cases = [json.loads(path.read_text()) for path in sorted(args.directory.glob("*.json"))
             if not path.name.endswith(".failed.json")]
    cases = [case for case in cases if case.get("parts")]
    if not cases:
        print("no case files with the estimate's parts")
        return 1

    def needed(case):  # the factor that makes this case's run estimate equal its steps' peak
        parts = case["parts"]
        unit = parts["run_scaled_unit"]
        return (case["step_peak_bytes"] - parts["run_fixed"]) / unit if unit else 0.0

    def predicted(case, factor):
        parts = case["parts"]
        return max(parts["build"], parts["run_fixed"] + factor * parts["run_scaled_unit"])

    judged = [case for case in cases if case["mode"] == JUDGED_MODE]
    factor = max((needed(case) for case in judged), default=float("nan"))
    print("| label | structure | ensemble | backend | precision | build peak / est. (MiB) | steps' peak / run est. "
          "(MiB) | factor needed | ratio now | ratio at the fitted factor |")
    print("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    ratios, builds = [], []
    for case in sorted(cases, key=lambda c: (c["mode"] != JUDGED_MODE, c["label"], c["structure"], c["ensemble"])):
        parts = case["parts"]
        fitted = predicted(case, factor) / case["peak_bytes"]
        if case["mode"] == JUDGED_MODE:
            ratios.append(fitted)
            builds.append(parts["build"] / case["after_build_bytes"])
        mib = 2**20
        print(f"| {case['label']} | {case['structure']} | {case['ensemble']} | {case['backend']} "
              f"| {case['precision']} | {case['after_build_bytes'] / mib:.0f} / {parts['build'] / mib:.0f} "
              f"| {case['step_peak_bytes'] / mib:.0f} / {parts['run'] / mib:.0f} | {needed(case):.3f} "
              f"| {case['ratio']:.2f} | {fitted:.2f} |")
    print(f"\nfitted SAFETY (largest factor the run needs over {len(judged)} graph-tier cases): {factor:.3f}")
    if ratios:
        print(f"build estimate / build peak: min {min(builds):.2f}, max {max(builds):.2f}")
        print(f"prediction / peak at the fitted factor: min {min(ratios):.2f}, max {max(ratios):.2f}; "
              f"pass (never below 1, at most {PASS_RATIO}): {min(ratios) >= 1.0 - 1e-9 and max(ratios) <= PASS_RATIO}")
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
