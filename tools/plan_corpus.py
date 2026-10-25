"""The plan corpus of phase 5's acceptance (plan task 5-T11, runbook item 3): no regression in the planner's choices.

``run`` plans every label of the label table on every standard workload with injected hardware and software (the
RTX 5060 Ti, torch 2.14, an empty environment) and the stand-in models of this version's
``tests/cpu/test_planner.py``, and writes one record per case: tier, backend, precision, the rewrites, the engine
fields, the neighbour sizes, or the refusal's codes. Run it once with ``PYTHONPATH`` on the baseline's tree (a
``git archive`` of ``fastmace-4``) and once on the freeze's:

    PYTHONPATH=<tree> python tools/plan_corpus.py run --root <tree> --output <dir>/corpus.json
    python tools/plan_corpus.py compare <baseline>/corpus.json <freeze>/corpus.json

``compare`` lists every difference. One is accepted only when it is explained: a field the runbook names for one of
the owner's decisions 57-59 on the labels and workloads that decision covers, a rule 12 or rule 15 decision, or a
field the baseline did not have (new in phase 5). Anything else fails (exit 1).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path

WORKLOADS = ("nvt", "npt", "npt-d3", "mc-swap", "relax-fixed", "relax-cell")
STRUCTURES = ("nacl", "nbcl5")
D3_CUTOFF_A = 40 * 0.52917726  # 40 bohr, torch-dftd's constant (tests/cpu/test_minimal_planner.py)

# Decisions 57-59 (owner, 24 October 2026): the fields each may change, and the cases it covers.
GENERIC_FAMILY = ("mp0-medium", "mp0-large", "mp0b3-medium", "mpa0-medium", "omat0-medium", "matpes-pbe",
                  "matpes-r2scan")
POLAR1 = ("polar1-s", "polar1-m", "polar1-l")
EXPLAINED = (
    ("decision 57", ("rewrites.grouped_launch",), lambda label, workload: label in GENERIC_FAMILY),
    ("decision 58", ("rewrites.fx_e3nn_linear", "rewrites.polar_graph_sums"), lambda label, workload: label in POLAR1),
    ("decision 59", ("engine.dynamic_edges",), lambda label, workload: True),  # the EngineConfig default
)
RULE_CODES = ("rule-12", "rule-15", "memory")


def _cases():
    for label in sorted(_labels()):
        for structure in STRUCTURES:
            for workload in WORKLOADS:
                yield label, structure, workload


def _labels():
    from fastmace.model import labels

    return labels.LABELS


def _flatten(prefix, value, out):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _flatten(f"{prefix}.{field.name}" if prefix else field.name, getattr(value, field.name), out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value
    elif isinstance(value, (tuple, list)):
        out[prefix] = json.dumps(value, default=str)
    else:
        out[prefix] = str(value)


def plan_case(label, structure, workload, tp, tm, structures):
    from fastmace.config import IntegratorConfig, OptimizerConfig
    from fastmace.planner.workload import Workload

    atoms = getattr(structures, structure)()
    entry = _labels()[label]
    loaded = types.SimpleNamespace(label=label, head=entry.head, sha256=entry.checkpoint.sha256)
    integrator = IntegratorConfig(ensemble="npt" if workload.startswith("npt") else "nvt", temperature_K=300.0)
    flags = dict(loaded=loaded)
    work = Workload(max_temperature_K=300.0)
    if workload == "npt-d3":
        flags["d3_cutoff_A"] = D3_CUTOFF_A
    elif workload == "mc-swap":
        work = Workload(max_temperature_K=300.0, species_changes=True, mc_moves=("swap",), min_mc_temperature_K=300.0)
    elif workload.startswith("relax"):
        flags["optimizer"] = OptimizerConfig(scheme="lbfgs",
                                             cell_filter="frechet" if workload == "relax-cell" else None)
    planner = tm.minimal()
    try:
        plan = planner.plan_traits(tp.traits(label), atoms, integrator, work, tp.GPU, precision="float32", **flags)
    except Exception as error:  # noqa: BLE001  (a refusal is a result)
        codes = sorted({getattr(reason, "code", None) or "" for reason in getattr(error, "reasons", ())})
        return {"refused": type(error).__name__, "codes": codes}
    record = {"tier": plan.tier}
    for part in ("engine", "precision", "rewrites", "neighbours"):
        _flatten(part, getattr(plan, part, None), record)
    record["decision_codes"] = sorted({f"{key}:{getattr(why, 'code', None)}" for key, _, why in plan.decisions
                                       if getattr(why, "code", None)})
    return record


def run(args):
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "tests"), str(root / "tests" / "cpu")]
    import test_minimal_planner as tm
    import test_planner as tp

    from _support import structures

    corpus = {}
    for label, structure, workload in _cases():
        corpus[f"{label}|{structure}|{workload}"] = plan_case(label, structure, workload, tp, tm, structures)
    commit = (root / ".source_commit").read_text().strip() if (root / ".source_commit").exists() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"commit": commit, "cases": corpus}, indent=1, sort_keys=True))
    refused = sum("refused" in record for record in corpus.values())
    print(f"{len(corpus)} cases ({refused} refused) -> {args.output}")


def explain(key, field, before, after):
    """The reason a difference is accepted, or None."""
    label, _, workload = key.split("|")
    if field == "decision_codes":
        old = dict(item.split(":", 1) for item in before or [])
        new = dict(item.split(":", 1) for item in after or [])
        named = ("dynamic_edges", "grouped_launch", "fx_e3nn_linear", "polar_graph_sums")
        for item in set(old) | set(new):
            if old.get(item) == new.get(item) or item not in old:
                continue  # unchanged, or a decision new in phase 5
            if any(name in item for name in named) or any(code in str(new.get(item)) + str(old.get(item))
                                                          for code in RULE_CODES):
                continue  # decisions 57-59, rules 12 and 15
            return None
        return "decisions new in phase 5, decisions 57-59 and rules 12/15"
    if before is None and field not in ("refused", "codes", "tier"):
        return "new in phase 5 (no baseline field)"
    for name, fields, covers in EXPLAINED:
        if field in fields and covers(label, workload):
            return name
    return None


def compare(args):
    base = json.loads(args.baseline.read_text())["cases"]
    freeze = json.loads(args.freeze.read_text())["cases"]
    accepted, failed = [], []
    for key in sorted(set(base) | set(freeze)):
        a, b = base.get(key), freeze.get(key)
        if a is None or b is None:
            failed.append((key, "case", a is not None, b is not None))
            continue
        for field in sorted(set(a) | set(b)):
            if a.get(field) == b.get(field):
                continue
            why = explain(key, field, a.get(field), b.get(field))
            (accepted if why else failed).append((key, field, a.get(field), b.get(field), why))
    by_reason = {}
    for item in accepted:
        by_reason[item[4]] = by_reason.get(item[4], 0) + 1
    print(f"{len(base)} baseline cases, {len(freeze)} freeze cases")
    print("accepted differences:", json.dumps(by_reason, sort_keys=True))
    for item in failed:
        print("UNEXPLAINED", *item[:4], sep=" | ")
    print("PASS" if not failed else f"FAIL ({len(failed)} unexplained)")
    return 0 if not failed else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", type=Path, required=True, help="the version's source tree (its tests/ give the stand-ins)")
    r.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("baseline", type=Path)
    c.add_argument("freeze", type=Path)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
