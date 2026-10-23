"""The plan corpus: the same planner inputs planned by two FastMACE versions (plan task 5-T11; choice 7).

Phase 5's "default plans do not change" (``plans/phase-5.md``, conductor's choice 7) made checkable on the
CPU: every case of a named set is planned by ``MinimalPlanner.plan_traits`` with injected hardware and
software, from the stand-ins of ``tools/plan_corpus_labels.py`` (no checkpoint is loaded), and written as a
canonical JSON record. ``compare`` sorts each case into

- ``identical``: refusals, tier, runner, superset, every resolved config, ``auto_options_off``,
  ``requested`` and the choice of every decision both sides have are equal (reason texts, rule numbers and
  decisions only the new side has are not compared);
- ``explained``: only the rewrite set differs, and the new side carries a decision with one of the
  explaining codes (rule 12's action, decision 51: it only turns rewrites off);
- ``different``: anything else (exit status 1).

``--package`` names the package to plan with: ``fastmace`` (this checkout) or a renamed export of a tag
(``python -m fastmace.bench.versions export --ref fastmace-4 --name fastmace_4``) on ``PYTHONPATH``. Every
difference between versions' APIs is handled in ``_plan``, nowhere else.

Usage::

    python tools/plan_corpus.py dump --set acceptance --package fastmace_4 --out old.json
    python tools/plan_corpus.py dump --set acceptance --package fastmace --out new.json
    python tools/plan_corpus.py compare old.json new.json --explain-rules 12 --out report.json

The format was agreed with 5-T10 (session notes, ``5-T11-plan-corpus-format.md``): cases with stable ids in
named sets, a package argument, canonical records whose reason texts sit apart under ``texts``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import plan_corpus_labels as corpus_labels  # noqa: E402

GiB = 2**30
TORCH_VERSION = "2.14.1"  # the GPU environments' torch (tests/cpu/test_planner.py's TORCH_2_14)
D3_CUTOFF_A = 40 * 0.52917726  # 40 bohr, torch-dftd's constant (tests/cpu/test_minimal_planner.py)
MAX_TEMPERATURE_K = 300.0
MIN_MC_TEMPERATURE_K = 250.0
# name -> (GPU name, compute capability, total bytes, free bytes). The reference GPU with the desktop's ~4 GiB
# taken (conductor, 23 October 2026), and a small GPU so that rule 12's action is exercised.
HARDWARE = {
    "rtx5060ti": ("NVIDIA GeForce RTX 5060 Ti", (12, 0), 16 * GiB, 12 * GiB),
    "low-memory": ("low-memory GPU (plan corpus)", (12, 0), 4 * GiB, 3 * GiB),
}
WORKLOADS = ("nvt", "npt", "npt-d3", "mc-swap", "relax-fixed", "relax-cell")
STRUCTURES = ("NaCl", "NbCl5")
BACKENDS = ("auto", "hybrid", "cueq")
# The largest models by parameter count: the low-memory cases.
LOW_MEMORY_LABELS = ("omol0-xl", "polar1-l", "mp0-large", "off23-large")
CONFIG_PARTS = ("rewrites", "engine", "precision", "neighbours", "polar", "integrator", "optimizer")
# Plan fields phase 5 added (or fills); recorded, never compared.
PHASE_5_FIELDS = ("calibration", "adaptation", "predicted_memory_bytes")


@dataclass(frozen=True)
class Case:
    """One planner input; ``id`` is stable across versions and runs."""

    label: str
    workload: str
    structure: str
    backend: str = "auto"
    hardware: str = "rtx5060ti"
    precision: str = "float32"

    @property
    def id(self) -> str:
        return "/".join((self.label, self.workload, self.structure, self.backend, self.hardware, self.precision))


def acceptance_cases() -> list[Case]:
    """5-T11's set: 19 labels x six workloads x NaCl, NbCl5 x three backends, plus the low-memory cases."""
    cases = [Case(label, workload, structure, backend)
             for label in corpus_labels.LABELS for workload in WORKLOADS for structure in STRUCTURES
             for backend in BACKENDS]
    cases += [Case(label, workload, structure, "auto", "low-memory")
              for label in LOW_MEMORY_LABELS for workload in ("nvt", "npt") for structure in STRUCTURES]
    return cases


SETS: dict[str, Callable[[], list[Case]]] = {"acceptance": acceptance_cases}


def cases(set_name: str) -> list[Case]:
    if set_name not in SETS:
        raise KeyError(f"no case set {set_name!r}; the sets are {sorted(SETS)}")
    return SETS[set_name]()


# --------------------------------------------------------------------------- planning


def structure(name: str):
    """NaCl and NbCl5 3x3x3 (NbCl5: the protocols' r2SCAN cell, ``fastmace/protocols/data``)."""
    from ase.build import bulk
    from ase.io import read

    if name == "NaCl":
        return bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    if name == "NbCl5":
        return read(ROOT / "fastmace" / "protocols" / "data" / "NbCl5_R2SCAN.cif").repeat((3, 3, 3))
    raise KeyError(name)


def _module(package: str, name: str):
    return importlib.import_module(f"{package}.{name}")


def _plan(package: str, case: Case):
    """``case`` planned by ``package``: the one place that knows the versions' APIs."""
    config = _module(package, "config")
    minimal = _module(package, "planner.minimal")
    plan_module = _module(package, "planner.plan")
    workload_module = _module(package, "planner.workload")
    capabilities = _module(package, "model.capabilities")
    constraints = _module(package, "engine.constraints")

    gpu, capability, total, free = HARDWARE[case.hardware]
    software = plan_module.SoftwareTraits(
        oeq=(True, "plan corpus: available"), triton=(True, "plan corpus: available"),
        nvrtc=(True, "plan corpus: available"), fused_cueq=(True, "plan corpus: available"),
        mace_imported_first=())
    planner = minimal.MinimalPlanner(engine=config.EngineConfig(backend=case.backend), software=software,
                                     process=constraints.ProcessRecord(), environ={},
                                     torch_version=TORCH_VERSION)
    # The free memory at plan time (CUDA's count in a real run); injected so both sides see the same GPU.
    planner._free_memory_bytes = lambda hardware: free
    traits = _with_dims(package, capabilities.model_traits(corpus_labels.stand_in(case.label)), case.label)

    atoms = structure(case.structure)
    integrator, optimizer, flags = config.IntegratorConfig(timestep_fs=1.0), None, {}
    workload = workload_module.Workload(max_temperature_K=MAX_TEMPERATURE_K)
    if case.workload in ("npt", "npt-d3"):
        integrator = config.IntegratorConfig(timestep_fs=1.0, ensemble="npt")
        flags["variable_cell"] = True
    if case.workload == "npt-d3":
        flags["d3_cutoff_A"] = D3_CUTOFF_A
    if case.workload == "mc-swap":
        workload = workload_module.Workload(max_temperature_K=MAX_TEMPERATURE_K, species_changes=True,
                                            mc_moves=("swap",), min_mc_temperature_K=MIN_MC_TEMPERATURE_K)
    if case.workload in ("relax-fixed", "relax-cell"):
        optimize = _module(package, "optimize")
        cell = optimize.FrechetCell() if case.workload == "relax-cell" else None
        optimizer = optimize.LBFGS(cell_filter=cell).to_config()
        flags["optimizer"] = optimizer
    hardware = plan_module.Hardware(gpu, capability, total)
    return planner.plan_traits(traits, atoms, integrator, workload, hardware, precision=case.precision,
                               rewrites="auto", **flags)


def _with_dims(package: str, traits, label: str):
    """Rule 12's model sizes on ``traits``, where the version's traits carry them (5-T03's ``dims``)."""
    if "dims" not in {field.name for field in dataclasses.fields(traits)}:
        return traits
    record = corpus_labels.model_dims(label)
    capabilities = _module(package, "model.capabilities")
    if record is None or not hasattr(getattr(capabilities, "ModelDims", None), "from_record"):
        return traits
    return dataclasses.replace(traits, dims=capabilities.ModelDims.from_record(record))


def _json(value: Any) -> Any:
    """``value`` as plain JSON (dataclasses, tuples, frozensets, enums and anything else by ``str``)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _reason_code(reason) -> list:
    return [getattr(reason, "rule", None), getattr(reason, "code", None)]


def record(plan) -> dict:
    """The canonical record of a plan: everything compared, and the texts apart."""
    return {
        "refused": None,
        "tier": plan.tier,
        "runner": getattr(plan, "runner", "in-process"),
        "superset": list(plan.superset),
        "allowances": list(plan.allowances),
        "configs": {part: _json(getattr(plan, part, None)) for part in CONFIG_PARTS},
        "off": dict(plan.auto_options_off),
        "requested": {item: str(value) for item, value in plan.requested},
        "decisions": {key: choice for key, choice, _ in plan.decisions},
        "decision_codes": {key: _reason_code(reason) for key, _, reason in plan.decisions},
        "phase5": {name: _json(getattr(plan, name, None)) for name in PHASE_5_FIELDS},
        "texts": {key: str(reason) for key, _, reason in plan.decisions},
    }


def refusal(error: Exception) -> dict:
    """The record of a refused case: the error's type and its (rule, code) pairs in emission order."""
    reasons = list(getattr(error, "reasons", ()) or ())
    return {"refused": {"error": type(error).__name__, "reasons": [_reason_code(reason) for reason in reasons]},
            "texts": {"error": str(error)}}


def _unrenamed(value: Any, package: str) -> Any:
    """``value`` with the export's package name read back as ``fastmace``.

    ``fastmace.bench.versions export`` renames every standalone word ``fastmace`` in the exported sources,
    string values included (``IntegratorConfig.dof == "fastmace"`` reads ``"fastmace_4"`` there).
    """
    if package == "fastmace":
        return value
    if isinstance(value, str):
        return re.sub(rf"\b{re.escape(package)}\b", "fastmace", value)
    if isinstance(value, dict):
        return {_unrenamed(key, package): _unrenamed(item, package) for key, item in value.items()}
    if isinstance(value, list):
        return [_unrenamed(item, package) for item in value]
    return value


def plan_case(package: str, case: Case) -> dict:
    errors = _module(package, "errors")
    refusals = tuple(getattr(errors, name) for name in ("PlanRefused", "ModelElementError",
                                                        "ProcessConstraintError") if hasattr(errors, name))
    try:
        entry = record(_plan(package, case))
    except refusals as error:
        entry = refusal(error)
    return {"id": case.id, **_unrenamed(entry, package)}


def dump(set_name: str, package: str) -> dict:
    """Every case of ``set_name`` planned by ``package``: ``{"package", "set", "cases": {id: record}}``."""
    planned = {}
    for case in cases(set_name):
        planned[case.id] = plan_case(package, case)
    return {"package": package, "version": getattr(importlib.import_module(package), "__version__", None),
            "set": set_name, "count": len(planned), "cases": planned}


# --------------------------------------------------------------------------- comparing


def _refusal_key(entry: dict):
    refused = entry.get("refused")
    if refused is None:
        return None
    return refused["error"], sorted(Counter(tuple(pair) for pair in refused["reasons"]).items())


def compare_case(old: dict, new: dict, explain_codes: frozenset[str]) -> dict:
    """One row: ``identical``, ``explained`` (with the codes) or ``different`` (with what differs)."""
    if old.get("refused") is not None or new.get("refused") is not None:
        same = _refusal_key(old) == _refusal_key(new)
        return {"verdict": "identical" if same else "different",
                "differs": [] if same else ["refused"], "old": old.get("refused"), "new": new.get("refused")}
    differs = []
    for name in ("tier", "runner", "superset", "allowances", "off", "requested"):
        if old[name] != new[name]:
            differs.append(name)
    for part in CONFIG_PARTS:
        if old["configs"].get(part) != new["configs"].get(part):
            differs.append(f"configs.{part}")
    changed = sorted(key for key in old["decisions"].keys() & new["decisions"].keys()
                     if old["decisions"][key] != new["decisions"][key])
    removed = sorted(old["decisions"].keys() - new["decisions"].keys())
    added = sorted(new["decisions"].keys() - old["decisions"].keys())
    row = {"added_decisions": added}
    if not differs and not changed and not removed:
        return {"verdict": "identical", **row}
    explaining = sorted({key for key, (_, code) in new["decision_codes"].items() if code in explain_codes})
    rewrites_only = set(differs) <= {"configs.rewrites"} and not removed
    if rewrites_only and explaining:
        return {"verdict": "explained", "by": explaining, "differs": differs, "changed_decisions": changed, **row}
    return {"verdict": "different", "differs": differs, "changed_decisions": changed,
            "removed_decisions": removed, **row}


def explaining_codes(package: str, rules: list[int], codes: list[str]) -> frozenset[str]:
    """``codes`` plus every code of ``rules`` in ``package``'s rule table (``planner/rules.py``, phase 5)."""
    out = set(codes)
    if rules:
        table = _module(package, "planner.rules")
        for number in rules:
            out.update(table.codes(number))
    return frozenset(out)


def compare(old: dict, new: dict, explain_codes: frozenset[str] = frozenset()) -> dict:
    """Every case of two dumps of the same set; cases one side lacks are ``different``."""
    if old["set"] != new["set"]:
        raise ValueError(f"the dumps are of different sets: {old['set']!r} and {new['set']!r}")
    rows = {}
    for case_id in sorted(old["cases"].keys() | new["cases"].keys()):
        if case_id not in old["cases"] or case_id not in new["cases"]:
            rows[case_id] = {"verdict": "different", "differs": ["missing"]}
            continue
        rows[case_id] = compare_case(old["cases"][case_id], new["cases"][case_id], explain_codes)
    counts = Counter(row["verdict"] for row in rows.values())
    return {"old": old["package"], "new": new["package"], "set": old["set"],
            "explain_codes": sorted(explain_codes), "counts": dict(sorted(counts.items())), "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dumping = commands.add_parser("dump", help="plan every case of a set with one package")
    dumping.add_argument("--set", default="acceptance", choices=sorted(SETS))
    dumping.add_argument("--package", default="fastmace")
    dumping.add_argument("--out", type=Path, required=True)
    comparing = commands.add_parser("compare", help="compare two dumps of the same set")
    comparing.add_argument("old", type=Path)
    comparing.add_argument("new", type=Path)
    comparing.add_argument("--explain-rules", default="", help="rules whose codes explain a rewrite change (12)")
    comparing.add_argument("--explain-codes", default="", help="further explaining codes, comma-separated")
    comparing.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "dump":
        result = dump(args.set, args.package)
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(f"{result['count']} cases of {args.set!r} planned by {args.package}")
        return 0
    old, new = json.loads(args.old.read_text()), json.loads(args.new.read_text())
    rules = [int(number) for number in args.explain_rules.split(",") if number]
    codes = [code for code in args.explain_codes.split(",") if code]
    report = compare(old, new, explaining_codes(new["package"], rules, codes))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{report['old']} -> {report['new']}, set {report['set']!r}: {report['counts']}")
    for case_id, row in report["rows"].items():
        if row["verdict"] != "identical":
            print(f"  {row['verdict']}: {case_id}: {row.get('differs')} {row.get('by', '')}")
    return 1 if report["counts"].get("different") else 0


if __name__ == "__main__":
    sys.exit(main())
