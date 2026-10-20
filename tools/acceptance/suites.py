"""Acceptance runs of phases 1a (1a-T19) and 1b (1b-T23): machines, suites and the command of every job.

Standard library only: ``run.py`` imports this module on the target machine before anything else,
with any Python, and the CPU tests check the job lists without a GPU.

Suites (plans/phase-1a.md, task 1a-T19 and the conductor decisions after T10, wave 1b and the
second outside review):

- ``speed``: same-process A/B of FastMACE against the frozen research engine, 12 alternating
  blocks of 300 steps after 300 warm-up steps. MH-1 (``mh1-r2scan``) on NaCl, NaNbO3 and NbCl5
  3x3x3, hybrid FP32 NVT, in three processes (ABAB; FastMACE built first; ABBA), plus one NPT case;
  the secondary set MP-0 medium, POLAR-1-S and OMOL-0 XL on NaCl and NbCl5. Each A/B process dumps
  FastMACE's starting forces. Then single-engine control processes of every configuration, with the
  SPEC protocol (4 blocks), for setup time, peak memory and FastMACE's forces outside an A/B process.
  The FastMACE-only controls run without the hybrid launcher.
- ``gate``: ``tools/force_stress_gate.py --engine fastmace`` for the 19 labels of SPEC §12.5, one
  label per run, MACE-OFF on the organic set (the water box unrepeated).
- ``workflow``: ``main.py`` smoke, ``--profile anneal`` on NaCl and ``--quench`` on its output, with
  ``--engine gpu`` and ``--engine fastmace`` (needs plan task 1a-T17's ``main.py``).

Outputs: ``<machine root>/<run id>/<job>/``. On workstations everything (outputs, caches,
temporary files) stays under /mnt/homes/zwingel (ResearchModule's AGENTS.md); nothing goes to /tmp.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

NAS_HOME = PurePosixPath("/mnt/homes/zwingel")

# The 19 labels of SPEC §12.5 (fastmace.model.labels.LABELS), in that table's order.
BENCHMARK_LABELS = ("mp0-medium", "mp0-large", "mp0b3-medium", "mpa0-medium", "omat0-medium", "matpes-pbe",
                    "matpes-r2scan", "mh1-omat", "mh1-mp", "mh1-r2scan")
LONG_RANGE_LABELS = ("polar1-s", "polar1-m", "polar1-l", "omol0-xl")
MACE_OFF_LABELS = ("off23-small", "off23-medium", "off23-large", "off23b-medium", "off24-medium")
GATE_LABELS = BENCHMARK_LABELS + LONG_RANGE_LABELS + MACE_OFF_LABELS

CRYSTALS = ("NaCl", "NaNbO3", "NbCl5")
SECONDARY_CRYSTALS = ("NaCl", "NbCl5")
PRIMARY_MODEL = "mh1-r2scan"
SECONDARY_MODELS = ("mp0-medium", "polar1-s", "omol0-xl")
NPT_CASE = dict(model=PRIMARY_MODEL, structures=("NaCl",), cell_mode="hydrostatic")
# Plain cuEq float64 is about 20x slower per step than the hybrid (1a-T13: 78 ms on NaCl), so NaCl only.
CUEQ_CASE = dict(structures=("NaCl",))
# The organic set of SPEC §12.5 (ResearchModule Structures/Organic): three COD crystals 3x3x3 and the
# 648-atom water box as it is (the plan's "--repeat 1 1 1", written as the structure's own repeat).
ORGANIC_STRUCTURES = ("Structures/Organic/urea_cod1008787.cif", "Structures/Organic/naphthalene_cod2311090.cif",
                      "Structures/Organic/paracetamol_cod2104364.cif", "Structures/Organic/water_216.extxyz@1x1x1")

AB_BLOCKS = 12  # conductor decision after wave 1b (noise floor 0.16 % with 12 blocks)
CONTROL_BLOCKS = 4  # SPEC §12.5 protocol
WARMUP = BLOCK = 300
LAUNCHER = "Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh"


@dataclass(frozen=True)
class Machine:
    """Where a suite runs: paths on that machine (``~`` is the user's home there)."""

    name: str
    kind: str  # "local" (WSL on the PC) or "workstation" (NAS home)
    root: str  # outputs: <root>/<run id>/
    fastmace: str  # the FastMACE checkout the jobs import (PYTHONPATH)
    baseline: str  # the fuller baseline export (1a-T18 layout, with COMMIT)
    models: str  # checkpoint directory (FASTMACE_MODELS)
    research_module: str  # a ResearchModule tree with Structures/Organic (and main.py for the workflow suite)
    environments: str  # directory of mace_cueq_oeq and mace_cueq_latest
    cache: str  # FASTMACE_CACHE, inductor, Triton, OEq and temporary files
    windows_smi: str | None = None  # local: Windows' nvidia-smi.exe, which also sees desktop programs
    versions: str = ""  # renamed earlier releases (fastmace.bench.versions export): <versions>/fastmace_1a

    def path(self, value: str) -> Path:
        return Path(os.path.expanduser(value))

    def python(self, environment: str) -> str:
        return str(self.path(self.environments) / environment / "bin" / "python")


MACHINES = {
    "local": Machine(
        name="local", kind="local", root="~/fastmace-acceptance", fastmace="/mnt/d/Margraf/FastMACE",
        baseline="~/fastmace-baseline/1a-T18", models="~/models", research_module="/mnt/g/Margraf/ResearchModule",
        environments="~/environments", cache="~/fastmace-cache/acceptance",
        windows_smi="/mnt/c/Windows/System32/nvidia-smi.exe", versions="~/fastmace-versions"),
    **{name: Machine(
        name=name, kind="workstation", root=str(NAS_HOME / "fastmace-acceptance"),
        fastmace=str(NAS_HOME / "FastMACE"), baseline=str(NAS_HOME / "fastmace-baseline" / "1a-T18"),
        models=str(NAS_HOME / "ResearchModule" / "Models"), research_module=str(NAS_HOME / "ResearchModule"),
        environments=str(NAS_HOME / "environments"), cache=str(NAS_HOME / "fastmace-cache" / f"acceptance-{name}"),
        versions=str(NAS_HOME / "fastmace-versions"))
       for name in ("murphy", "leto", "coraline")},
}


class LocationError(ValueError):
    """An output, cache or temporary path breaks ResearchModule's rules for that machine."""


def check_location(machine: Machine, path) -> Path:
    """``path`` (expanded) if a run may write there: on workstations only under /mnt/homes/zwingel (so no /tmp)."""
    resolved = Path(os.path.normpath(os.path.expanduser(str(path))))
    posix = PurePosixPath(resolved.as_posix())
    if machine.kind == "workstation" and NAS_HOME not in posix.parents:
        raise LocationError(f"{path}: on workstations everything goes under {NAS_HOME} (ResearchModule AGENTS.md)")
    return resolved


@dataclass(frozen=True)
class Job:
    """One process of a suite. ``gpu`` jobs need an idle GPU; ``timing`` jobs also nothing else on it."""

    name: str
    argv: tuple
    output: str
    gpu: bool = True
    timing: bool = False
    env: dict = field(default_factory=dict)
    note: str = ""

    def command_line(self) -> str:
        return shlex.join(self.argv)


def checkpoint_path(machine: Machine, label: str) -> str:
    """The checkpoint file of ``label`` in the machine's model directory (the release file name)."""
    name = CHECKPOINT_FILES[CHECKPOINT_OF[label]]
    return str(machine.path(machine.models) / name)


# Release file names (fastmace.model.labels: the last part of each checkpoint URL).
CHECKPOINT_FILES = {
    "medium": "2023-12-03-mace-128-L1_epoch-199.model", "large": "MACE_MPtrj_2022.9.model",
    "medium-0b3": "mace-mp-0b3-medium.model", "medium-mpa-0": "mace-mpa-0-medium.model",
    "medium-omat-0": "mace-omat-0-medium.model", "mace-matpes-pbe-0": "MACE-matpes-pbe-omat-ft.model",
    "mace-matpes-r2scan-0": "MACE-matpes-r2scan-omat-ft.model", "mh-1": "mace-mh-1.model",
    "polar-1-s": "MACE-POLAR-1-S.model", "polar-1-m": "MACE-POLAR-1-M.model", "polar-1-l": "MACE-POLAR-1-L.model",
    "omol-0-xl": "MACE-omol-0-extra-large-1024.model", "off23-small": "MACE-OFF23_small.model",
    "off23-medium": "MACE-OFF23_medium.model", "off23-large": "MACE-OFF23_large.model",
    "off23b-medium": "MACE-OFF23b_medium.model", "off24-medium": "MACE-OFF24_medium.model",
}
CHECKPOINT_OF = {
    "mp0-medium": "medium", "mp0-large": "large", "mp0b3-medium": "medium-0b3", "mpa0-medium": "medium-mpa-0",
    "omat0-medium": "medium-omat-0", "matpes-pbe": "mace-matpes-pbe-0", "matpes-r2scan": "mace-matpes-r2scan-0",
    "mh1-omat": "mh-1", "mh1-mp": "mh-1", "mh1-r2scan": "mh-1", "polar1-s": "polar-1-s", "polar1-m": "polar-1-m",
    "polar1-l": "polar-1-l", "omol0-xl": "omol-0-xl", "off23-small": "off23-small", "off23-medium": "off23-medium",
    "off23-large": "off23-large", "off23b-medium": "off23b-medium", "off24-medium": "off24-medium",
}


def environment(machine: Machine, run_root: Path) -> dict:
    """Variables every job gets: the checkout, caches and temporary files under the machine's cache root."""
    cache = check_location(machine, machine.path(machine.cache))
    tmp = check_location(machine, run_root / "tmp")
    return {
        "PYTHONPATH": str(machine.path(machine.fastmace)), "PYTHONDONTWRITEBYTECODE": "1",
        "FASTMACE_CACHE": str(cache), "FASTMACE_MODELS": str(machine.path(machine.models)),
        "FASTMACE_STRUCTURES": str(machine.path(machine.baseline)),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "MACE_COMPILE_CACHE": str(cache / "compile"), "XDG_CACHE_HOME": str(cache / "xdg"),
        "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        # Inductor's compile workers (conductor rule for 1b-T23: 4, so a compile cannot crowd out the machine).
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
    }


def hybrid_launcher(machine: Machine, run_root: Path) -> list:
    """The research engine's OEq launcher (its export), for processes that build the research engine."""
    oeq = machine.python("mace_cueq_oeq").removesuffix("/bin/python")
    return ["bash", str(machine.path(machine.baseline) / LAUNCHER), oeq,
            str(check_location(machine, machine.path(machine.cache) / "oeq-launcher")), machine.python("mace_cueq_oeq")]


def _tool(machine: Machine, name: str) -> str:
    return str(machine.path(machine.fastmace) / "tools" / name)


def _ab_job(machine, run_root, name, model, structures, sides, *, blocks, pattern="abab", ensemble="nvt",
            backend="hybrid", dump=True, note=""):
    output = check_location(machine, run_root / name)
    args = [_tool(machine, "ab_step_time.py"), "--output", str(output), "--model", model,
            "--checkpoint", checkpoint_path(machine, model), "--structures", *structures, "--sides", *sides,
            "--warmup", str(WARMUP), "--block", str(BLOCK), "--blocks", str(blocks), "--pattern", pattern,
            "--ensemble", ensemble, "--backend", backend]
    if ensemble == "npt":
        args += ["--cell-mode", NPT_CASE["cell_mode"]]
    if dump:
        args.append("--dump-forces")
    if "baseline" in sides:
        args += ["--baseline", str(machine.path(machine.baseline))]
    if backend == "cueq":
        argv = [machine.python("mace_cueq_latest")] + args  # plain cuEq: its own environment, no launcher
    elif "baseline" in sides:
        argv = hybrid_launcher(machine, run_root) + args
    else:
        argv = [machine.python("mace_cueq_oeq")] + args  # FastMACE alone: no launcher (acceptance item 4)
    return Job(name=name, argv=tuple(argv), output=str(output), timing=True, note=note)


def speed_configurations():
    """(slug, model, structures, ensemble, backend) of every A/B configuration."""
    out = [("mh1-nvt", PRIMARY_MODEL, CRYSTALS, "nvt", "hybrid"),
           ("mh1-npt", NPT_CASE["model"], NPT_CASE["structures"], "npt", "hybrid")]
    out += [(f"{model}-nvt", model, SECONDARY_CRYSTALS, "nvt", "hybrid") for model in SECONDARY_MODELS]
    out.append(("mh1-cueq-nvt", PRIMARY_MODEL, CUEQ_CASE["structures"], "nvt", "cueq"))
    return out


def speed_jobs(machine: Machine, run_root: Path) -> list:
    """The speed suite: A/B processes first (primary set three times), then the single-engine controls."""
    run_root = Path(run_root)
    jobs = [
        _ab_job(machine, run_root, "ab-mh1-nvt-r1", PRIMARY_MODEL, CRYSTALS, ("baseline", "fastmace"),
                blocks=AB_BLOCKS, note="ABAB, baseline built first"),
        _ab_job(machine, run_root, "ab-mh1-nvt-r2", PRIMARY_MODEL, CRYSTALS, ("fastmace", "baseline"),
                blocks=AB_BLOCKS, note="ABAB, FastMACE built first (build-order swap, decision after wave 1b)"),
        _ab_job(machine, run_root, "ab-mh1-nvt-r3", PRIMARY_MODEL, CRYSTALS, ("baseline", "fastmace"),
                blocks=AB_BLOCKS, pattern="abba", note="ABBA, baseline built first"),
        _ab_job(machine, run_root, "ab-mh1-npt", NPT_CASE["model"], NPT_CASE["structures"], ("baseline", "fastmace"),
                blocks=AB_BLOCKS, ensemble="npt", note="NPT BAOAB, hydrostatic"),
    ]
    jobs += [_ab_job(machine, run_root, f"ab-{model}-nvt", model, SECONDARY_CRYSTALS, ("baseline", "fastmace"),
                     blocks=AB_BLOCKS, note="secondary set") for model in SECONDARY_MODELS]
    jobs.append(_ab_job(machine, run_root, "ab-mh1-cueq-nvt", PRIMARY_MODEL, CUEQ_CASE["structures"],
                        ("baseline", "fastmace"), blocks=AB_BLOCKS, backend="cueq",
                        note="plain cuEq float64, the plan --backend cueq gets (decision 5 after T15)"))
    for slug, model, structures, ensemble, backend in speed_configurations():
        for side in ("baseline", "fastmace"):
            jobs.append(_ab_job(machine, run_root, f"control-{slug}-{side}", model, structures, (side,),
                                blocks=CONTROL_BLOCKS, ensemble=ensemble, backend=backend, dump=side == "fastmace",
                                note="single engine: setup time, peak memory, FastMACE-only forces"))
    return jobs


# Models whose two engines cannot share a process: the research engine's polar_gpu replaces MACE's and
# graph_longrange's PolarMACE functions process-wide, and FastMACE's qualification then refuses its own
# GPU-only polar path (murphy, 18 October 2026). Their speed comes from alternating single-engine
# processes (the plan's fallback for interfering engines), baseline/FastMACE in the order b f f b b f.
ALTERNATING_MODELS = ("polar1-s",)
ALTERNATING_ROUNDS = 3


# Plain cuEq float64 MH-1 NaCl compiled while timed in the same-process A/B even after 1a-T13b's fix (the
# research engine's side this time, at the 7168 -> 7680 transition), but not in either single-engine
# process: it falls back to alternating processes too (conductor's condition 1).
ALTERNATING_CASES = {"polar1-s": ("polar1-s", SECONDARY_CRYSTALS, "hybrid"),
                     "mh1-cueq": (PRIMARY_MODEL, ("NaCl",), "cueq")}


def alternating_jobs(machine: Machine, run_root: Path, models=ALTERNATING_MODELS, rounds=ALTERNATING_ROUNDS) -> list:
    """Alternating single-engine processes, ``rounds`` per engine, 12 blocks each, in the order b f f b b f ...

    ``models`` are keys of ALTERNATING_CASES (a model label, or "mh1-cueq" for MH-1 on plain cuEq).
    """
    run_root = Path(run_root)
    jobs = []
    for slug in models:
        model, structures, backend = ALTERNATING_CASES[slug]
        order = [side for index in range(rounds)
                 for side in (("baseline", "fastmace") if index % 2 == 0 else ("fastmace", "baseline"))]
        counts = {"baseline": 0, "fastmace": 0}
        for side in order:
            counts[side] += 1
            jobs.append(_ab_job(machine, run_root, f"alt-{slug}-nvt-p{counts[side]}-{side}", model, structures,
                                (side,), blocks=AB_BLOCKS, backend=backend, dump=side == "fastmace",
                                note="alternating single-engine processes: the engines cannot share a process"))
    return jobs


# Long-range labels: the crystals where ResearchModule's own screen finds the model usable
# (studies/other_models/README.md, "Accuracy gate" and "NaNbO3"; conductor decision of 18 October 2026,
# SPEC §4.3: model-level failures are warnings, not engine failures). Each entry is one gate run:
# (job suffix, structures, configurations per cell mode, role). NbCl5 takes 4 configurations for every
# POLAR size (POLAR-1-S is float32-sensitive there; the screen used 4). "@2x2x2" is the screen's reduced
# cell where float64 does not fit (SPEC §12.3 passed-reduced). Role "model-limited" runs only to show
# the reference failing too (absurd pressures); it is no engine verdict.
LONG_RANGE_GATES = {
    "polar1-s": (("", ("NaCl",), 3, "counts"),
                 ("-NbCl5x4", ("NbCl5",), 4, "supplementary: float32-sensitive model, NaCl only counts")),
    "polar1-m": (("", ("NaCl", "NaNbO3"), 3, "counts"), ("-NbCl5x4", ("NbCl5",), 4, "counts")),
    "polar1-l": (("", ("NaCl", "NaNbO3@2x2x2"), 3, "counts (NaNbO3 reduced)"),
                 ("-NbCl5x4", ("NbCl5@2x2x2",), 4, "counts (reduced)")),
    "omol0-xl": (("", ("NaCl", "NbCl5@2x2x2"), 3, "counts (NbCl5 reduced)"),
                 ("-NaNbO3-model-limited", ("NaNbO3@2x2x2",), 1, "model-limited")),
}
# POLAR-1-S on NaNbO3 is already shown model-limited by its first gate run (references at 86-75,347 GPa).


# MACE-OFF23 large's float64 reference needs one 14.8 GiB allocation on paracetamol 3x3x3 (leto, 20 GB,
# 18 October 2026): that crystal is gated on 2x2x2 (SPEC §12.3 passed-reduced).
REDUCED_GATES = {
    "off23-large": (("", tuple(spec + "@2x2x2" if "paracetamol" in spec else spec for spec in ORGANIC_STRUCTURES), 3,
                     "counts (paracetamol reduced: the float64 reference does not fit)"),),
}


def gate_cases(label):
    """(job suffix, structures, configs, role) of each gate run of ``label``."""
    if label in LONG_RANGE_GATES:
        return LONG_RANGE_GATES[label]
    if label in REDUCED_GATES:
        return REDUCED_GATES[label]
    if label in MACE_OFF_LABELS:
        return (("", tuple(ORGANIC_STRUCTURES), 3, "counts"),)
    return (("", CRYSTALS, 3, "counts"),)


def gate_jobs(machine: Machine, run_root: Path, labels=GATE_LABELS, engine="fastmace", structures=None) -> list:
    """The gate runs (a driver with its reference and production children each) of every label.

    ``structures`` replaces each label's cases by one run on those structures (the polar stress check).
    """
    run_root = Path(run_root)
    jobs = []
    for label in labels:
        if label not in GATE_LABELS:
            raise ValueError(f"{label!r} is not one of the 19 labels")
        cases = gate_cases(label) if structures is None else (
            # job names are file names: each structure by its stem and repeat, never a path
            ("-" + "-".join(PurePosixPath(spec).name.replace(".", "_") for spec in structures), tuple(structures), 3,
             "check"),)
        for suffix, specs, configs, role in cases:
            name = f"gate-{label}{suffix}" + ("" if engine == "fastmace" else f"-{engine}")
            output = check_location(machine, run_root / name)
            production = (machine.python("mace_cueq_oeq") if engine == "fastmace"
                          else shlex.join(hybrid_launcher(machine, run_root)))
            paths = [str(machine.path(machine.research_module) / spec) if spec.startswith("Structures/") else spec
                     for spec in specs]
            argv = [machine.python("mace_cueq_latest"), _tool(machine, "force_stress_gate.py"), "run",
                    "--output", str(output), "--engine", engine, "--baseline", str(machine.path(machine.baseline)),
                    "--checkpoint", checkpoint_path(machine, label), "--model", label, "--structures", *paths,
                    "--configs", str(configs),
                    # references without OEq (plain cuEq), in the plain environment; production in the hybrid one
                    "--reference-command", machine.python("mace_cueq_latest"), "--production-command", production]
            jobs.append(Job(name=name, argv=tuple(argv), output=str(output), note=role))
    return jobs


def workflow_jobs(machine: Machine, run_root: Path, research_module=None, engines=("gpu", "fastmace")) -> list:
    """main.py smoke (NaNbO3, 0.5 ps to 400 K and back), --profile anneal on NaCl (1 ps ramps), --quench of it.

    As ResearchModule's qualify_workstation.sh steps 4 and 5, once per engine: ``--engine gpu`` through
    the hybrid launcher (the research engine needs it), ``--engine fastmace`` without it (FastMACE
    prepares the OEq runtime in-process; ResearchModule fb81217b). ``research_module`` is the tree with
    1a-T17's main.py.
    """
    run_root = Path(run_root)
    main = Path(research_module or machine.path(machine.research_module)) / "Experiments/Langevin_Thermostat/main.py"
    checkpoint = checkpoint_path(machine, PRIMARY_MODEL)
    jobs = []
    for engine in engines:
        smoke = check_location(machine, run_root / f"main-smoke-NaNbO3-{engine}")
        anneal = check_location(machine, run_root / f"anneal-NaCl-{engine}")
        common = [str(main), "--engine", engine, "--checkpoint", checkpoint]
        launcher = hybrid_launcher(machine, run_root) if engine == "gpu" else [machine.python("mace_cueq_oeq")]
        jobs += [
            Job(name=f"main-smoke-NaNbO3-{engine}", output=str(smoke), argv=tuple(
                launcher + common + ["--output", str(smoke), "--structures", "NaNbO3", "--heat-ps", "0.5",
                                     "--cool-ps", "0.5", "--peak-K", "400"])),
            Job(name=f"anneal-NaCl-{engine}", output=str(anneal), argv=tuple(
                launcher + common + ["--output", str(anneal), "--profile", "anneal", "--structures", "NaCl",
                                     "--heat-ps", "1", "--cool-ps", "1"])),
            Job(name=f"quench-NaCl-{engine}", output=str(anneal), argv=tuple(
                launcher + [str(main), "--checkpoint", checkpoint, "--quench", str(anneal),
                            "--quench-threads", str(os.cpu_count() or 1)])),
        ]
    return jobs


def nacl_rerun_jobs(machine: Machine, run_root: Path) -> list:
    """MH-1 NaCl again after 1a-T13b's mask_head fix: hybrid NVT in both build orders, and plain cuEq float64.

    The first runs compiled while timed at NaCl's first new edge bucket (7168 -> 7680) because both engines
    patch mask_head process-wide; FastMACE now reinstalls its function before each capture.
    """
    run_root = Path(run_root)
    note = "rerun after 1a-T13b (mask_head reinstalled before each capture)"
    return [
        _ab_job(machine, run_root, "ab-mh1-nvt-NaCl-r1", PRIMARY_MODEL, ("NaCl",), ("baseline", "fastmace"),
                blocks=AB_BLOCKS, note=note + "; baseline built first"),
        _ab_job(machine, run_root, "ab-mh1-nvt-NaCl-r2", PRIMARY_MODEL, ("NaCl",), ("fastmace", "baseline"),
                blocks=AB_BLOCKS, note=note + "; FastMACE built first"),
        _ab_job(machine, run_root, "ab-mh1-cueq-nvt", PRIMARY_MODEL, CUEQ_CASE["structures"], ("baseline", "fastmace"),
                blocks=AB_BLOCKS, backend="cueq", note=note + "; plain cuEq float64"),
    ]


def primary_jobs(machine: Machine, run_root: Path) -> list:
    """The primary set only (MH-1 NVT on the three crystals, three A/B processes) and the NPT case."""
    return [job for job in speed_jobs(machine, run_root) if job.name.startswith(("ab-mh1-nvt-r", "ab-mh1-npt"))]


def alternating_cueq_jobs(machine: Machine, run_root: Path) -> list:
    """MH-1 NaCl on plain cuEq float64 in alternating single-engine processes."""
    return alternating_jobs(machine, run_root, models=("mh1-cueq",))


# ------------------------------------------------------------------ phase 1b (plan task 1b-T23)

# The gate subset on the RTX 5060 Ti (plan 1b-T23): MH-1, POLAR-1-S, OMOL-0 XL and OFF24-M.
LOCAL_GATE_SUBSET = ("mh1-r2scan", "polar1-s", "omol0-xl", "off24-medium")
REGRESSION_STRUCTURES = ("NaCl", "NbCl5")  # acceptance item 5: MH-1 NaCl and NbCl5, hybrid NVT


def gate_1b_jobs(machine: Machine, run_root: Path, labels=GATE_LABELS) -> list:
    """Acceptance item 1: the rebuilt driver (fastmace.reference, selfcheck's rule, energy gated) through the
    minimal planner, on 1a's structures that count (the same cases as ``gate_jobs``)."""
    jobs = []
    for job in gate_jobs(machine, run_root, labels=labels):
        argv = list(job.argv)
        at = argv.index("--baseline")
        argv[at:at + 2] = ["--structures-root", argv[at + 1], "--planner", "minimal"]
        jobs.append(Job(name=job.name, argv=tuple(argv), output=job.output, note=job.note))
    return jobs


def gate_1b_subset_jobs(machine: Machine, run_root: Path) -> list:
    return gate_1b_jobs(machine, run_root, labels=LOCAL_GATE_SUBSET)


def _counting_structure(label):
    """One structure that counts for ``label`` (the in-library gate of acceptance item 1): NaCl for the
    crystal labels and the long-range ones (NaCl counts for every POLAR size and OMOL-0 XL), urea for MACE-OFF."""
    if label in MACE_OFF_LABELS:
        return ORGANIC_STRUCTURES[0]
    return "NaCl"


def library_gate_jobs(machine: Machine, run_root: Path, labels=GATE_LABELS) -> list:
    """Acceptance item 1, second half: the in-library gate (check="gate", NVT and NPT) once per label."""
    run_root = Path(run_root)
    jobs = []
    for label in labels:
        output = check_location(machine, run_root / f"library-gate-{label}")
        spec = _counting_structure(label)
        structure = str(machine.path(machine.research_module) / spec) if spec.startswith("Structures/") else spec
        argv = [machine.python("mace_cueq_oeq"), _tool(machine, "library_gate.py"), "--output", str(output),
                "--model", label, "--structure", structure, "--check", "gate"]
        jobs.append(Job(name=f"library-gate-{label}", argv=tuple(argv), output=str(output),
                        note="in-library reduced gate (1b-T16), captured replay"))
    return jobs


def _regression_job(machine, run_root, name, sides, *, blocks, pattern="abab", note=""):
    output = check_location(machine, run_root / name)
    argv = [machine.python("mace_cueq_oeq"), _tool(machine, "ab_fastmace.py"), "--output", str(output),
            "--model", PRIMARY_MODEL, "--checkpoint", checkpoint_path(machine, PRIMARY_MODEL),
            "--structures", *REGRESSION_STRUCTURES, "--sides", *sides, "--warmup", str(WARMUP), "--block", str(BLOCK),
            "--blocks", str(blocks), "--pattern", pattern, "--dump-forces"]
    if "1a" in sides:
        argv += ["--renamed", str(machine.path(machine.versions))]
    return Job(name=name, argv=tuple(argv), output=str(output), timing=True, note=note)


def regression_jobs(machine: Machine, run_root: Path) -> list:
    """Acceptance item 5: 1b against the 1a tag in one process (three A/B processes), then one control per side."""
    run_root = Path(run_root)
    jobs = [
        _regression_job(machine, run_root, "regression-ab-abab", ("1a", "1b"), blocks=AB_BLOCKS,
                        note="ABAB, 1a built first"),
        _regression_job(machine, run_root, "regression-ab-1b-first", ("1b", "1a"), blocks=AB_BLOCKS,
                        note="ABAB, 1b built first"),
        _regression_job(machine, run_root, "regression-ab-abba", ("1a", "1b"), blocks=AB_BLOCKS, pattern="abba",
                        note="ABBA, 1a built first"),
    ]
    jobs += [_regression_job(machine, run_root, f"regression-control-{side}", (side,), blocks=CONTROL_BLOCKS,
                             note="one version alone: setup per stage, forces outside an A/B process")
             for side in ("1a", "1b")]
    return jobs


def frontends_jobs(machine: Machine, run_root: Path) -> list:
    """Acceptance item 4: Simulation, fm.md and fastmace.ase.Langevin on MH-1 NaCl 3x3x3 in one process."""
    output = check_location(machine, Path(run_root) / "frontends-mh1-nacl")
    argv = [machine.python("mace_cueq_oeq"), _tool(machine, "ab_frontends.py"), "--output", str(output),
            "--model", PRIMARY_MODEL, "--structure", "NaCl", "--blocks", str(AB_BLOCKS), "--warmup", str(WARMUP),
            "--block", str(BLOCK)]
    return [Job(name="frontends-mh1-nacl", argv=tuple(argv), output=str(output), timing=True,
                note="three front ends, round-robin blocks")]


def gpu_suite_jobs(machine: Machine, run_root: Path) -> list:
    """Acceptance items 2 and 3: the full GPU suite in one process (registry and capture state, canonical
    restart, the hybrid restart statistics), with hybrid children in the OEq environment."""
    run_root = Path(run_root)
    output = check_location(machine, run_root / "gpu-suite")
    # pytest creates --basetemp but not its parent: keep it directly under the run root (the 21 setup errors of
    # run 1b-T23-local-20261019 came from a basetemp inside the not yet created output directory).
    basetemp = check_location(machine, run_root / "gpu-suite-pytest")
    tests = machine.path(machine.fastmace) / "tests" / "gpu"
    argv = [machine.python("mace_cueq_latest"), "-m", "pytest", str(tests), "-p", "no:cacheprovider",
            f"--basetemp={basetemp}", "--durations=40", "-rfEs", f"--junitxml={output / 'junit.xml'}"]
    # The comparisons against the research engine need its export, and test_reference_gpu 1a-T19's stored POLAR-1-S
    # references (on the NAS only); without them 77 tests skip (run 1b-T23-murphy-20261019).
    env = {"FASTMACE_HYBRID_PYTHON": machine.python("mace_cueq_oeq"),
           "FASTMACE_BASELINE": str(machine.path(machine.baseline)),
           "FASTMACE_1A_T19_POLAR1S": str(machine.path(machine.root) / "1a-T19-gate-leto-20261018" / "gate-polar1-s")}
    return [Job(name="gpu-suite", argv=tuple(argv), output=str(output), env=env, note="full tier, one process")]


def adapter_jobs(machine: Machine, run_root: Path) -> list:
    """Acceptance item 6: ResearchModule's adapter tests against this checkout, reading ResearchModule only
    (no bytecode, no pytest cache, temporary files outside its tree; conductor answer to 1b-T23)."""
    run_root = Path(run_root)
    output = check_location(machine, run_root / "adapter-tests")
    basetemp = check_location(machine, run_root / "adapter-tests-pytest")  # pytest creates it; its parent exists
    tests = machine.path(machine.research_module) / "Experiments" / "Langevin_Thermostat" / "tests" / "core"
    argv = [machine.python("mace_cueq_oeq"), "-m", "pytest", str(tests / "test_fastmace_engine.py"),
            "-p", "no:cacheprovider", f"--basetemp={basetemp}", f"--junitxml={output / 'junit.xml'}"]
    return [Job(name="adapter-tests", argv=tuple(argv), output=str(output), env={"PYTHONDONTWRITEBYTECODE": "1"},
                note="ResearchModule tests/core/test_fastmace_engine.py, read only")]


def compile_checks_jobs(machine: Machine, run_root: Path) -> list:
    """Items left to 1b-T23: compiled PolarMACE NPT across restart points (1b-T14), and fm.calc's compiles at
    its first calls on both backends (1b-T18, measured only)."""
    run_root = Path(run_root)
    jobs = []
    for name, environment, argv in (
            ("compile-polar1s-npt-restart", "mace_cueq_oeq", ["polar-npt", "--model", "polar1-s"]),
            ("compile-fm-calc-hybrid", "mace_cueq_oeq", ["calc", "--model", PRIMARY_MODEL]),
            ("compile-fm-calc-cueq", "mace_cueq_latest", ["calc", "--model", PRIMARY_MODEL]),
            ("setup-mace-imported-first", "mace_cueq_oeq", ["mace-first", "--model", PRIMARY_MODEL])):
        output = check_location(machine, run_root / name)
        jobs.append(Job(name=name, argv=(machine.python(environment), _tool(machine, "compile_checks.py"), *argv[:1],
                                         "--output", str(output), *argv[1:]), output=str(output), timing=True))
    return jobs


# --- phase 2 (plan task 2-T23) ------------------------------------------------------------------------------


# Options whose default changed after fastmace-1b by owner decision (optimisation round 1, 20 October 2026: OPT-T02
# stages A and B on by default). Item 8 of the combined acceptance expects exactly these plan differences.
ROUND_OPTION_CHANGES = ("gated_rows", "segment_gemm")


def _regression2_job(machine, run_root, name, sides, *, blocks, pattern="abab", note=""):
    output = check_location(machine, run_root / name)
    argv = [machine.python("mace_cueq_oeq"), _tool(machine, "ab_fastmace.py"), "--release", "1b",
            "--output", str(output), "--model", PRIMARY_MODEL, "--checkpoint", checkpoint_path(machine, PRIMARY_MODEL),
            "--structures", *REGRESSION_STRUCTURES, "--sides", *sides, "--warmup", str(WARMUP), "--block", str(BLOCK),
            "--blocks", str(blocks), "--pattern", pattern, "--dump-forces",
            "--allow-option-changes", *ROUND_OPTION_CHANGES]
    if "1b" in sides:
        argv += ["--renamed", str(machine.path(machine.versions))]
    return Job(name=name, argv=tuple(argv), output=str(output), timing=True, note=note)


def regression2_jobs(machine: Machine, run_root: Path) -> list:
    """Phase 2 acceptance item 8: this checkout (side 2) against the tag fastmace-1b in one process, MH-1 NaCl
    and NbCl5 hybrid NVT, 12 blocks, three A/B processes, then one control per side."""
    run_root = Path(run_root)
    jobs = [
        _regression2_job(machine, run_root, "regression2-ab-abab", ("1b", "2"), blocks=AB_BLOCKS,
                         note="ABAB, 1b built first"),
        _regression2_job(machine, run_root, "regression2-ab-2-first", ("2", "1b"), blocks=AB_BLOCKS,
                         note="ABAB, 2 built first"),
        _regression2_job(machine, run_root, "regression2-ab-abba", ("1b", "2"), blocks=AB_BLOCKS, pattern="abba",
                         note="ABBA, 1b built first"),
    ]
    jobs += [_regression2_job(machine, run_root, f"regression2-control-{side}", (side,), blocks=CONTROL_BLOCKS,
                              note="one version alone: setup, forces outside an A/B process")
             for side in ("1b", "2")]
    return jobs


def regression2_nacl_jobs(machine: Machine, run_root: Path) -> list:
    """Three more NaCl-only A/B processes for item 8 on murphy (conductor, 20 October 2026, after one of the first
    three NaCl processes ended 0.004 points over the floor): 1b built first in two of them."""
    run_root = Path(run_root)
    jobs = [
        _regression2_job(machine, run_root, "regression2-nacl-abab-1b-first", ("1b", "2"), blocks=AB_BLOCKS,
                         note="NaCl only, ABAB, 1b built first"),
        _regression2_job(machine, run_root, "regression2-nacl-abab-2-first", ("2", "1b"), blocks=AB_BLOCKS,
                         note="NaCl only, ABAB, 2 built first"),
        _regression2_job(machine, run_root, "regression2-nacl-abba-1b-first", ("1b", "2"), blocks=AB_BLOCKS,
                         pattern="abba", note="NaCl only, ABBA, 1b built first"),
    ]
    out = []
    for job in jobs:
        argv = list(job.argv)
        start = argv.index("--structures") + 1
        argv[start:start + len(REGRESSION_STRUCTURES)] = ["NaCl"]
        out.append(replace(job, argv=tuple(argv)))
    return out


# 1b-T14's protocols: plain cuEq float64 500 steps (positions every 50), hybrid 20,000 steps (every 20), two seeds;
# the target ramps from 300 to 600 K.
TRAJECTORY_CASES = {"cueq": dict(environment="mace_cueq_latest", steps=500, every=50, seeds=(20261019,),
                                 positions=True),
                    "hybrid": dict(environment="mace_cueq_oeq", steps=20_000, every=20, seeds=(20261001, 20261101),
                                   positions=False)}


def _trajectory2_job(machine, run_root, backend, side, suffix=""):
    case = TRAJECTORY_CASES[backend]
    name = f"trajectory2-{backend}-{side}{suffix}"
    output = check_location(machine, run_root / name)
    argv = [machine.python(case["environment"]), _tool(machine, "trajectory_versions.py"), "run", "--side", side,
            "--output", str(output), "--model", PRIMARY_MODEL, "--checkpoint", checkpoint_path(machine, PRIMARY_MODEL),
            "--structure", "NaCl", "--backend", backend, "--steps", str(case["steps"]), "--every", str(case["every"]),
            "--seeds", *(str(seed) for seed in case["seeds"])]
    if case["positions"]:
        argv.append("--positions")
    if side == "1b":
        argv += ["--renamed", str(machine.path(machine.versions))]
    return Job(name=name, argv=tuple(argv), output=str(output),
               note="default trajectory, one version per process (1b-T14's measures)")


def trajectory2_jobs(machine: Machine, run_root: Path) -> list:
    """Phase 2 acceptance item 8, trajectories: MH-1 NaCl 3x3x3 on this checkout and on fastmace-1b, one version per
    process; plain cuEq float64 (with a repeat of this checkout as the floor) and hybrid statistics. Untimed."""
    run_root = Path(run_root)
    return [_trajectory2_job(machine, run_root, "cueq", "1b"), _trajectory2_job(machine, run_root, "cueq", "2"),
            _trajectory2_job(machine, run_root, "cueq", "2", "-repeat"),
            _trajectory2_job(machine, run_root, "hybrid", "1b"), _trajectory2_job(machine, run_root, "hybrid", "2")]


def runner_statistics_jobs(machine: Machine, run_root: Path) -> list:
    """Phase 2 acceptance item 4, hybrid: MH-1 NaCl in subprocess-runner children (after a compiled plain cuEq)
    against a fresh process, 20,000 steps, two seeds, 1b-T14's measure. Untimed."""
    jobs = []
    for case in ("fresh", "runner"):
        name = f"runner-statistics-{case}"
        output = check_location(machine, Path(run_root) / name)
        argv = [machine.python("mace_cueq_oeq"), _tool(machine, "runner_statistics.py"), "run", "--case", case,
                "--output", str(output), "--model", PRIMARY_MODEL]
        jobs.append(Job(name=name, argv=tuple(argv), output=str(output), note="hybrid, 1b-T14's measure"))
    return jobs


REPEAT_SEEDS = (20261001, 20261101, 20261201, 20261301)


def runner_statistics_repeat_jobs(machine: Machine, run_root: Path) -> list:
    """Item 4's repeat (owner, 20 October 2026, after the first comparison was dominated by the heating transient
    from 0 K): one equilibrated start (300 K: 2 ps at 0.5 fs, then 10 ps at 2 fs), then fresh and runner from it,
    four seeds. Untimed."""
    run_root = Path(run_root)
    equilibrated = check_location(machine, run_root / "runner-repeat-equilibrate")
    jobs = [Job(name="runner-repeat-equilibrate", output=str(equilibrated), note="the shared start state",
                argv=(machine.python("mace_cueq_oeq"), _tool(machine, "runner_statistics.py"), "equilibrate",
                      "--output", str(equilibrated), "--model", PRIMARY_MODEL))]
    for case in ("fresh", "runner"):
        name = f"runner-repeat-{case}"
        output = check_location(machine, run_root / name)
        argv = [machine.python("mace_cueq_oeq"), _tool(machine, "runner_statistics.py"), "run", "--case", case,
                "--output", str(output), "--model", PRIMARY_MODEL, "--start", str(equilibrated / "start.extxyz"),
                "--seeds", *(str(seed) for seed in REPEAT_SEEDS)]
        jobs.append(Job(name=name, argv=tuple(argv), output=str(output), note="hybrid, equilibrated start, 4 seeds"))
    return jobs


def _tier_job(machine, run_root, name, marker, *, timing, note, files=None):
    output = check_location(machine, run_root / name)
    basetemp = check_location(machine, run_root / f"{name}-pytest")
    gpu = machine.path(machine.fastmace) / "tests" / "gpu"
    tests = [str(gpu / file) for file in files] if files else [str(gpu)]
    argv = [machine.python("mace_cueq_latest"), "-m", "pytest", *tests, "-m", marker, "-p", "no:cacheprovider",
            f"--basetemp={basetemp}", "--durations=60", "-rfEs", f"--junitxml={output / 'junit.xml'}"]
    env = {"FASTMACE_HYBRID_PYTHON": machine.python("mace_cueq_oeq"),
           "FASTMACE_BASELINE": str(machine.path(machine.baseline)),
           "FASTMACE_RESULTS": str(output / "results")}
    # 1a-T19's stored POLAR-1-S references are on the NAS only; elsewhere the test skips (run 2-T23-local-20261020:
    # the variable pointed at a missing directory and the test failed reading it).
    polar = machine.path(machine.root) / "1a-T19-gate-leto-20261018" / "gate-polar1-s"
    if polar.is_dir():
        env["FASTMACE_1A_T19_POLAR1S"] = str(polar)
    return Job(name=name, argv=tuple(argv), output=str(output), env=env, timing=timing, note=note)


def acceptance_tier_jobs(machine: Machine, run_root: Path) -> list:
    """The acceptance GPU tier (docs/development.md, "Test tiers") on final main, one process. It holds the
    planner's step-time assertions (2-T14d), so it needs an idle GPU."""
    return [_tier_job(machine, Path(run_root), "acceptance-tier", "acceptance", timing=True,
                      note="pytest tests/gpu -m acceptance")]


# The combined acceptance's own GPU files (optimisation round 1): the non-acceptance tests of the restart,
# reproducible-mode and runner files with the round's rewrites on by default, stage B's kernel test, and OPT-T06/T07's
# GPU tests if merged at the freeze (their files are added here then). The acceptance-marked tests of these files run
# in the acceptance tier.
ROUND_TEST_FILES = ("test_canonical_restart_gpu.py", "test_reproducible_gpu.py", "test_subprocess_runner_gpu.py",
                    "test_segment_gemm_cuda.py")


def round_tests_jobs(machine: Machine, run_root: Path) -> list:
    """The round's GPU files outside the acceptance tier, one process (untimed)."""
    present = [file for file in ROUND_TEST_FILES if (machine.path(machine.fastmace) / "tests" / "gpu" / file).is_file()]
    return [_tier_job(machine, Path(run_root), "round-tests", "not acceptance", timing=False, files=present,
                      note="restart, reproducible, runner and round files outside the acceptance tier")]


# The 19-label gate split over the two cards: labels whose float64 references fit the RTX 5060 Ti's 16 GB (1b-T23's
# local runs and 1b-T03's peaks) run there; the polar, MACE-OFF and MP-0 large labels run on murphy's 20 GB.
LOCAL_GATE_LABELS = ("mp0-medium", "mp0b3-medium", "mpa0-medium", "omat0-medium", "matpes-pbe", "matpes-r2scan",
                     "mh1-omat", "mh1-mp", "mh1-r2scan", "omol0-xl")
MURPHY_GATE_LABELS = tuple(label for label in GATE_LABELS if label not in LOCAL_GATE_LABELS)


def gate_local_jobs(machine: Machine, run_root: Path) -> list:
    return gate_1b_jobs(machine, run_root, labels=LOCAL_GATE_LABELS)


def gate_murphy_jobs(machine: Machine, run_root: Path) -> list:
    return gate_1b_jobs(machine, run_root, labels=MURPHY_GATE_LABELS)


def full_tier_jobs(machine: Machine, run_root: Path) -> list:
    """The full GPU tier (merge tier) on final main, one process; no step-time assertions."""
    return [_tier_job(machine, Path(run_root), "full-tier", "not acceptance", timing=False,
                      note='pytest tests/gpu -m "not acceptance"')]


SUITES = {"speed": speed_jobs, "primary": primary_jobs, "nacl-rerun": nacl_rerun_jobs, "alternating": alternating_jobs,
          "alternating-cueq": alternating_cueq_jobs,
          "gate": gate_jobs, "workflow": workflow_jobs,
          "gate-1b": gate_1b_jobs, "gate-1b-subset": gate_1b_subset_jobs, "library-gate": library_gate_jobs,
          "regression": regression_jobs, "frontends": frontends_jobs, "gpu-suite": gpu_suite_jobs,
          "adapter-tests": adapter_jobs, "compile-checks": compile_checks_jobs,
          "regression2": regression2_jobs, "regression2-nacl": regression2_nacl_jobs, "trajectory2": trajectory2_jobs,
          "runner-statistics": runner_statistics_jobs,
          "runner-repeat": runner_statistics_repeat_jobs,
          "acceptance-tier": acceptance_tier_jobs, "full-tier": full_tier_jobs, "round-tests": round_tests_jobs,
          "gate-local": gate_local_jobs, "gate-murphy": gate_murphy_jobs}
