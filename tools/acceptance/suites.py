"""Phase 1a acceptance runs (plan task 1a-T19): machines, suites and the command of every job.

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
from dataclasses import dataclass, field
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

    def path(self, value: str) -> Path:
        return Path(os.path.expanduser(value))

    def python(self, environment: str) -> str:
        return str(self.path(self.environments) / environment / "bin" / "python")


MACHINES = {
    "local": Machine(
        name="local", kind="local", root="~/fastmace-acceptance", fastmace="/mnt/d/Margraf/FastMACE",
        baseline="~/fastmace-baseline/1a-T18", models="~/models", research_module="/mnt/g/Margraf/ResearchModule",
        environments="~/environments", cache="~/fastmace-cache/acceptance",
        windows_smi="/mnt/c/Windows/System32/nvidia-smi.exe"),
    **{name: Machine(
        name=name, kind="workstation", root=str(NAS_HOME / "fastmace-acceptance"),
        fastmace=str(NAS_HOME / "FastMACE"), baseline=str(NAS_HOME / "fastmace-baseline" / "1a-T18"),
        models=str(NAS_HOME / "ResearchModule" / "Models"), research_module=str(NAS_HOME / "ResearchModule"),
        environments=str(NAS_HOME / "environments"), cache=str(NAS_HOME / "fastmace-cache" / f"acceptance-{name}"))
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


def gate_structures(machine: Machine, label: str) -> list:
    if label in MACE_OFF_LABELS:
        return [str(machine.path(machine.research_module) / spec) for spec in ORGANIC_STRUCTURES]
    return list(CRYSTALS)


def gate_jobs(machine: Machine, run_root: Path, labels=GATE_LABELS, engine="fastmace") -> list:
    """One gate process (a driver with its reference and production children) per label."""
    run_root = Path(run_root)
    jobs = []
    for label in labels:
        if label not in GATE_LABELS:
            raise ValueError(f"{label!r} is not one of the 19 labels")
        name = f"gate-{label}" + ("" if engine == "fastmace" else f"-{engine}")
        output = check_location(machine, run_root / name)
        production = (machine.python("mace_cueq_oeq") if engine == "fastmace"
                      else shlex.join(hybrid_launcher(machine, run_root)))
        argv = [machine.python("mace_cueq_latest"), _tool(machine, "force_stress_gate.py"), "run",
                "--output", str(output), "--engine", engine, "--baseline", str(machine.path(machine.baseline)),
                "--checkpoint", checkpoint_path(machine, label), "--model", label,
                "--structures", *gate_structures(machine, label),
                # references without OEq (plain cuEq), in the plain environment; production in the hybrid one
                "--reference-command", machine.python("mace_cueq_latest"), "--production-command", production]
        jobs.append(Job(name=name, argv=tuple(argv), output=str(output),
                        note="MACE-OFF: organic set" if label in MACE_OFF_LABELS else "benchmark crystals"))
    return jobs


def workflow_jobs(machine: Machine, run_root: Path, research_module=None, engines=("gpu", "fastmace")) -> list:
    """main.py smoke (NaNbO3, 0.5 ps to 400 K and back), --profile anneal on NaCl (1 ps ramps), --quench of it.

    As ResearchModule's qualify_workstation.sh steps 4 and 5, once per engine, all through the hybrid
    launcher (the research engine's default needs it). ``research_module`` is the tree with 1a-T17's main.py.
    """
    run_root = Path(run_root)
    main = Path(research_module or machine.path(machine.research_module)) / "Experiments/Langevin_Thermostat/main.py"
    checkpoint = checkpoint_path(machine, PRIMARY_MODEL)
    jobs = []
    for engine in engines:
        smoke = check_location(machine, run_root / f"main-smoke-NaNbO3-{engine}")
        anneal = check_location(machine, run_root / f"anneal-NaCl-{engine}")
        common = [str(main), "--engine", engine, "--checkpoint", checkpoint]
        launcher = hybrid_launcher(machine, run_root)
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


SUITES = {"speed": speed_jobs, "gate": gate_jobs, "workflow": workflow_jobs}
