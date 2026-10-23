"""Run profiles as data (SPEC §10.5): today's ``slow``, ``fast``, ``ultrafast`` and ``anneal``.

The research engine's ``workflow.PROFILES``: the schedule, the output intervals, the initial
relaxation and, for ``anneal``, the quench settings (``docs/assets/Simulated_Annealing.txt``). The
relaxation and quench settings are data only here; nothing in this module runs them.

Profiles from files: TOML (``tomllib``, ``tomli`` before Python 3.11) or YAML (PyYAML's
``safe_load`` only, imported when needed).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fastmace.protocols.schedule import Schedule, ramp

OPTIMIZERS = ("LBFGS", "BFGS")
QUENCH_KEYS = ("min_K", "bin_K", "per_bin", "fmax", "optimizer", "max_steps")
ENSEMBLES = ("NVT", "NPT")  # NVE comes with its integrator (phase 4)
DISPERSIONS = ("none", "d3bj")
_PROFILE_KEYS = {"name", "segments", "sample_interval", "trajectory_interval", "relax_fmax", "relax_optimizer",
                 "quench", "ensemble", "pressure_GPa", "dispersion"}


def _interval(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and at least 0, got {value!r}")
    return float(value)


@dataclass(frozen=True, eq=True)
class Profile:
    """A named schedule with its output intervals, initial relaxation and quench settings.

    ``relax_fmax`` 0 means no initial relaxation (fixed cell, eV/Å). ``quench`` is ``None`` or a
    mapping with ``QUENCH_KEYS`` (frames per ``bin_K`` bin of the target from ``min_K``, quenched to
    ``fmax`` with ``optimizer``, at most ``max_steps``). Relaxation and quench keep the cell fixed;
    after NPT each frame is quenched at its own cell.

    ``ensemble`` is ``"NVT"`` or ``"NPT"`` (``fm.md``'s; any case). ``pressure_GPa`` is NPT's
    constant hydrostatic target, ``None`` for NVT and 0 GPa by default for NPT; pressure schedules
    are refused until the pressure setter (plan task 3b-T05). ``dispersion`` is ``"none"`` or
    ``"d3bj"`` (``fm.md``'s, SPEC §7.7).
    """

    name: str
    schedule: Schedule
    sample_interval: int = 100
    trajectory_interval: int = 100
    relax_fmax: float = 0.0
    relax_optimizer: str = "LBFGS"
    quench: Mapping[str, Any] | None = None
    ensemble: str = "NVT"
    pressure_GPa: float | None = None
    dispersion: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a profile needs a name, got {self.name!r}")
        if not isinstance(self.schedule, Schedule):
            raise ValueError(f"schedule must be a Schedule, got {type(self.schedule).__name__}")
        _interval(self.sample_interval, "sample_interval")
        _interval(self.trajectory_interval, "trajectory_interval")
        object.__setattr__(self, "relax_fmax", _nonnegative(self.relax_fmax, "relax_fmax"))
        if self.relax_optimizer not in OPTIMIZERS:
            raise ValueError(f"relax_optimizer must be one of {OPTIMIZERS}, got {self.relax_optimizer!r}")
        if self.quench is not None:
            object.__setattr__(self, "quench", MappingProxyType(_quench(self.quench)))
        if not isinstance(self.ensemble, str) or self.ensemble.upper() not in ENSEMBLES:
            raise ValueError(f"ensemble must be one of {ENSEMBLES}, got {self.ensemble!r}")
        object.__setattr__(self, "ensemble", self.ensemble.upper())
        object.__setattr__(self, "pressure_GPa", _pressure(self.pressure_GPa, self.ensemble))
        if self.dispersion not in DISPERSIONS:
            raise ValueError(f"dispersion must be one of {DISPERSIONS}, got {self.dispersion!r}")

    def md_options(self) -> dict[str, Any]:
        """``fm.md``'s schedule, ensemble, pressure and dispersion arguments of this profile."""
        options: dict[str, Any] = {"schedule": self.schedule, "ensemble": self.ensemble}
        if self.ensemble == "NPT":
            options["pressure_GPa"] = self.pressure_GPa
        if self.dispersion != "none":
            options["dispersion"] = self.dispersion
        return options

    def __hash__(self) -> int:
        return hash((self.name, self.schedule))


def _pressure(value: Any, ensemble: str) -> float | None:
    if ensemble != "NPT":
        if value is not None:
            raise ValueError(f"pressure_GPa needs ensemble NPT, got {value!r} with {ensemble}")
        return None
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple, Mapping)):
        raise ValueError("pressure_GPa takes a constant; pressure schedules come with the pressure setter "
                         "(plan task 3b-T05)")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"pressure_GPa must be a finite number, got {value!r}")
    return float(value)


def _quench(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != set(QUENCH_KEYS):
        raise ValueError(f"quench needs exactly the keys {QUENCH_KEYS}, got {data!r}")
    quench = {key: _nonnegative(data[key], f"quench {key}") for key in ("min_K", "bin_K", "fmax")}
    if quench["bin_K"] <= 0 or quench["fmax"] <= 0:
        raise ValueError("quench bin_K and fmax must be positive")
    quench["per_bin"] = _interval(data["per_bin"], "quench per_bin")
    quench["max_steps"] = _interval(data["max_steps"], "quench max_steps")
    if data["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"quench optimizer must be one of {OPTIMIZERS}, got {data['optimizer']!r}")
    quench["optimizer"] = data["optimizer"]
    return {key: quench[key] for key in QUENCH_KEYS}


def _research_profile(name: str, heat_ps: float, cool_ps: float, peak_K: float = 400.0, final_K: float = 0.0,
                      interval: int = 100, relax_fmax: float = 0.0, quench: Mapping[str, Any] | None = None
                      ) -> Profile:
    return Profile(name, ramp(0.0, peak_K, ps=heat_ps) + ramp(peak_K, final_K, ps=cool_ps),
                   sample_interval=interval, trajectory_interval=interval, relax_fmax=relax_fmax, quench=quench)


# workflow.PROFILES. slow, fast and ultrafast: 0 -> 400 -> 0 K with the 1:8 heating/cooling ratio.
# anneal: 0 -> 1000 K in 20 ps, then 1000 -> 250 K over 250 ps (3 K/ps), every 20th step saved, a tight
# initial relaxation and the quench defaults of main.py --quench (Simulated_Annealing.txt).
PROFILES: Mapping[str, Profile] = MappingProxyType({
    "slow": _research_profile("slow", 30.0, 240.0),
    "fast": _research_profile("fast", 3.0, 24.0),
    "ultrafast": _research_profile("ultrafast", 1.0, 8.0),
    "anneal": _research_profile("anneal", 20.0, 250.0, peak_K=1000.0, final_K=250.0, interval=20,
                                relax_fmax=0.01,
                                quench={"min_K": 500.0, "bin_K": 100.0, "per_bin": 2, "fmax": 0.01,
                                        "optimizer": "LBFGS", "max_steps": 5000}),
})


def profile_from_data(data: Mapping[str, Any]) -> Profile:
    """A profile from a mapping: ``name``, ``segments`` (``Schedule.from_data``) and optional
    ``sample_interval``, ``trajectory_interval``, ``relax_fmax``, ``relax_optimizer``, ``quench``,
    ``ensemble``, ``pressure_GPa``, ``dispersion``.
    Unknown keys and invalid values raise ``ValueError``."""
    if not isinstance(data, Mapping):
        raise ValueError(f"a profile must be a mapping, got {type(data).__name__}")
    unknown = set(data) - _PROFILE_KEYS
    if unknown:
        raise ValueError(f"unknown profile keys {sorted(unknown)}; known: {sorted(_PROFILE_KEYS)}")
    if "name" not in data or "segments" not in data:
        raise ValueError("a profile needs name and segments")
    options = {key: data[key] for key in data if key not in ("name", "segments")}
    return Profile(data["name"], Schedule.from_data(data["segments"]), **options)


def _read_toml(path: Path) -> Any:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:
        raise ImportError(f"reading the YAML profile {path} needs PyYAML (pip install pyyaml); "
                          "TOML profiles need nothing extra") from None
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_profile(path: str | Path) -> Profile:
    """A profile from a ``.toml``, ``.yaml`` or ``.yml`` file (YAML through ``safe_load`` only)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".toml":
        data = _read_toml(path)
    elif suffix in (".yaml", ".yml"):
        data = _read_yaml(path)
    else:
        raise ValueError(f"profile files are .toml or .yaml/.yml, got {path.name}")
    return profile_from_data(data)
