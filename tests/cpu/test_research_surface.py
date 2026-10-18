"""The FastMACE surface ResearchModule's ``main.py`` uses (plan task 1b-T01, "main.py keeps running").

ResearchModule's adapter ``Experiments/Langevin_Thermostat/langevin/fastmace_engine.py`` and its tests
(``tests/core/test_fastmace_engine.py``) import these names and call them with these keywords.
``main.py --engine fastmace`` is the production default since 19 October 2026, so a 1b task that
moves one of them leaves a re-export, and this test keeps it honest. Removing a name is a
cross-repository task.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

# (module, names) exactly as the adapter and its tests import them.
IMPORTS = (
    ("fastmace.config", ("EngineConfig", "IntegratorConfig", "NeighbourConfig", "PolarConfig", "PrecisionConfig",
                         "RewriteConfig")),
    ("fastmace.planner.fixed", ("FixedPlanner", "PlanRefused", "REWRITE_FIELDS", "auto_options_off")),
    ("fastmace.planner.plan", ("Hardware", "Plan")),
    ("fastmace.model.compile", ("CompiledModelCache",)),
    ("fastmace.engine.base", ("Engine", "Snapshot")),
    ("fastmace.provenance", ("model_block", "fastmace_block")),
    ("fastmace.io.trajectory", ("observables", "truncate_outputs", "wait")),
    ("fastmace.io.restart", ("state_arrays", "load_state", "read_restart")),
)


@pytest.mark.parametrize(("module", "names"), IMPORTS, ids=[module for module, _ in IMPORTS])
def test_adapter_imports(module, names):
    loaded = importlib.import_module(module)
    missing = [name for name in names if not hasattr(loaded, name)]
    assert not missing, f"{module} lost {missing}; leave a re-export (plan phase-1b, working principles)"


def test_fastmace_importable_as_a_package():
    import fastmace

    assert fastmace.__file__  # the adapter locates the checkout from it


def _accepts(function, *names):
    parameters = inspect.signature(function).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    return [name for name in names if name not in parameters]


def test_config_keywords_the_adapter_passes():
    from fastmace import config

    calls = {
        config.EngineConfig: ("backend", "compile_mode", "device", "pipelined", "async_output", "speculative_graph",
                              "cueq_conv_fusion"),
        config.PrecisionConfig: ("dtype",),
        config.NeighbourConfig: ("skin", "sort_receivers", "gpu_neighbour_list"),
        config.PolarConfig: ("fukui_regularisation", "dipole_cap"),
        config.IntegratorConfig: ("ensemble", "timestep_fs", "friction", "seed", "fixcm", "pressure_GPa", "cell_mode",
                                  "barostat_tau_fs", "barostat_mass_factor"),
    }
    for cls, names in calls.items():
        assert not _accepts(cls, *names), cls.__name__
    # Every other field keeps a default, so the adapter's constructions stay valid.
    for cls in calls:
        required = [f.name for f in dataclasses.fields(cls)
                    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
        assert not required, (cls.__name__, required)


def test_rewrite_fields_name_rewrite_config_fields():
    from fastmace.config import RewriteConfig
    from fastmace.planner.fixed import REWRITE_FIELDS

    fields = {field.name for field in dataclasses.fields(RewriteConfig)}
    assert set(REWRITE_FIELDS) <= fields


def test_fixed_planner_methods_and_keywords():
    from fastmace.planner.fixed import FixedPlanner

    assert not _accepts(FixedPlanner, "rewrites", "engine", "precision", "neighbours", "polar", "environ")
    for name in ("configure_process", "hardware", "load", "plan", "build"):
        assert callable(getattr(FixedPlanner, name)), name
    assert not _accepts(FixedPlanner.load, "source", "checkpoint", "trusted")
    assert not _accepts(FixedPlanner.plan, "traits", "atoms", "integrator", "hardware")
    assert not _accepts(FixedPlanner.build, "plan", "model", "atoms", "temperature_K", "compile_cache")


def test_plan_fields_the_adapter_reads():
    from fastmace.planner.plan import Hardware, Plan

    fields = {field.name for field in dataclasses.fields(Plan)}
    assert {"rewrites", "engine", "precision", "neighbours", "polar", "integrator", "decisions", "requested",
            "auto_options_off"} <= fields
    assert Hardware("cpu", (0, 0), 0).name == "cpu"  # positional, as the adapter's tests build it


def test_engine_methods_the_adapter_calls():
    from fastmace.engine.base import Engine, Snapshot

    for name in ("step", "flush", "snapshot", "sync", "info"):
        assert callable(getattr(Engine, name)), name
    assert callable(Snapshot.wait)


def test_io_and_provenance_keywords():
    from fastmace import provenance
    from fastmace.io import restart, trajectory

    assert not _accepts(restart.state_arrays, "state", "engine", "integrator_state", "variable_cell")
    assert not _accepts(restart.load_state, "restart", "state", "integrator")
    assert not _accepts(restart.read_restart, "path", "engine")
    assert not _accepts(trajectory.observables, "values", "natoms", "variable_cell")
    assert not _accepts(trajectory.truncate_outputs, "directory", "step", "frame_steps")
    assert not _accepts(provenance.model_block, "loaded", "rewrites", "engine", "neighbours", "polar",
                        "auto_options_off")
    assert not _accepts(provenance.fastmace_block, "device")
