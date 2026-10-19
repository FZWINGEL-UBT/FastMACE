"""``fm.md`` on a MACE other than 0.3.16 (plan task 2-T04, phase 2 acceptance 1, SPEC §18 decision 2).

Runs only where the installed mace-torch is not the qualified 0.3.16, which is
``~/environments/fastmace_mace_other`` (mace-torch 0.3.15; the recipe is in
``docs/validation/2-T04-unknown-mace.md``). On 0.3.16 every test here is skipped. A tiny
MH-1-class model on NaCl, the minimal planner on the CPU without compilation, as in
``test_quick.py``:

- every record whose pinned source does not hash is off with its reason;
- the always-on treatments follow 2-T04's rule (``fastmace.rewrites.versions``);
- the run completes on the path the plan reports and passes the gate;
- no source was patched that does not hash.
"""

from __future__ import annotations

import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import _patches
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.planner import minimal
from fastmace.rewrites import registry as reg
from fastmace.rewrites import versions

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.skipif(reg.installed_mace_version() in versions.QUALIFIED_MACE_VERSIONS,
                       reason="needs a MACE other than the qualified 0.3.16 (fastmace_mace_other)"),
]

NA_CL = (11, 17)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return str(path)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch):
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


@pytest.fixture
def no_engines_yet():
    from fastmace.engine import base

    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


def test_unhashed_records_are_off_with_their_reason():
    version = reg.installed_mace_version()
    registry = reg.default_registry()
    unhashed = [name for name in registry.names() if not registry.qualification(name).supported]
    assert unhashed, f"every pinned source hashes on MACE {version}; this environment tests nothing"
    for name in unhashed:
        reason = registry.qualification(name).reason
        assert f"MACE {version}" in reason and ("differs" in reason or "cannot be read" in reason)


def test_md_runs_on_the_reported_path_and_passes_the_gate(model_path, tmp_path, no_engines_yet):
    from fastmace.quick import md

    run = md(nacl(repeat=1, rattle=0.05, seed=3), model=model_path, trusted=True, temperature_K=300.0,
             steps=3, seed=5, output=tmp_path / "run", check="gate")
    assert run.gate.status == "passed"
    registry = reg.default_registry()
    for name in registry.names():
        qualified = registry.qualification(name)
        if not qualified.supported:
            field = registry.config_field(name)
            key = f"RewriteConfig.{field}"
            enabled = [other for other in registry.names()
                       if registry.config_field(other) == field and registry.qualification(other).supported]
            if not enabled:
                assert run.plan.decision(key)[1] == "off", key
    precision, promoted = versions.effective_precision("float32", polar_model=False)
    decision = run.plan.decision("force_precision")
    assert decision[1] == precision
    if promoted is not None:
        assert getattr(decision[2], "code", None) == promoted.code
    for record in _patches.applied():
        if record.name in _patches.SOURCE_PINS:
            assert _patches.patch_problems(record.name) == [], record.name
