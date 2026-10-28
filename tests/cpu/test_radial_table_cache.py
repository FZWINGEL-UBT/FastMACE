"""The radial-table cache: fitted and certified ``radial_tables`` reused across builds (28 October 2026).

A float64 MH-1 build spends about half its time fitting and certifying the per-species-pair radial tables
(ResearchModule Experiments/Energy_Hull, profile on trillian: 42 of 87 s, ``certify`` 30.5 s). The tables depend only
on the model's state, the species set, the dtype, r_max and the fit's constants and code, so
``tabulate_radials`` stores them under ``FASTMACE_CACHE/radial-tables`` and reuses them when all of those match.
The self-check still runs on cached tables.
"""

import copy

import pytest
import torch

from fastmace import _cache
from fastmace.rewrites import mh1, radial
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, tiny  # noqa: E402


def _registry():
    out = reg.Registry()
    for family in (mh1, radial):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def _tabled(model, batch):
    regs = _registry()
    asked = {name: "on" for name in PACKAGE} | {"radial_tables": "on"}
    asked |= {name: "off" for name in regs.names() if name not in asked}
    out = copy.deepcopy(model)
    applied = regs.apply(out, regs.resolve(out, asked, strict=False), context(batch), original=model)
    return out, next(entry for entry in applied if entry.name == "radial_tables")


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv(_cache.ENVIRONMENT_VARIABLE, str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def calls(monkeypatch):
    counts = {"fit": 0, "certify": 0}
    fit, certify = radial.fit, radial.certify

    def counting_fit(*args, **kwargs):
        counts["fit"] += 1
        return fit(*args, **kwargs)

    def counting_certify(*args, **kwargs):
        counts["certify"] += 1
        return certify(*args, **kwargs)

    monkeypatch.setattr(radial, "fit", counting_fit)
    monkeypatch.setattr(radial, "certify", counting_certify)
    return counts


def _state(model):
    return [(table.coef, table.dcoef, table.starts, table.scales, table.bins, table.r_lo, table.bin_scale, table.r_max,
             table.fit_error, table.certificate) for table in (i.radial_table for i in model.interactions)]


def _same(first, second):
    for a, b in zip(first, second, strict=True):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y) if torch.is_tensor(x) else x == y


def test_a_second_build_reuses_the_tables_bitwise(cache_root, calls):
    model, batch = tiny("mh1")
    first, entry = _tabled(model, batch)
    interactions = len(model.interactions)
    assert calls == {"fit": interactions, "certify": interactions} and entry.result.passed
    assert len(list((cache_root / radial.CACHE_KIND).glob("*.pt"))) == interactions
    second, entry = _tabled(model, batch)
    assert calls == {"fit": interactions, "certify": interactions}  # nothing fitted or certified again
    assert entry.result.passed  # the self-check ran on the cached tables
    _same(_state(first), _state(second))
    _, forces = energy_and_forces(first, batch)
    _, cached_forces = energy_and_forces(second, batch)
    # the tables are bitwise equal (above); the CPU forces of one model already differ by ~1 ulp between two calls
    torch.testing.assert_close(cached_forces, forces, rtol=1e-14, atol=1e-17)


def test_other_weights_miss_the_cache(cache_root, calls):
    model, batch = tiny("mh1")
    _tabled(model, batch)
    changed = copy.deepcopy(model)
    with torch.no_grad():
        next(changed.parameters()).add_(1e-6)
    _tabled(changed, batch)
    assert calls["fit"] == 2 * len(model.interactions)


def test_the_key_holds_the_species_dtype_constants_and_code(cache_root):
    model, _ = tiny("mh1")
    key = radial.cache_key(model, 0, [0, 1], torch.float64, 5.0)
    assert key == radial.cache_key(copy.deepcopy(model), 0, [0, 1], torch.float64, 5.0)
    assert key != radial.cache_key(model, 1, [0, 1], torch.float64, 5.0)
    assert key != radial.cache_key(model, 0, [1, 0], torch.float64, 5.0)
    assert key != radial.cache_key(model, 0, [0, 1], torch.float32, 5.0)
    assert key != radial.cache_key(model, 0, [0, 1], torch.float64, 5.5)


def test_a_damaged_entry_is_refitted_and_replaced(cache_root, calls):
    model, batch = tiny("mh1")
    first, _ = _tabled(model, batch)
    for path in (cache_root / radial.CACHE_KIND).glob("*.pt"):
        path.write_bytes(b"not a torch file")
    second, entry = _tabled(model, batch)
    assert calls["fit"] == 2 * len(model.interactions) and entry.result.passed
    _same(_state(first), _state(second))
    third, _ = _tabled(model, batch)
    assert calls["fit"] == 2 * len(model.interactions)  # the replaced entries hit again
