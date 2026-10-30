"""The capacity tools' correctness check (tools/bench/capacity_scan.py ``check_point``).

A capacity point counts only if its list has the structure's edges per atom and its forces are not all zero: on
31 October 2026 two points "ran" at 1.73M and 2.0M atoms with an empty list (the GPU pair screen's launch was
skipped), and a scan counted them as passes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "bench"


@pytest.fixture(scope="module")
def capacity_scan():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("_capacity_scan_check", TOOLS / "capacity_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rocksalt(structure):
    from ase.build import bulk

    return "NaCl", bulk("NaCl", "rocksalt", a=5.64, cubic=True)


def test_reference_edges_per_atom_is_the_small_cells(capacity_scan, monkeypatch):
    from ase.neighborlist import neighbor_list

    monkeypatch.setattr(capacity_scan, "base_cell", rocksalt)
    reference = capacity_scan.reference_edges_per_atom("NaCl", 6.0)
    big = rocksalt(None)[1].repeat(4)
    assert reference == pytest.approx(len(neighbor_list("i", big, 6.0)) / len(big))


def test_check_point_refuses_an_empty_list_and_zero_forces(capacity_scan):
    valid, detail = capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.5, reference=32.0)
    assert valid and detail["deviation"] == 0
    assert capacity_scan.check_point("NaCl", 6.0, 1000, 33_000, 0.5, reference=32.0)[0]  # 3 % thermal
    assert not capacity_scan.check_point("NaCl", 6.0, 1_728_000, 0, 0.5, reference=26.3)[0]  # no edges
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.0, reference=32.0)[0]  # zero forces
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, None, reference=32.0)[0]
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 40_000, 0.5, reference=32.0)[0]  # 25 % off
