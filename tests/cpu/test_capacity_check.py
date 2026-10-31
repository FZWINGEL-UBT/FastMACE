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


def test_the_initial_list_must_have_exactly_the_reference_edges(capacity_scan):
    valid, detail = capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.5, reference=32.0)
    assert valid and detail["initial_ok"] and detail["expected_initial_edges"] == 32_000
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_001, 0.5, reference=32.0)[0]  # one pair off
    assert not capacity_scan.check_point("NaCl", 6.0, 1_728_000, 0, 0.5, reference=26.3)[0]  # no edges
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, None, 0.5, reference=32.0)[0]
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.0, reference=32.0)[0]  # zero forces
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, None, reference=32.0)[0]


def test_after_md_the_live_list_must_equal_a_fresh_rebuild_exactly(capacity_scan):
    # NbCl5's count grows 5 % in 200 steps at 300 K: thermal drift is fine, a mismatch with a fresh list is not.
    ok = capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.5, live_edges=33_700, fresh=33_700, reference=32.0)
    assert ok[0] and ok[1]["live_ok"]
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.5, live_edges=33_700, fresh=33_702,
                                         reference=32.0)[0]
    assert not capacity_scan.check_point("NaCl", 6.0, 1000, 32_000, 0.5, live_edges=33_700, reference=32.0)[0]


def test_fresh_edges_counts_the_pairs_at_the_given_positions(capacity_scan):
    import numpy as np
    from ase.neighborlist import neighbor_list

    cell = rocksalt(None)[1].repeat(3)
    moved = cell.positions + np.random.default_rng(1).normal(0.0, 0.1, cell.positions.shape)
    shifted = cell.copy()
    shifted.positions = moved
    assert capacity_scan.fresh_edges(cell, moved, 6.0) == len(neighbor_list("i", shifted, 6.0))
