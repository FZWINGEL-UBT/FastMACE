"""tools/bench/ab_swap.py keeps a caller's own side-a constants (step-layout-cleanup, 12 November 2026): argparse keeps
only the last ``--a-constant``, so the tool's ``ab_swap:SIDE='a'`` joins the caller's list instead of replacing it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
ab_swap = pytest.importorskip("ab_swap")


@pytest.mark.parametrize("own", [[], ["--a-constant", "m:X=False", "m:Y=1"]])
def test_side_a_constants_are_one_list(monkeypatch, own):
    seen = []
    monkeypatch.setattr(ab_swap.ab_step_time, "prepare_fastmace_hybrid", lambda: None)
    monkeypatch.setattr(ab_swap, "install", lambda swaps: None)
    monkeypatch.setattr(ab_swap.ab_rewrites, "main", lambda argv: seen.append(argv))
    ab_swap.main(["--swap", "a.b:c=d", "--checkpoint", "x", *own])
    argv = seen[0]
    assert argv.count("--a-constant") == 1
    start = argv.index("--a-constant") + 1
    constants = argv[start:start + 1 + len(own[2:]) if own else start + 1]
    assert "ab_swap:SIDE='a'" in constants and all(c in argv for c in own[1:])
