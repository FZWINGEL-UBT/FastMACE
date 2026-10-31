"""Rule 12's fit on the measured GPU peaks, as a regression test (plan task 5-T03; SPEC §9.3 rule 12, decision 51).

``tests/_support/data/rule12_fit.json`` holds 60 measured cases from docs/validation/5-T03.md:
- the 14 labels that run on Na, Nb and Cl (the 5 MACE-OFF models have no Na embedding), on NaCl and NbCl5
  3x3x3, NVT and NPT, graph tier, hybrid float32, on the RTX 5060 Ti and the A100;
- MH-1 and MP-0 medium on the plain cuEq backend.

For each it keeps the inputs of the planner's final ``memory.estimate`` (re-planned on the CPU with the
recorded GPU and SM count):
- the model's ``ModelDims``, edges, atoms and capacity;
- the backend path;
- the names of the rewrites whose terms apply;

and the measured peaks. The test rebuilds each estimate with today's code and terms, so a change to the
formula, a rewrite's declared term, ``SAFETY`` or the workspaces that moves any case outside [1.0, 1.5] of
its measured peak fails here.

``tests/_support/data/rule12_capacity.json`` holds the capacity scan's MH-1 cells (decision 67, plans/capacity.md
CAP-T02; docs/validation/capacity-T02.md): NaCl, NbCl5 and SrTiO3 from 2,000 to 30,600 atoms, float32 and float64,
the default plan with the fused path on, measured on the RTX 5060 Ti (WSL2) and the RTX 4000 Ada (Linux). Their
estimate must stay between the measured peak and ``CAPACITY_HIGH`` times it, so that rule 12 predicts the
limit instead of acting at half of it.
"""

import json
from pathlib import Path

import pytest

from fastmace.model.capabilities import ModelDims
from fastmace.planner import memory, minimal

DATA = Path(__file__).resolve().parents[1] / "_support" / "data" / "rule12_fit.json"
FIT = json.loads(DATA.read_text())
LOW, HIGH = 1.0, 1.5
CAPACITY = json.loads((DATA.parent / "rule12_capacity.json").read_text())
CAPACITY_HIGH = 1.4
RESCUE = json.loads((DATA.parent / "rule12_rescue.json").read_text())


def predicted(case):
    registry = minimal._rewrite_registry()
    dims = ModelDims.from_record(FIT["dims"][case["label"]])
    dtype = case["dtype"]
    return memory.estimate(
        dims, edges=case["edges"], nodes=case["nodes"], dtype=dtype, capacity=case["capacity"], list_dtype=dtype,
        fused_conv=case["fused_conv"],
        rewrite_terms=memory.rewrite_memory_terms(registry, dims, case["rewrites"], dtype),
        selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, case["selfchecks"], dtype, case["sms"]),
        resident_terms=memory.resident_memory_terms(registry, dims, case["resident"], dtype))


def test_the_fit_covers_both_backends_and_gpus():
    cases = FIT["cases"]
    assert len(cases) == 60
    assert {case["backend"] for case in cases} == {"hybrid", "cueq"}
    assert {case["sms"] for case in cases} == {36, 108}  # the RTX 5060 Ti and the A100


@pytest.mark.rule(12)
@pytest.mark.parametrize("case", FIT["cases"],
                         ids=[f"{c['label']}-{c['structure']}-{c['ensemble']}-{c['backend']}" for c in FIT["cases"]])
def test_the_prediction_stays_within_the_fit(case):
    """Never below the measured peak (rule 12 guards memory), at most 1.5x above it."""
    found = predicted(case)
    ratio = found.total / case["peak_bytes"]
    assert LOW <= ratio <= HIGH, (ratio, found.text(), case["peak_bytes"])


def test_polar_streaming_backward_term():
    """5-T03 rerun (24 October 2026): PolarMACE on the streaming convolution keeps one more [E, W_max] buffer
    (the convolution backward's weight gradient) live at the steps' peak. The term is W_max floats per edge."""
    dims = ModelDims.from_record(FIT["dims"]["polar1-l"])
    name, term = memory.polar_streaming_backward_term(dims, "float32")
    w_max = max(block.weight_numel for block in dims.interactions)
    assert name == memory.POLAR_STREAMING_BACKWARD and term.edge_bytes == 4 * w_max and term.node_bytes == 0
    assert term.edge_sized


@pytest.mark.rule(12)
@pytest.mark.parametrize("case", CAPACITY, ids=[f"{c['structure']}-{c['natoms']}-{c['dtype']}-{c['platform']}"
                                                 for c in CAPACITY])
def test_large_mh1_cells_stay_within_the_capacity_fit(case):
    """Decision 67: never below the measured peak, at most CAPACITY_HIGH above it."""
    found = predicted(case)
    ratio = found.total / case["peak_bytes"]
    assert LOW <= ratio <= CAPACITY_HIGH, (ratio, found.text(), case["peak_bytes"])


@pytest.mark.rule(12)
@pytest.mark.parametrize("case", RESCUE, ids=[f"{c['structure']}-{c['natoms']}-{c['stage']}" for c in RESCUE])
def test_every_stage_of_the_memory_rescue_stays_within_the_capacity_fit(case):
    """docs/validation/estimate-refit.md: with the memory-rescue records on, the estimate stays between the measured
    peak and CAPACITY_HIGH times it at every stage of the ladder, as the default plan's (counted as absolute bytes,
    the default plan's overstatement grew to 2.5x the peak with every record on)."""
    found = predicted(case)
    ratio = found.total / case["peak_bytes"]
    assert LOW <= ratio <= CAPACITY_HIGH, (ratio, found.text(), case["peak_bytes"])


def test_the_rescue_fit_covers_the_whole_ladder():
    stages = {case["stage"] for case in RESCUE}
    assert stages == {"s0-default", "s1-conv", "s2-table", "s3-segments", "s4-geoff", "s5-region", "s6-region-ge"}
    for structure in ("NaCl", "NbCl5"):
        sizes = {case["natoms"] for case in RESCUE if case["structure"] == structure and case["stage"] == "s5-region"}
        assert len(sizes) >= 3 and max(sizes) >= 85_000, structure


def test_the_capacity_fit_covers_both_platforms_and_precisions():
    assert {case["dtype"] for case in CAPACITY} == {"float32", "float64"}
    assert {case["structure"] for case in CAPACITY} == {"NaCl", "NbCl5", "SrTiO3"}
    assert max(case["natoms"] for case in CAPACITY) >= 25_000
