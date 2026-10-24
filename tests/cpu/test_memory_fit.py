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
"""

import json
from pathlib import Path

import pytest

from fastmace.model.capabilities import ModelDims
from fastmace.planner import memory, minimal

DATA = Path(__file__).resolve().parents[1] / "_support" / "data" / "rule12_fit.json"
FIT = json.loads(DATA.read_text())
LOW, HIGH = 1.0, 1.5


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
