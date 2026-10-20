"""The gate's resource ladder on a 16 GB GPU (plan task 2-T05; SPEC §12.3 "Resources", acceptance 3).

On NbCl5 with 1944 atoms (the primitive cell 3x3x6, the size of 1b-T03's 3x3x3) the float64
references of POLAR-1-L and OMOL-0 XL need about 18.6 and 17.1 GB (1b-T03), more than the RTX 5060
Ti has. The gate must then end on the CPU in float64 (within the configured time limit) or on a
reduced cell (``passed-reduced``), never larger than a cell that ran out of memory. Each case runs in
a fresh child process (one backend family per process, SPEC §8.7). Only on a GPU with less memory
than the reference needs: on a larger card the first step runs and the case says so.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON")

# Acceptance tier (docs/development.md, "Test tiers"): two released checkpoints that run out of GPU
# memory, then CPU float64 or a reduced cell on 1944 atoms, minutes each. The ladder itself is tested
# on the CPU (tests/cpu/test_gate_resources.py, simulated out-of-memory at each step).
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.acceptance,
              pytest.mark.skipif(not MODELS, reason="FASTMACE_MODELS is not set")]

CHILD = textwrap.dedent("""
    import json, sys, time
    import fastmace  # before MACE, so the planner may choose the hybrid backend
    from fastmace.config import GateConfig, IntegratorConfig
    from fastmace.errors import GateFailed
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal, selfcheck
    from fastmace.planner.workload import Workload
    from _support import structures
    import torch

    SPEC = json.loads(sys.argv[1])
    loaded = loading.load_pristine(SPEC["label"])
    atoms = structures.nbcl5(repeat=1).repeat(tuple(SPEC["repeat"]))
    atoms.rattle(0.03, seed=1)
    integrator = IntegratorConfig(temperature_K=300.0)
    ff = ForceField(loaded, atoms)
    planner = minimal.default_planner(ff)
    plan = planner.plan(ff, atoms, integrator, Workload(max_temperature_K=300.0), planner.hardware())
    started = time.perf_counter()
    references = selfcheck.references(plan, ff, atoms, integrator, check="allow_unavailable", seed=1,
                                      gate=GateConfig(**SPEC.get("gate", {})))
    referenced = time.perf_counter()
    engine = planner.build(plan, ff, atoms)
    try:
        record = selfcheck.production(engine, references, check="allow_unavailable").to_metadata()
    except GateFailed as error:
        record = error.gate_record.to_metadata()
    out = {"backend": plan.engine.backend, "dtype": plan.precision.dtype, "atoms": len(atoms),
           "gpu": torch.cuda.get_device_name(), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
           "gate": record, "seconds": {"references": referenced - started,
                                       "total": time.perf_counter() - started}}
    print(json.dumps(out, default=str))
""")


def _available(label):
    from fastmace.model import labels

    return bool(MODELS) and (Path(MODELS) / labels.LABELS[label].checkpoint.file_name).is_file()


def run_gate(spec, timeout=5400):
    python = HYBRID_PYTHON or sys.executable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run([python, "-c", CHILD, json.dumps(spec)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


# 1b-T03: the float64 reference on 1944 atoms needs 18.6 GB (POLAR-1-L) and 17.1 GB (OMOL-0 XL).
CASES = {"polar1-l": 18.6e9, "omol0-xl": 17.1e9}


@pytest.mark.parametrize("label", list(CASES))
def test_out_of_memory_ends_on_the_cpu_or_a_reduced_cell(label):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not _available(label):
        pytest.skip(f"{label} is not under FASTMACE_MODELS")
    out = run_gate({"label": label, "repeat": [3, 3, 6]})
    gate = out["gate"]
    steps = gate["steps"]
    print(f"\n{label} on {out['gpu']} ({out['gpu_bytes'] / 1e9:.1f} GB), {out['atoms']} atoms: {gate['status']}, "
          f"device {gate['device']}, reduced cell {gate['reduced_cell']}, seconds {out['seconds']}")
    for index, step in enumerate(steps, 1):
        print(f"  {step['device']} {step['cell']} {step['repeat']} {step['atoms']} atoms: {step['outcome']} "
              f"({gate['seconds'][f'step_{index}_s']:.1f} s) {step['reason'] or ''}")
    print(json.dumps({"case": label, **out}, default=str))
    if out["gpu_bytes"] >= CASES[label]:
        assert steps[0]["outcome"] == "ran", "the reference fits this card: acceptance 3 needs a 16 GB card"
        return
    assert steps[0]["outcome"] == "out of memory"
    assert gate["status"] in ("passed", "passed-reduced"), gate
    if gate["status"] == "passed":
        assert gate["device"] == "cpu" and gate["reduced_cell"] is None
    else:
        assert gate["replay"]["passed"] and gate["reduced_cell"] is not None
    # Memory never grows: after an out-of-memory no later step uses a larger cell.
    smallest = None
    for step in steps:
        if step["atoms"] is not None and smallest is not None:
            assert step["atoms"] <= smallest, steps
        if step["outcome"] == "out of memory":
            smallest = step["atoms"] if smallest is None else min(smallest, step["atoms"])
