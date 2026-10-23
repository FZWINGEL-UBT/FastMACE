"""The gate on the user's structure on the GPU (plan task 1b-T16; SPEC §12.3).

Each case runs in a fresh child process (the plan fixes the process's backend family, SPEC §8.7),
with ``FASTMACE_HYBRID_PYTHON`` when it is set (the hybrid backend where OEq builds), else this
environment's Python. The child loads the label with ``fm.load``, plans with the minimal planner,
runs the gate's references, builds the engine, and runs the gate's production half in graph mode
(captured replay):

- MH-1, MP-0, POLAR-1-S and OMOL-0 XL on NaCl 3x3x3 pass (MH-1 in NVT and full-cell NPT);
- a broken captured graph (a stand-in for a wrong kernel that only the captured stage runs, like
  the PDL failure: forces scaled by 1.001 inside the capture) fails the gate on forces, while the
  same comparison through the eager stages passes;
- references computed in-process (before any production engine) equal references computed in a
  fresh child process after the engine was built: float64 within 1e-12, the float32 yardstick within
  the model's process-to-process float32 scatter (OMOL-0 XL 2e-4 eV/Å, 1a-T19; others 1e-5);
- the gate's time on NaCl 3x3x3 and NbCl5 2x2x2 (MH-1) is recorded ("seconds to a minute");
- 4-T12 (SPEC decision 49): POLAR-1-S passes on NaCl 3x3x3 and NbCl5 3x3x3, fixed cell and NPT, with
  the PolarMACE record's stress floor of 5e-7 eV/Å^3 in the gate record; a deliberate stress error of
  6e-7 eV/Å^3 in the production path still fails, on stress alone.
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

pytestmark = pytest.mark.skipif(not MODELS, reason="FASTMACE_MODELS is not set")


def _available(label):
    if not MODELS:
        return False
    from fastmace.model import labels

    return (Path(MODELS) / labels.LABELS[label].checkpoint.file_name).is_file()


def run_gate(spec, timeout=3600):
    """The child below with ``spec``; returns the JSON of its last stdout line."""
    python = HYBRID_PYTHON or sys.executable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    result = subprocess.run([python, "-c", CHILD, json.dumps(spec)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


CHILD = textwrap.dedent("""
    import json, sys, time
    import fastmace  # before MACE, so the planner may choose the hybrid backend
    from fastmace.config import IntegratorConfig
    from fastmace.errors import GateFailed
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal, selfcheck
    from fastmace.planner.workload import Workload
    from _support import structures

    SPEC = json.loads(sys.argv[1])
    loaded = loading.load_pristine(SPEC["label"])
    atoms = getattr(structures, SPEC["structure"])(repeat=SPEC["repeat"])
    integrator = IntegratorConfig(temperature_K=300.0, **SPEC.get("integrator", {}))
    ff = ForceField(loaded, atoms)
    planner = minimal.default_planner(ff)
    hardware = planner.hardware()
    plan = planner.plan(ff, atoms, integrator, Workload(max_temperature_K=300.0), hardware)
    import torch

    started = time.perf_counter()
    references = selfcheck.references(plan, ff, atoms, integrator, check="gate", seed=1)
    engine = planner.build(plan, ff, atoms)
    built = time.perf_counter()

    def gate():
        try:
            return selfcheck.production(engine, references, check="gate").to_metadata()
        except GateFailed as error:
            return error.gate_record.to_metadata()

    out = {"backend": plan.engine.backend, "dtype": plan.precision.dtype, "capture": engine.capture,
           "atoms": len(atoms), "gpu": torch.cuda.get_device_name()}
    if SPEC.get("broken_capture"):
        engine.capture = False
        out["eager"] = gate()
        engine.capture = True
        stage = engine._stage

        def broken(key):
            run = stage(key)
            if not (isinstance(key, tuple) and key[0] == "B"):
                return run

            def wrong():  # a wrong kernel that only the captured stage runs
                run()
                if torch.cuda.is_current_stream_capturing():
                    engine.state.forces.mul_(1.001)

            return wrong

        engine._stage = broken
        engine._graphs.clear()
    gate_started = time.perf_counter()
    out["gate"] = gate()
    if SPEC.get("stress_offset"):
        read = selfcheck._read_outputs

        def shifted(engine_, stress):  # a deliberate stress error in the production path, nothing else
            energy, forces, values = read(engine_, stress)
            values = values.copy()
            values[0, 0] += SPEC["stress_offset"]
            return energy, forces, values

        selfcheck._read_outputs = shifted
        out["offset_gate"] = gate()
        selfcheck._read_outputs = read
    out["seconds"] = {"references": references.seconds, "build": built - started - references.seconds,
                      "production": time.perf_counter() - gate_started}
    if SPEC.get("fresh_references"):
        again = selfcheck.references(plan, ff, atoms, integrator, check="gate", seed=1)
        assert again.ran == "child process", again.ran
        diff = {}
        for role in ("reference", "yardstick"):
            values = [(a[role], b[role]) for a, b in zip(references.values, again.values)]
            diff[role] = {
                "energy_eV": max(abs(x[0] - y[0]) for x, y in values),
                "forces_eV_per_A": max(float(abs(x[1] - y[1]).max()) for x, y in values),
                "stress_eV_per_A3": max(float(abs(x[2] - y[2]).max()) for x, y in values)}
        out["fresh_references"] = diff
    print(json.dumps(out))
""")

NACL = {"structure": "nacl", "repeat": 3}
CASES = {
    "mh1-r2scan-nvt": dict(label="mh1-r2scan", **NACL, fresh_references=True),
    "mh1-r2scan-npt-full": dict(label="mh1-r2scan", **NACL, integrator={"ensemble": "npt", "cell_mode": "full"}),
    "mp0-medium": dict(label="mp0-medium", **NACL),
    "polar1-s": dict(label="polar1-s", **NACL, fresh_references=True),
    "omol0-xl": dict(label="omol0-xl", **NACL, fresh_references=True),
}
SCATTER = {"omol0-xl": 2e-4}  # float32 process-to-process scatter (1a-T19); 1e-5 eV/Å otherwise


def _report(name, out):
    gate = out["gate"]
    print(f"\n{name}: {out['backend']} {out['dtype']} on {out['gpu']}, {out['atoms']} atoms: {gate['status']} "
          f"({gate['copies']}; energy {gate['energy_meV_per_atom']} meV/atom, force {gate['force_eV_per_A']} eV/Å, "
          f"stress {gate['stress_eV_per_A3']} eV/Å^3; {gate['mode']}); seconds {out['seconds']}")
    print(json.dumps({"case": name, **out}))


# OMOL-0 XL (147 s) checks a further model at acceptance (1b-T27); the gate itself stays in the full tier.
@pytest.mark.slow
@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.acceptance if name == "omol0-xl" else ())
                                  for name in CASES])
def test_gate_passes_in_graph_mode(name):
    spec = CASES[name]
    if not _available(spec["label"]):
        pytest.skip(f"{spec['label']} is not in FASTMACE_MODELS")
    out = run_gate(spec)
    _report(name, out)
    gate = out["gate"]
    assert out["capture"] and gate["mode"] == "captured replay of stages C, B"
    assert gate["status"] == "passed", gate
    assert gate["references"]["ran"] == "child process"  # a CUDA plan's GPU references: always a child (2-T05)
    if "npt" in name:
        assert gate["stress_eV_per_A3"] is not None
    else:
        assert gate["quantities"]["stress_eV_per_A3"]["note"] == "stress: not in plan"
    fresh = out.get("fresh_references")
    if fresh:
        assert fresh["reference"]["forces_eV_per_A"] <= 1e-12 and fresh["reference"]["stress_eV_per_A3"] <= 1e-12
        assert fresh["reference"]["energy_eV"] <= 1e-12 * out["atoms"]
        assert fresh["yardstick"]["forces_eV_per_A"] <= SCATTER.get(spec["label"], 1e-5)


NPT = {"ensemble": "npt"}
POLAR_CASES = {
    "polar1-s-nacl-npt": dict(label="polar1-s", **NACL, integrator=NPT, stress_offset=6e-7),
    "polar1-s-nacl-npt-full": dict(label="polar1-s", **NACL, integrator={**NPT, "cell_mode": "full"}),
    "polar1-s-nbcl5-nvt": dict(label="polar1-s", structure="nbcl5", repeat=3),
    "polar1-s-nbcl5-npt": dict(label="polar1-s", structure="nbcl5", repeat=3, integrator=NPT, stress_offset=6e-7),
}
POLAR_STRESS_FLOOR = 5e-7  # the PolarMACE record's entry (SPEC decision 49)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(POLAR_CASES))
def test_polar_gate_uses_the_record_stress_floor(name):
    from fastmace.reference import records

    spec = POLAR_CASES[name]
    if not _available(spec["label"]):
        pytest.skip(f"{spec['label']} is not in FASTMACE_MODELS")
    out = run_gate(spec)
    _report(name, out)
    gate = out["gate"]
    assert gate["status"] == "passed", gate
    assert gate["references"]["record_id"] == records.record_for("0.3.16", "polar").id
    assert gate["floors"]["stress_eV_per_A3"] == POLAR_STRESS_FLOOR
    assert gate["floors"]["force_eV_per_A"] == 1e-5 and gate["floors"]["energy_meV_per_atom"] == 0.01
    if "npt" not in name:
        assert gate["quantities"]["stress_eV_per_A3"]["note"] == "stress: not in plan"
        return
    stress = gate["quantities"]["stress_eV_per_A3"]
    assert stress["floor"] == POLAR_STRESS_FLOOR and stress["floor_source"] == "reference record"
    if spec.get("stress_offset"):
        offset = out["offset_gate"]
        assert offset["status"] == "failed", offset
        failed = [quantity for quantity, q in offset["quantities"].items() if not q["passed"]]
        assert failed == ["stress_eV_per_A3"], offset
        assert offset["stress_eV_per_A3"] > POLAR_STRESS_FLOOR


@pytest.mark.slow
@pytest.mark.skipif(not _available("mh1-r2scan"), reason="needs MH-1 in FASTMACE_MODELS")
def test_a_broken_captured_graph_fails_while_eager_passes():
    out = run_gate(dict(label="mh1-r2scan", **NACL, broken_capture=True))
    _report("mh1-broken-capture", out)
    assert out["eager"]["status"] == "passed" and out["eager"]["mode"] == "eager stages C, B"
    gate = out["gate"]
    assert gate["status"] == "failed" and gate["mode"] == "captured replay of stages C, B"
    assert not gate["quantities"]["force_eV_per_A"]["passed"]
    assert gate["quantities"]["energy_meV_per_atom"]["passed"]


@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.skipif(not _available("mh1-r2scan"), reason="needs MH-1 in FASTMACE_MODELS")
def test_gate_time_on_nbcl5():
    out = run_gate(dict(label="mh1-r2scan", structure="nbcl5", repeat=2))
    _report("mh1-r2scan-nbcl5", out)
    assert out["gate"]["status"] == "passed", out["gate"]
