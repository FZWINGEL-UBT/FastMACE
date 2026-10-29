"""The run report (plan task 1b-T15; SPEC §7.7 "The report", §12.5 setup stages).

- Golden text for a constant-T NVT run: the lines of the §7.7 example, two stages (``speed:
  pending``, then one measured line, then the whole-run mean).
- The NPT "experimental: not yet qualified" line only for NPT; PolarMACE NPT has its own line.
- The licence line comes from the label table for every label, never by hand.
- Every refusal and downgrade code that the minimal planner publishes (``docs/design/plan-1b.md``)
  has a report reason; the report matches codes, never text.
- The speed line never carries a prediction, and the whole-run mean replaces it at the end.
- A ``reproducible=True`` request shows §9.3's refusal.
- ``to_metadata()`` is JSON-safe and goes through T10's ``metadata_record`` under ``"report"``.
"""

import json
import re
from pathlib import Path

import pytest

from fastmace import provenance, report
from fastmace.config import EngineConfig, IntegratorConfig, PolarConfig, PrecisionConfig
from fastmace.errors import PlanRefused, Reason
from fastmace.model import labels
from fastmace.planner.fixed import REWRITE_FIELDS
from fastmace.planner.plan import Plan
from fastmace.planner.workload import Workload

ROOT = Path(__file__).resolve().parents[2]
MH1 = labels.CHECKPOINTS["mh-1"]
SETUP = {"load": 6.0, "rewrites": 9.0, "compilation": 17.0, "capture": 4.0, "gate": 6.0}
OUTPUTS = ("run/trajectory.000.traj", "run/thermo.csv", "run/metadata.json")
GATE_PASSED = {"status": "passed", "copies": "2 rattled", "energy_meV_per_atom": 0.04, "force_eV_per_A": 2e-6}


def _rewrites(off=("streaming_conv", "single_graph_strain", "grouped_launch", "folded_backward_tiles",
                   "direct_node_feats", "folded_constants", "radial_tables", "combo_kernels", "save_transcendentals",
                   "tf32x3_gemms",
                   "fused_folded_forward", "w_tables", "wprime_backward",
                   "fx_e3nn_linear", "polar_graph_sums")):
    return tuple((f"RewriteConfig.{name}", "off" if name in off else "on", "auto") for name in REWRITE_FIELDS)


def make_plan(ensemble="nvt", backend="hybrid", backend_code="backend-hybrid", polar=False, workload=None,
              extra=()):
    decisions = (("mode", "graph", Reason("graph tier", code="mode-graph")),
                 ("EngineConfig.backend", backend, Reason("auto", code=backend_code)),
                 ("force_precision", "float32", "the ForceField's precision"),
                 *_rewrites(), *extra)
    return Plan(tier="graph", engine=EngineConfig(backend=backend), precision=PrecisionConfig(dtype="float32"),
                polar=PolarConfig() if polar else None,
                integrator=IntegratorConfig(ensemble=ensemble, timestep_fs=0.5), decisions=decisions,
                workload=workload or Workload(max_temperature_K=300.0))


def make_report(plan="default", **changes):
    arguments = dict(model="mh1", head="matpes_r2scan", sha256=MH1.sha256, structure="NaCl", n_atoms=216,
                     repeat=(3, 3, 3), plan=make_plan() if plan == "default" else plan, gate=GATE_PASSED,
                     setup_s=SETUP, outputs=OUTPUTS)
    arguments.update(changes)
    return report.Report(**arguments)


GOLDEN_SETUP = """\
FastMACE {version}  |  mh1 (head matpes_r2scan)  |  NaCl, 216 atoms (3×3×3), 3D PBC, NVT Langevin 0.5 fs
physics: no dispersion; charge/spin not used by this model
plan:    graph, hybrid cuEq/OEq, float32, 18 rewrites on, 15 off (reasons in run.plan)
gate:    passed (2 rattled; 0.04 meV/atom, 2e-06 eV/Å; floors §12.3)
setup:   42 s (load 6 s, rewrites 9 s, compilation 17 s, capture 4 s, gate 6 s)
speed:   pending (measured over steps 1,000–2,000 of this run)
output:  run/trajectory.000.traj, run/thermo.csv, run/metadata.json
licence: ASL (non-commercial), from the label table (§4.3)
warn:    none"""


def test_golden_constant_t_nvt():
    from fastmace import __version__
    rep = make_report()
    assert rep.setup_text() == GOLDEN_SETUP.format(version=__version__)
    line = rep.record_speed(0.0041, 1000, 2000)
    assert line == "speed:   4.1 ms/step, measured over steps 1,000–2,000 of this run"
    assert line in rep.text()
    assert "pending" not in rep.text()
    rep.finish(0.0042, steps=1_000_000)
    assert "speed:   4.2 ms/step, whole-run mean over steps 1–1,000,000" in rep.text()
    assert "measured over" not in rep.text()


def test_speed_window_from_arguments():
    assert make_report().window == (1000, 2000)
    rep = make_report(warmup_steps=500, window_steps=200, start_step=10_000)
    assert rep.window == (10_500, 10_700)
    assert "measured over steps 10,500–10,700" in rep.setup_text()
    with pytest.raises(ValueError):
        make_report(window_steps=0)


def test_speed_line_never_predicts():
    rep = make_report()
    speed = [line for line in rep.setup_text().splitlines() if line.startswith("speed:")]
    assert speed == ["speed:   pending (measured over steps 1,000–2,000 of this run)"]
    assert not re.search(r"\d+(\.\d+)?\s*(ms|s)/step|×|expected|speed-up", speed[0])
    with pytest.raises(ValueError, match="positive"):
        rep.record_speed(-1.0, 1000, 2000)
    with pytest.raises(ValueError, match="positive"):
        rep.finish(float("nan"), steps=10)


def test_short_run_gets_the_whole_run_mean_without_a_window():
    rep = make_report()
    rep.finish(0.005, steps=300)
    assert "speed:   5 ms/step, whole-run mean over steps 1–300" in rep.text()
    assert rep.to_metadata()["speed"] == {"status": "whole-run mean", "seconds_per_step": 0.005,
                                          "first_step": 1, "last_step": 300}


def test_resumed_run_mean_counts_from_the_start_step():
    rep = make_report(start_step=5000)
    rep.finish(0.004, steps=2000)
    assert "whole-run mean over steps 5,001–7,000" in rep.text()


# --------------------------------------------------------------------------- conditional lines


def test_npt_line_only_for_npt():
    nvt = make_report().setup_text()
    npt = make_report(make_plan("npt")).setup_text()
    assert "experimental" not in nvt
    assert "NPT barostat: experimental: not yet qualified" in npt
    assert "NPT LangevinBAOAB 0.5 fs" in npt.splitlines()[0]


def test_polar_npt_line_only_for_polar_npt():
    assert "PolarMACE NPT" not in make_report(make_plan("nvt", polar=True)).setup_text()
    assert "PolarMACE NPT" not in make_report(make_plan("npt")).setup_text()
    text = make_report(make_plan("npt", polar=True), model="polar1-s").setup_text()
    assert "PolarMACE NPT stress: experimental" in text


def test_d3_and_mc_lines_only_when_used():
    plain = make_report().setup_text()
    assert "D3" not in plain and "MC" not in plain
    text = make_report(dispersion={"functional": "pbe", "damping": "bj"}).setup_text()
    assert "physics: D3(BJ) functional pbe" in text
    assert "D3: experimental: not yet qualified" in text


def test_charge_spin_shown_when_used():
    text = make_report(model="omol0-xl", head=None, sha256=labels.CHECKPOINTS["omol-0-xl"].sha256,
                       charge_spin=(0, 1)).setup_text()
    assert "physics: no dispersion; total charge 0, spin multiplicity 1" in text


def test_extra_warnings_are_listed():
    rep = make_report(warnings=("observer writes go through Simulation.set_positions (D4): correct and slow",))
    rep.add_warning("second warning")
    lines = rep.setup_text().splitlines()
    assert "warn:    observer writes go through Simulation.set_positions (D4): correct and slow" in lines
    assert "         second warning" in lines


def test_model_known_issues_only_for_model_and_structure():
    assert "Fukui" not in make_report().setup_text()
    assert "Fukui" in make_report(model="polar1-s", species=("Na", "Nb", "O")).setup_text()
    assert "Fukui" not in make_report(model="polar1-m", species=("Na", "Nb", "O")).setup_text()
    assert "NaTaO3" in make_report(model="polar1-l", species=("Na", "Ta", "O")).setup_text()
    assert "training domain" in make_report(model="omol0-xl", species=("Na", "Cl")).setup_text()
    assert "training domain" not in make_report(model="omol0-xl", species=("C", "H", "O")).setup_text()
    volume = make_report(make_plan("npt"), model="off23-medium", species=("C", "H", "N", "O")).setup_text()
    assert "below the experimental volume" in volume
    assert "below the experimental volume" not in make_report(model="off23-medium",
                                                              species=("C", "H", "N", "O")).setup_text()


def test_small_cell_warning():
    assert "periodic image" not in make_report(cell_heights_A=(16.9,) * 3, cutoff_A=6.0).setup_text()
    assert "periodic image" in make_report(cell_heights_A=(5.6,) * 3, cutoff_A=6.0).setup_text()


def test_gate_lines():
    assert 'gate:    not run (opt-in: check="gate")' in make_report(gate=None).setup_text()
    assert 'gate:    not run (opt-in: check="gate")' in make_report(gate=provenance.GATE_SKIPPED).setup_text()
    text = make_report(gate={"status": "unavailable", "reason": "reference out of memory"}).setup_text()
    assert 'gate:    unavailable (reference out of memory); run allowed by check="allow_unavailable"' in text
    stress = dict(GATE_PASSED, stress_eV_per_A3=3e-8)
    assert "3e-08 eV/Å³" in make_report(gate=stress).setup_text()


@pytest.mark.parametrize("label", sorted(labels.LABELS))
def test_licence_line_from_label_table(label):
    entry = labels.LABELS[label]
    expected = {labels.ASL: "ASL (non-commercial)", labels.MIT: "MIT"}[entry.licence]
    by_hash = make_report(model=label, head=entry.head, sha256=entry.checkpoint.sha256).setup_text()
    by_label = make_report(model=label, head=entry.head, sha256=None).setup_text()
    for text in (by_hash, by_label):
        assert f"licence: {expected}, from the label table (§4.3)" in text.splitlines()


def test_licence_line_for_a_model_outside_the_table():
    text = make_report(model="my.model", head=None, sha256="ab" * 32).setup_text()
    assert "licence: not a label of the FastMACE table (SHA-256 abababababab…); check the model's release" in text


# --------------------------------------------------------------------------- not on the graph engine


def _codes_of_table(heading, column):
    text = (ROOT / "docs" / "design" / "plan-1b.md").read_text(encoding="utf-8")
    block = text.split(heading, 1)[1].split("\n\n", 2)[1]
    return [code for row in block.splitlines()[2:]
            for code in re.findall(r"`([a-z0-9-]+)`", row.split("|")[column])]


REFUSAL_CODES = _codes_of_table("**Refusals**", 1)
BACKEND_CODES = [code for code in _codes_of_table("**Decision codes**", 2) if code.startswith("backend-cueq")]


def test_published_tables_were_read():
    assert "reproducible-family-not-qualified" in REFUSAL_CODES and len(REFUSAL_CODES) >= 25
    assert set(BACKEND_CODES) >= {"backend-cueq-oeq-unavailable", "backend-cueq-mace-imported-first"}


@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_every_refusal_code_has_a_reason(code):
    error = PlanRefused.from_reasons([Reason("planner text, not read by the report", rule=None, code=code)])
    text = make_report(plan=None, refusal=error, integrator=IntegratorConfig()).setup_text()
    lines = text.splitlines()
    assert "plan:    refused (reasons below; run.plan has none)" in lines
    assert "planner text" not in text
    assert any(line.startswith("refused:") and report.REFUSALS[code] in line for line in lines)


COMPILED_CODES = ["mode-compiled-tier", "mode-compiled-rule-4"]  # 3-T06 (rule 9: decision 38, 3-T06b)


@pytest.mark.parametrize("code", BACKEND_CODES + ["host-callbacks-graph-sync", "allow-naive-cueq",
                                                  "reproducible-qualified"] + COMPILED_CODES)
def test_every_downgrade_code_has_a_reason(code):
    if code.startswith("backend-cueq"):
        plan = make_plan(backend="cueq", backend_code=code)
    elif code in COMPILED_CODES:
        plan = make_plan(extra=(("mode", "compiled", Reason("x", code=code)),))
    elif code == "host-callbacks-graph-sync":
        plan = make_plan(extra=(("workload.host_callbacks_every", "graph, host sync every 5 steps",
                                 Reason("x", code=code)),))
    else:
        plan = make_plan(extra=(("allow_naive_cueq", "on", Reason("x", code=code)),))
    lines = make_report(plan).setup_text().splitlines()
    assert any(line.startswith("why:") and report.DOWNGRADES[code] in line for line in lines), lines
    if code in COMPILED_CODES:
        assert "compiled tier: no CUDA graphs" in report.DOWNGRADES[code]


def test_hybrid_plan_has_no_downgrade_line():
    assert not any(line.startswith("why:") for line in make_report().setup_text().splitlines())


def test_naive_cueq_line_only_when_the_kernels_are_naive():
    """1b-T22b: the minimal planner's rule 6 record on a (mocked) hybrid GPU plan. With the fused
    kernels loading, the report has no naive-cuEq line, also with allow_naive_cueq=True; with naive
    kernels allowed, it has the line."""
    test_minimal_planner = pytest.importorskip("test_minimal_planner")
    naive = report.DOWNGRADES["allow-naive-cueq"]
    for allow in (False, True):
        plan = test_minimal_planner.plan_label("mh1-r2scan", allow_naive_cueq=allow)
        assert plan.engine.backend == "hybrid"
        assert plan.decision("allow_naive_cueq")[2].code == "fused-cueq"
        lines = make_report(plan).setup_text().splitlines()
        assert not any(naive in line for line in lines), lines
        assert not any(line.startswith("why:") for line in lines), lines
    planner = test_minimal_planner.minimal(
        soft=test_minimal_planner.software(fused_cueq=(False, "cuequivariance_ops_torch cannot be imported")))
    plan = test_minimal_planner.plan_label("mh1-r2scan", planner=planner, allow_naive_cueq=True)
    assert plan.engine.backend == "hybrid"
    assert plan.decision("allow_naive_cueq")[1] == "on"
    assert plan.decision("allow_naive_cueq")[2].code == "allow-naive-cueq"
    assert "allow_naive_cueq" in plan.allowances
    lines = make_report(plan).setup_text().splitlines()
    assert any(line.startswith("why:") and naive in line for line in lines), lines


def test_unknown_code_is_named_not_dropped():
    error = PlanRefused.from_reasons([Reason("future rule", code="some-future-code")])
    text = make_report(plan=None, refusal=error, integrator=IntegratorConfig()).setup_text()
    assert "refused: some-future-code: future rule" in text


def test_reproducible_shows_the_9_3_refusal():
    test_minimal_planner = pytest.importorskip("test_minimal_planner")
    with pytest.raises(PlanRefused) as caught:
        test_minimal_planner.plan_label("mp0-medium", workload=Workload(max_temperature_K=300.0, reproducible=True))
    assert "reproducible-family-not-qualified" in caught.value.codes
    text = make_report(plan=None, refusal=caught.value, integrator=IntegratorConfig()).setup_text()
    assert "reproducible=True is qualified for plain cuEq MH-1 only (rule 5, §9.3)" in text


def test_real_plans_from_the_minimal_planner():
    test_minimal_planner = pytest.importorskip("test_minimal_planner")
    plan = test_minimal_planner.plan_label("mh1-r2scan")
    text = make_report(plan).setup_text()
    on = sum(plan.decision(f"RewriteConfig.{name}")[1] == "on" for name in REWRITE_FIELDS)
    assert f"plan:    graph, hybrid cuEq/OEq, float32, {on} rewrites on, {len(REWRITE_FIELDS) - on} off" in text
    soft = test_minimal_planner.software(oeq=(False, "openequivariance is not installed"))
    plain = test_minimal_planner.plan_label("mh1-r2scan", planner=test_minimal_planner.minimal(soft=soft))
    lines = make_report(plain).setup_text().splitlines()
    assert any(line.startswith("why:") and report.DOWNGRADES["backend-cueq-oeq-unavailable"] in line
               for line in lines)


# --------------------------------------------------------------------------- metadata


def test_to_metadata_is_json_and_follows_the_stages():
    rep = make_report()
    first = rep.to_metadata()
    assert first["speed"]["status"] == "pending"
    assert first["text"] == rep.setup_text()
    assert first["setup_s"] == SETUP and first["setup_total_s"] == 42.0
    rep.record_speed(0.0041, 1000, 2000)
    assert rep.to_metadata()["speed"] == {"status": "measured", "seconds_per_step": 0.0041,
                                          "first_step": 1000, "last_step": 2000}
    rep.finish(0.0042, steps=10_000)
    record = rep.to_metadata()
    assert record["speed"]["status"] == "whole-run mean"
    assert record["measured_window"] == {"seconds_per_step": 0.0041, "first_step": 1000, "last_step": 2000}
    assert json.loads(json.dumps(record)) == record
    assert record["licence"] == {"licence": "ASL", "source": "label table"}


def test_written_under_report_by_t10(tmp_path):
    rep = make_report()
    record = provenance.metadata_record(report=rep)
    assert record["report"] == rep.to_metadata()
    provenance.write_metadata(tmp_path / "metadata.json", record)
    assert json.loads((tmp_path / "metadata.json").read_text())["report"]["text"] == rep.setup_text()


def test_import_is_light():
    import subprocess
    import sys
    code = "import sys, fastmace.report; print('torch' in sys.modules, 'mace' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_the_mc_line():  # 3-T15
    from fastmace.mc import Displace, Swap

    mc = {"moves": (Swap(pairs=[("Na", "Nb")], every=100, attempts=50), Displace(delta_A=0.1, every=10, attempts=8)),
          "ranges": [[0, 1000], [3000, 5000]], "blocks": 30, "device_ms": 24.0, "timed_attempts": 120,
          "counters": {"swap": {"attempted": 100, "impossible": 0, "accepted": 7},
                       "displace": {"attempted": 80, "impossible": 0, "accepted": 30}}}
    text = make_report(mc=mc).text()
    line = next(line for line in text.splitlines() if line.startswith("mc:"))
    assert "swap Na-Nb every 100, 50 attempts (exchange); accepted 7/100" in line
    assert "displace δ 0.1 Å every 10, 8 attempts; accepted 30/80" in line
    assert "0.2 ms/attempt over 30 blocks" in line  # 24 ms over the 120 timed attempts
    assert "attached over steps 0–1,000, 3,000–5,000" in line
    assert "MC: experimental: not yet qualified" in text
    metadata = make_report(mc=mc).to_metadata()["mc"]
    assert metadata["ranges"] == [[0, 1000], [3000, 5000]] and metadata["timed_attempts"] == 120
    untimed = make_report(mc={**mc, "device_ms": None, "timed_attempts": 0}).text()
    assert "ms/attempt" not in untimed and "30 blocks" in untimed
    assert "\nmc:" not in make_report().text() and make_report().to_metadata()["mc"] is None
