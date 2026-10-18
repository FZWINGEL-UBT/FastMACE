"""Acceptance tools of plan task 1a-T19: the gate driver's process layout, the A/B tool's arguments, the
acceptance suites and their summaries, all without a GPU or a model.

The gate's rule itself (summary, floors, non-finite values) is tested in test_bench.py. Here: the driver
writes the configurations, runs every reference in its own child process before the one production
process, and imports neither torch nor the research engine (conductor decisions after T10 and the
second outside review). Children are replaced by fakes that write what a child would.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY_ROOT / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


gate = _load(TOOLS / "force_stress_gate.py", "_tool_gate_t19")
ab_tool = _load(TOOLS / "ab_step_time.py", "_tool_ab_t19")
sys.path.insert(0, str(TOOLS / "acceptance"))
import suites  # noqa: E402  (run.py imports it by this name)

run = _load(TOOLS / "acceptance" / "run.py", "_tool_run_t19")
summarize = _load(TOOLS / "acceptance" / "summarize.py", "_tool_summarize_t19")


# --- fixtures: a fake baseline export with one benchmark CIF -------------------------------------------


@pytest.fixture
def export(tmp_path):
    from ase.build import bulk
    from ase.io import write

    root = tmp_path / "export"
    (root / "Experiments" / "Langevin_Thermostat" / "langevin").mkdir(parents=True)
    (root / "Experiments" / "Langevin_Thermostat" / "langevin" / "__init__.py").write_text("")
    (root / "COMMIT").write_text("d2faa9fd\n")
    structures = root / gate.BENCHMARK_DIRECTORY
    structures.mkdir(parents=True)
    write(structures / "NaCl.cif", bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    return root


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.model"
    path.write_bytes(b"not a model")
    return path


def _fake_child(record):
    """A runner that does what a child would: read its job, write values and a result."""

    def runner(argv, log_path):
        kind, job_file = argv[-3], Path(argv[-1])
        job = json.loads(job_file.read_text())
        assert job["kind"] == kind
        record.append(job)
        values = {}
        if kind == "reference":
            base, configs = gate.read_configurations(job["configurations"])
            offset = 0.0 if job["dtype"] == "float64" else 1e-6
            for key in job["keys"]:
                values[f"{job['structure']}/{key}"] = _model(configs[key], offset)
        else:
            for structure, entry in job["structures"].items():
                base, configs = gate.read_configurations(entry["configurations"])
                for key in entry["keys"]:
                    values[f"{structure}/{key}"] = _model(configs[key], 1.5e-6)
        gate.save_values(job["values"], values)
        Path(job["result"]).write_text(json.dumps(dict(status="complete", pid=len(record))))
        Path(log_path).write_text("ok\n")
        return 0

    return runner


def _model(atoms, offset):
    """A smooth fake force field, plus ``offset`` on every value (the 'rounding' of a path)."""
    forces = np.sin(atoms.positions) + offset
    stress = np.diag(np.diag(atoms.cell.array)) * 1e-3 + offset * 1e-2
    return float(np.sum(atoms.positions)) + offset, forces, stress


def _gate_argv(tmp_path, export, checkpoint, *extra):
    return ["--output", str(tmp_path / "out"), "--baseline", str(export), "--checkpoint", str(checkpoint),
            "--structures", "NaCl", "--repeat", "1", "1", "1", "--configs", "2", *extra]


# --- configurations ------------------------------------------------------------------------------------


def test_configurations_are_the_earlier_gate_draws():
    from ase.build import bulk

    base = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 1, 1))
    for cell_mode in gate.CELL_MODES:
        configs = gate.make_configurations(base, cell_mode, 3, 20261015, 0.02, 0.05)
        rng = np.random.default_rng(20261015)  # the loop of the gate before 1a-T19
        for atoms in configs:
            expected = base.copy()
            expected.set_cell(expected.cell.array @ gate.random_strain(rng, 0.02, cell_mode == "full"),
                              scale_atoms=True)
            expected.positions += 0.05 * rng.standard_normal(expected.positions.shape)
            np.testing.assert_array_equal(atoms.positions, expected.positions)
            np.testing.assert_array_equal(atoms.cell.array, expected.cell.array)


def test_configuration_files_round_trip_exactly(tmp_path):
    from ase.build import bulk

    base = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    configs = {mode: gate.make_configurations(base, mode, 2, 1, 0.02, 0.05) for mode in gate.CELL_MODES}
    gate.write_configurations(tmp_path / "c.npz", base, configs)
    read_base, read = gate.read_configurations(tmp_path / "c.npz")
    assert list(read) == ["hydrostatic-0", "hydrostatic-1", "full-0", "full-1"]
    np.testing.assert_array_equal(read_base.cell.array, base.cell.array)
    for mode, items in configs.items():
        for index, atoms in enumerate(items):
            got = read[gate.config_key(mode, index)]
            np.testing.assert_array_equal(got.positions, atoms.positions)
            np.testing.assert_array_equal(got.cell.array, atoms.cell.array)
            np.testing.assert_array_equal(got.numbers, atoms.numbers)
            assert all(got.pbc)


def test_structure_spec(export, tmp_path):
    name, path, repeat = gate.structure_spec("NaCl", (3, 3, 3), export)
    assert (name, path.name, repeat) == ("NaCl", "NaCl.cif", (3, 3, 3))
    water = tmp_path / "water_216.extxyz"
    water.write_text("")
    assert gate.structure_spec(f"{water}@1x1x1", (3, 3, 3), export)[::2] == ("water_216", (1, 1, 1))
    with pytest.raises(ValueError):
        gate.structure_spec(f"{water}@1x1", (3, 3, 3), export)
    with pytest.raises(FileNotFoundError):
        gate.structure_spec("NbCl5", (3, 3, 3), export)  # not in this export


# --- jobs ----------------------------------------------------------------------------------------------


def test_reference_jobs_one_process_per_structure_and_calculator():
    keys = {"A": ["hydrostatic-0", "full-0"], "B": ["hydrostatic-0"]}
    jobs = gate.reference_jobs(["A", "B"], keys)
    assert [(j["structure"], j["reference"], j["keys"]) for j in jobs] == [
        ("A", "f64", keys["A"]), ("A", "ase_f32", keys["A"]), ("B", "f64", keys["B"]), ("B", "ase_f32", keys["B"])]
    assert [j["dtype"] for j in jobs[:2]] == ["float64", "float32"]
    assert len({j["name"] for j in jobs}) == len(jobs)


def test_reference_jobs_one_process_per_configuration():
    jobs = gate.reference_jobs(["A"], {"A": ["hydrostatic-0", "full-0"]}, scope="configuration")
    assert [(j["reference"], j["keys"]) for j in jobs] == [
        ("f64", ["hydrostatic-0"]), ("f64", ["full-0"]), ("ase_f32", ["hydrostatic-0"]), ("ase_f32", ["full-0"])]
    with pytest.raises(ValueError):
        gate.reference_jobs(["A"], {"A": []}, scope="process")


def test_child_argv_takes_a_launcher_prefix(tmp_path):
    job = tmp_path / "j.json"
    job.write_text(json.dumps(dict(kind="production")))
    argv = gate.child_argv("bash /x/oeq_no_nvcc.sh /env '/cache dir' /env/bin/python", "/t/gate.py", job)
    assert argv == ["bash", "/x/oeq_no_nvcc.sh", "/env", "/cache dir", "/env/bin/python", "/t/gate.py",
                    "production", "--job", str(job)]
    with pytest.raises(ValueError):
        gate.child_argv("", "/t/gate.py", job)


def test_values_round_trip(tmp_path):
    values = {"NaCl/full-1": (1.5, np.ones((2, 3)), np.eye(3)), "Na/Cl/hydrostatic-0": (-2.0, np.zeros((1, 3)),
                                                                                         np.zeros((3, 3)))}
    gate.save_values(tmp_path / "v.npz", values)
    got = gate.load_values(tmp_path / "v.npz")
    assert set(got) == set(values)
    for key, (energy, forces, stress) in values.items():
        assert got[key][0] == energy
        np.testing.assert_array_equal(got[key][1], forces)
        np.testing.assert_array_equal(got[key][2], stress)


def test_assemble_row():
    f64 = (10.0, np.zeros((2, 3)), np.zeros((3, 3)))
    f32 = (10.001, np.full((2, 3), 1e-6), np.full((3, 3), 1e-8))
    prod = (10.002, np.full((2, 3), 2e-6), np.full((3, 3), 3e-8))
    row = gate.assemble_row("full-2", 2, f64, f32, prod)
    assert (row["cell_mode"], row["index"]) == ("full", 2)
    assert row["production"]["forces"]["max_abs"] == pytest.approx(2e-6)
    assert row["ase_f32"]["energy_meV_atom"] == pytest.approx(0.5)
    assert gate.summarize([row])["passed"]


# --- the driver ----------------------------------------------------------------------------------------


def test_driver_runs_every_reference_before_the_production_process(tmp_path, export, checkpoint):
    jobs = []
    meta = gate.main(_gate_argv(tmp_path, export, checkpoint, "--engine", "fastmace"), runner=_fake_child(jobs),
                     gpu_check=lambda: [])
    kinds = [job["kind"] for job in jobs]
    assert kinds == ["reference", "reference", "production"]
    assert [job["dtype"] for job in jobs[:2]] == ["float64", "float32"]
    assert jobs[-1]["engine"] == "fastmace"
    assert meta["status"] == "complete" and meta["passed"]
    assert len(meta["structures"]["NaCl"]["configs"]) == 4  # 2 configurations x 2 cell modes
    assert [p["kind"] for p in meta["processes"]] == kinds
    written = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert written["summary"]["NaCl"]["passed"]
    assert written["baseline"]["commit"] == "d2faa9fd"


def test_driver_fails_when_a_production_error_exceeds_the_rule(tmp_path, export, checkpoint):
    jobs = []
    runner = _fake_child(jobs)

    def bad(argv, log_path):
        code = runner(argv, log_path)
        job = jobs[-1]
        if job["kind"] == "production":
            values = {k: (e, f + 1e-3, s) for k, (e, f, s) in gate.load_values(job["values"]).items()}
            gate.save_values(job["values"], values)
        return code

    meta = gate.main(_gate_argv(tmp_path, export, checkpoint), runner=bad, gpu_check=lambda: [])
    assert meta["status"] == "complete" and not meta["passed"]
    assert not meta["summary"]["NaCl"]["forces"]["passed"]


def test_driver_reports_a_failed_child(tmp_path, export, checkpoint):
    with pytest.raises(gate.GateJobError, match="reference-NaCl-f64"):
        gate.main(_gate_argv(tmp_path, export, checkpoint), runner=lambda argv, log: 3, gpu_check=lambda: [])
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["processes"][0]["exit_code"] == 3


def test_driver_stops_when_the_gpu_is_busy_before_a_child(tmp_path, export, checkpoint):
    calls = iter([[], [], [4242]])
    with pytest.raises(gate.GateJobError, match="GPU busy"):
        gate.main(_gate_argv(tmp_path, export, checkpoint), runner=_fake_child([]), gpu_check=lambda: next(calls))


def test_driver_refuses_main_args_for_fastmace(tmp_path, export, checkpoint):
    with pytest.raises(SystemExit):
        gate.parse_args(_gate_argv(tmp_path, export, checkpoint, "--engine", "fastmace", "--main-args", "--no-x"))


def test_driver_needs_the_baseline_export_for_its_references(tmp_path, checkpoint):
    with pytest.raises(SystemExit):
        gate.parse_args(["--output", str(tmp_path / "o"), "--checkpoint", str(checkpoint)])


def test_driver_imports_neither_torch_nor_the_research_engine(tmp_path, export, checkpoint):
    script = f"""
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate", {str(TOOLS / "force_stress_gate.py")!r})
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
def runner(argv, log):
    job = json.loads(Path(argv[-1]).read_text())
    values = {{}}
    groups = [(job["structure"], job["configurations"], job["keys"])] if job["kind"] == "reference" else [
        (s, e["configurations"], e["keys"]) for s, e in job["structures"].items()]
    for structure, path, keys in groups:
        base, configs = gate.read_configurations(path)
        for key in keys:
            a = configs[key]
            values[structure + "/" + key] = (0.0, a.positions * 0, a.cell.array * 0)
    gate.save_values(job["values"], values)
    return 0
gate.main({_gate_argv(tmp_path, export, checkpoint)!r}, runner=runner, gpu_check=lambda: [])
bad = sorted(m for m in sys.modules if m == "torch" or m.startswith(("torch.", "langevin", "mace")))
print(json.dumps(bad))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_old_invocation_without_subcommand_is_the_driver(tmp_path, export, checkpoint):
    jobs = []
    gate.main(["run", *_gate_argv(tmp_path, export, checkpoint)], runner=_fake_child(jobs), gpu_check=lambda: [])
    assert jobs[-1]["engine"] == "baseline"


# --- the A/B tool's arguments --------------------------------------------------------------------------


# The engine options workflow.run_structure passes to GPUEngine at the baseline tag (main.py's plan).
MAIN_PY_ENGINE_OPTIONS = ("factor_species", "fold_cutoff", "sort_receivers", "density_first", "reassociate_first",
                          "joint_radial_input", "fused_folded_backward", "segmented_first_sum", "gpu_neighbour_list",
                          "block_linear", "grouped_linear", "merged_residual", "species_skip", "streaming_conv",
                          "fused_radial_hidden", "speculative_graph", "pipelined")


def test_baseline_sides_get_every_engine_option_main_py_passes():
    assert set(ab_tool.FLAGS) == set(MAIN_PY_ENGINE_OPTIONS)
    assert set(gate.FLAG_NAMES) == set(MAIN_PY_ENGINE_OPTIONS)
    assert set(ab_tool.NPT_KEYWORDS) == {"pressure_GPa", "barostat_tau_fs", "barostat_mass_factor"}


def test_ab_tool_accepts_a_fastmace_only_process_without_a_baseline(tmp_path):
    args = ab_tool.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--sides", "fastmace"])
    assert args.sides == ["fastmace"] and args.baseline is None


def test_ab_tool_needs_the_export_for_a_baseline_side(tmp_path):
    with pytest.raises(SystemExit):
        ab_tool.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--sides", "fastmace", "baseline"])


def test_ab_tool_npt_and_build_order(tmp_path):
    args = ab_tool.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--baseline", "b", "--sides",
                               "fastmace", "baseline", "--ensemble", "npt", "--cell-mode", "full", "--dump-forces"])
    assert args.sides == ["fastmace", "baseline"]
    assert (args.ensemble, args.cell_mode, args.dump_forces) == ("npt", "full", True)


def test_ab_tool_fastmace_integrator_follows_the_arguments(tmp_path):
    args = ab_tool.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--sides", "fastmace",
                               "--ensemble", "npt", "--seed", "7"])
    config = ab_tool.fastmace_integrator(args)
    assert (config.ensemble, config.cell_mode, config.seed) == ("npt", "hydrostatic", 7)


# --- suites --------------------------------------------------------------------------------------------


def test_gate_labels_are_the_19_labels():
    from fastmace.model.labels import LABELS, MACE_OFF_LABELS

    assert len(suites.GATE_LABELS) == 19
    assert set(suites.GATE_LABELS) == set(LABELS)
    assert suites.MACE_OFF_LABELS == MACE_OFF_LABELS
    assert set(suites.CHECKPOINT_OF) == set(LABELS)


def test_checkpoint_files_are_the_release_names():
    from fastmace.model.labels import LABELS

    for label, entry in LABELS.items():
        assert suites.CHECKPOINT_FILES[suites.CHECKPOINT_OF[label]] == entry.checkpoint.url.rsplit("/", 1)[1]


def _flag_values(argv, flag):
    argv = list(argv)
    start = argv.index(flag) + 1
    end = next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))
    return argv[start:end]


@pytest.mark.parametrize("machine", ["local", "murphy"])
def test_gate_jobs(machine):
    m = suites.MACHINES[machine]
    root = m.path(m.root) / "run"
    jobs = suites.gate_jobs(m, root)
    assert len(jobs) == 19
    for job in jobs:
        label = _flag_values(job.argv, "--model")[0]
        structures = _flag_values(job.argv, "--structures")
        assert _flag_values(job.argv, "--engine") == ["fastmace"]
        if label in suites.MACE_OFF_LABELS:
            assert [Path(s.split("@")[0]).name for s in structures] == [
                "urea_cod1008787.cif", "naphthalene_cod2311090.cif", "paracetamol_cod2104364.cif",
                "water_216.extxyz"]
            assert structures[-1].endswith("@1x1x1")
        else:
            assert structures == ["NaCl", "NaNbO3", "NbCl5"]
        # references in the plain environment, production in the hybrid one, without the launcher
        assert _flag_values(job.argv, "--reference-command")[0].endswith("mace_cueq_latest/bin/python")
        assert _flag_values(job.argv, "--production-command")[0].endswith("mace_cueq_oeq/bin/python")
        assert job.output.startswith(str(root))


def _jobs_by_name(machine="murphy"):
    m = suites.MACHINES[machine]
    return {job.name: job for job in suites.speed_jobs(m, m.path(m.root) / "run")}


def test_speed_primary_set_three_processes_one_with_fastmace_built_first():
    jobs = _jobs_by_name()
    primary = [jobs[f"ab-mh1-nvt-r{i}"] for i in (1, 2, 3)]
    for job in primary:
        assert _flag_values(job.argv, "--structures") == ["NaCl", "NaNbO3", "NbCl5"]
        assert _flag_values(job.argv, "--model") == ["mh1-r2scan"]
        assert _flag_values(job.argv, "--blocks") == ["12"]
        assert _flag_values(job.argv, "--ensemble") == ["nvt"]
        assert "--dump-forces" in job.argv and job.timing
    assert [_flag_values(job.argv, "--sides") for job in primary] == [
        ["baseline", "fastmace"], ["fastmace", "baseline"], ["baseline", "fastmace"]]
    assert [_flag_values(job.argv, "--pattern") for job in primary] == [["abab"], ["abab"], ["abba"]]


def test_speed_npt_and_secondary_sets():
    jobs = _jobs_by_name()
    npt = jobs["ab-mh1-npt"]
    assert _flag_values(npt.argv, "--ensemble") == ["npt"]
    for model in ("mp0-medium", "polar1-s", "omol0-xl"):
        job = jobs[f"ab-{model}-nvt"]
        assert _flag_values(job.argv, "--structures") == ["NaCl", "NbCl5"]
        assert _flag_values(job.argv, "--sides") == ["baseline", "fastmace"]


def test_speed_controls_are_single_engine_processes_and_fastmace_needs_no_launcher():
    jobs = _jobs_by_name()
    controls = [job for name, job in jobs.items() if name.startswith("control-")]
    assert len(controls) == 2 * len(suites.speed_configurations())
    for job in controls:
        sides = _flag_values(job.argv, "--sides")
        assert len(sides) == 1
        assert _flag_values(job.argv, "--blocks") == ["4"]
        uses_launcher = any(word.endswith("oeq_no_nvcc.sh") for word in job.argv)
        cueq = _flag_values(job.argv, "--backend") == ["cueq"]
        assert uses_launcher == (sides == ["baseline"] and not cueq)
        assert ("--dump-forces" in job.argv) == (sides == ["fastmace"])
        if sides == ["fastmace"]:
            assert "--baseline" not in job.argv
    for name, job in jobs.items():
        if name.startswith("ab-"):
            hybrid = _flag_values(job.argv, "--backend") == ["hybrid"]
            # the research engine's hybrid needs the launcher; plain cuEq runs in its own environment
            assert any(word.endswith("oeq_no_nvcc.sh") for word in job.argv) == hybrid
            assert job.argv[0].endswith("mace_cueq_latest/bin/python") == (not hybrid)


def test_speed_measures_the_real_cueq_plan():
    job = _jobs_by_name()["ab-mh1-cueq-nvt"]
    assert _flag_values(job.argv, "--backend") == ["cueq"]
    assert _flag_values(job.argv, "--sides") == ["baseline", "fastmace"]
    assert "--main-args" not in job.argv  # main.py's plan for --backend cueq, nothing added


def test_speed_jobs_run_a_b_before_controls():
    names = list(_jobs_by_name())
    last_ab = max(i for i, n in enumerate(names) if n.startswith("ab-"))
    first_control = min(i for i, n in enumerate(names) if n.startswith("control-"))
    assert last_ab < first_control


def test_workstation_paths_stay_in_the_nas_home():
    m = suites.MACHINES["leto"]
    root = m.path(m.root) / "run"
    env = suites.environment(m, root)
    for key in ("FASTMACE_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME"):
        assert env[key].startswith("/mnt/homes/zwingel/")
    for job in suites.speed_jobs(m, root) + suites.gate_jobs(m, root) + suites.workflow_jobs(m, root):
        assert job.output.startswith("/mnt/homes/zwingel/")
        assert not any(str(word).startswith("/tmp") for word in job.argv)
    with pytest.raises(suites.LocationError):
        suites.check_location(m, "/scratch/run")
    with pytest.raises(suites.LocationError):
        suites.check_location(m, "/mnt/homes/zwingel/../other/run")


def test_nothing_goes_to_tmp_on_workstations():
    with pytest.raises(suites.LocationError):
        suites.check_location(suites.MACHINES["murphy"], "/tmp/fastmace")
    assert suites.check_location(suites.MACHINES["local"], "~/fastmace-acceptance/x").is_absolute()


def test_workflow_jobs_cover_both_engines():
    m = suites.MACHINES["murphy"]
    names = [job.name for job in suites.workflow_jobs(m, m.path(m.root) / "run")]
    for engine in ("gpu", "fastmace"):
        assert {f"main-smoke-NaNbO3-{engine}", f"anneal-NaCl-{engine}", f"quench-NaCl-{engine}"} <= set(names)
        assert names.index(f"anneal-NaCl-{engine}") < names.index(f"quench-NaCl-{engine}")
    for job in suites.workflow_jobs(m, m.path(m.root) / "run"):
        uses_launcher = any(word.endswith("oeq_no_nvcc.sh") for word in job.argv)
        assert uses_launcher == job.name.endswith("-gpu")  # main.py --engine fastmace needs no launcher


# --- run.py --------------------------------------------------------------------------------------------


def test_idle_verdict():
    gpus = run.parse_gpus("NVIDIA GeForce RTX 5060 Ti, 610.88, 3, 4100, 16311\n")
    assert gpus[0]["utilization_percent"] == 3.0 and gpus[0]["memory_total_MiB"] == 16311.0
    assert run.idle_verdict([], gpus, timing=True)[0]
    assert not run.idle_verdict([123], gpus)[0]
    busy = run.parse_gpus("NVIDIA GeForce RTX 5060 Ti, 610.88, 40, 9000, 16311")
    assert run.idle_verdict([], busy, timing=False)[0]  # utilization matters for timing jobs only
    idle, reasons = run.idle_verdict([], gpus, windows_gpus=busy, timing=True)
    assert not idle and "Windows" in reasons[0]


def test_idle_check_waits_for_utilization_to_settle_but_not_for_a_process():
    busy, idle = dict(idle=False, compute_pids=[]), dict(idle=True, compute_pids=[])
    busy_then_idle = [dict(busy), dict(busy), idle]
    records = iter(busy_then_idle)
    slept = []
    record = run.settled_idle_check(lambda: next(records), sleep=slept.append)
    assert record["idle"] and record["attempts"] == 3 and slept == [5.0, 5.0]
    busy = run.settled_idle_check(lambda: dict(idle=False, compute_pids=[42]), sleep=slept.append)
    assert not busy["idle"] and busy["attempts"] == 1
    never = run.settled_idle_check(lambda: dict(idle=False, compute_pids=[]), settle_s=10, sleep=lambda s: None)
    assert not never["idle"] and never["attempts"] == 3


def test_run_stops_at_a_busy_gpu_and_records_everything(tmp_path, monkeypatch):
    machine = suites.Machine(name="test", kind="local", root=str(tmp_path / "root"), fastmace=str(REPOSITORY_ROOT),
                             baseline=str(tmp_path / "export"), models=str(tmp_path / "models"),
                             research_module=str(tmp_path / "rm"), environments=str(tmp_path / "envs"),
                             cache=str(tmp_path / "cache"))
    monkeypatch.setitem(suites.MACHINES, "test", machine)
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "COMMIT").write_text("d2faa9fd\n")
    executed = []

    def execute(job, env, log):
        executed.append(job.name)
        assert env["PYTHONPATH"] == str(REPOSITORY_ROOT)
        return 0

    checks = iter([True, True, False])
    record = run.main(["speed", "--machine", "test", "--run-id", "r1"], execute=execute,
                      check=lambda job: dict(idle=next(checks), reasons=[]))
    assert executed == ["ab-mh1-nvt-r1", "ab-mh1-nvt-r2"]
    assert record["status"] == "stopped" and "ab-mh1-nvt-r3" in record["stopped"]
    manifest = json.loads((tmp_path / "root" / "r1" / "manifest.json").read_text())
    jobs = manifest["runs"][-1]["jobs"]
    assert [j["status"] for j in jobs] == ["complete", "complete", "gpu-busy"]
    assert manifest["runs"][-1]["baseline_commit"] == "d2faa9fd"
    assert "fastmace_commit" in manifest["runs"][-1]  # None in WSL on a Windows worktree (decision after wave 1b)


def test_run_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    machine = suites.Machine(name="t", kind="local", root=str(tmp_path / "root"), fastmace="/f", baseline="/b",
                             models="/m", research_module="/r", environments="/e", cache=str(tmp_path / "c"))
    monkeypatch.setitem(suites.MACHINES, "t", machine)
    run.main(["gate", "--machine", "t", "--run-id", "x", "--dry-run", "--gate-labels", "off23-small"])
    assert "gate-off23-small" in capsys.readouterr().out
    assert not (tmp_path / "root").exists()


def test_launch_script_detaches():
    text = (TOOLS / "acceptance" / "launch.sh").read_text()
    assert "setsid nohup" in text and "< /dev/null" in text and "&" in text


# --- summaries -----------------------------------------------------------------------------------------


def _side(median, blocks, setup=10.0, memory=2**30):
    return dict(median_ms=median, block_ms=blocks, setup_s=setup, peak_memory_bytes=memory, spread=0.0)


def _ab_meta(sides, medians, blocks=12, pattern="abab", valid=True):
    return dict(model="mh1-r2scan", ensemble="nvt", pattern=pattern, sides=sides, build_order=sides,
                side_kinds={s: s for s in sides}, protocol=dict(blocks=blocks),
                structures={"NaCl": dict(natoms=216, sides={s: _side(m, [m] * blocks) for s, m in zip(sides, medians)},
                                         observables_difference={"temperature_K": 1e-4},
                                         compile_during_timing=dict(valid=valid, changes={}))})


def test_a_run_that_compiled_while_timed_is_invalid():
    row = summarize.ab_rows("ab-x-r1", _ab_meta(["baseline", "fastmace"], [4.0, 4.0], valid=False))[0]
    assert not row["valid"] and not row["passed"]
    assert "invalid" in summarize.markdown(dict(speed=[row], controls=[], forces=[], gate=[]))


class FakeInductorConfig:
    post_grad_custom_pre_pass = None


class FakeKey:
    def __init__(self, uuid):
        self._uuid = uuid

    def uuid(self):
        return self._uuid


def test_custom_pass_slot_gives_each_side_its_own_key():
    config = FakeInductorConfig()
    slot = ab_tool.CustomPassSlot(config)
    langevin, fastmace = FakeKey("langevin-unfused-full-addmm-v2"), FakeKey("fastmace-unfused-full-addmm-v1")
    slot.enter("baseline")
    assert config.post_grad_custom_pre_pass is None
    config.post_grad_custom_pre_pass = langevin  # the research engine installs its key
    slot.leave("baseline")
    slot.enter("fastmace")
    assert config.post_grad_custom_pre_pass is None  # FastMACE would refuse the other key
    config.post_grad_custom_pre_pass = fastmace
    slot.leave("fastmace")
    slot.enter("baseline")  # the next structure: the research engine finds its own key again
    assert config.post_grad_custom_pre_pass is langevin
    assert slot.uuids() == {"baseline": "langevin-unfused-full-addmm-v2", "fastmace": "fastmace-unfused-full-addmm-v1"}


def test_compile_changes():
    before = {"stats": {"unique_graphs": 4, "calls_captured": 10}, "frames": {"ok": 4, "total": 4}}
    same = {"stats": {"unique_graphs": 4, "calls_captured": 10}, "frames": {"ok": 4, "total": 4}}
    assert ab_tool.compile_changes(before, same) == ({}, False)
    calls = {"stats": {"unique_graphs": 4, "calls_captured": 12}, "frames": {"ok": 4, "total": 4}}
    changes, recompiled = ab_tool.compile_changes(before, calls)
    assert changes == {"stats.calls_captured": [10, 12]} and not recompiled
    graph = {"stats": {"unique_graphs": 5, "calls_captured": 10}, "frames": {"ok": 5, "total": 5}}
    assert ab_tool.compile_changes(before, graph)[1]
    assert ab_tool.compile_changes(before, {**same, "recompiles": {"cache_size_limit": 1}})[1]


def test_speed_ratio_is_fastmace_over_baseline_whatever_the_build_order():
    a = summarize.ab_rows("ab-x-r1", _ab_meta(["baseline", "fastmace"], [4.0, 4.04]))[0]
    b = summarize.ab_rows("ab-x-r2", _ab_meta(["fastmace", "baseline"], [4.04, 4.0]))[0]
    assert a["ratio"] == pytest.approx(1.01) == b["ratio"]
    assert a["passed"] and b["passed"]


def test_speed_verdict_reports_criterion_and_noise_floor():
    verdict = summarize.speed_verdict(1.031, 12)
    assert not verdict["passed"]
    assert verdict["criterion"] == 0.03 and verdict["noise_floor"] == 0.0016
    assert summarize.speed_verdict(0.971, 4)["passed"]
    assert summarize.speed_verdict(1.0, 4)["noise_floor"] == 0.0032
    # a noise floor above 3 % would become the criterion
    assert summarize.speed_verdict(1.035, 12, limit=0.001)["criterion"] == 0.0016


def test_compare_forces():
    a = np.ones((4, 3))
    assert summarize.compare_forces(a, a.copy())["bitwise"]
    near = summarize.compare_forces(a, a + 2e-6)
    assert near["passed"] and not near["bitwise"] and near["max_abs"] == pytest.approx(2e-6)
    assert not summarize.compare_forces(a, a + 1e-3)["passed"]
    assert not summarize.compare_forces(a, np.ones((3, 3)))["passed"]
    assert not summarize.compare_forces(a, np.full((4, 3), np.nan))["passed"]


def test_summarize_run_pairs_ab_processes_with_their_controls(tmp_path):
    def write(name, meta, forces=None):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "metadata.json").write_text(json.dumps(meta))
        if forces is not None:
            (directory / "forces").mkdir()
            np.save(directory / "forces" / "NaCl-fastmace.npy", forces)

    forces = np.arange(12.0).reshape(4, 3)
    write("ab-mh1-nvt-r1", _ab_meta(["baseline", "fastmace"], [4.0, 4.02]), forces)
    write("ab-mh1-nvt-r2", _ab_meta(["fastmace", "baseline"], [4.02, 4.0]), forces + 1e-3)
    write("control-mh1-nvt-fastmace", _ab_meta(["fastmace"], [4.01], blocks=4), forces)
    write("control-mh1-nvt-baseline", _ab_meta(["baseline"], [4.0], blocks=4))
    write("ab-mh1-nvt-r9.invalid-species-skip", _ab_meta(["baseline", "fastmace"], [4.0, 9.0]))  # ignored
    write("gate-mh1-r2scan", dict(model="mh1-r2scan", engine="fastmace", status="complete", passed=True,
                                  summary={"NaCl": dict(passed=True, forces=dict(ratio=1.1), stress=dict(ratio=0.9))}))
    summary = summarize.summarize_run(tmp_path)
    assert len(summary["speed"]) == 2 and summary["passed"]["speed"]
    assert {r["job"]: r["passed"] for r in summary["forces"]} == {"ab-mh1-nvt-r1": True, "ab-mh1-nvt-r2": False}
    assert summary["passed"]["forces"] is False
    assert summary["gate"][0]["worst_force_ratio"] == 1.1 and summary["passed"]["gate"]
    assert len(summary["controls"]) == 2
    text = summarize.markdown(summary)
    assert "| ab-mh1-nvt-r1 |" in text and "| mh1-r2scan | fastmace |" in text


def test_slug():
    assert summarize._slug("ab-mh1-nvt-r2") == "mh1-nvt"
    assert summarize._slug("ab-mh1-npt") == "mh1-npt"
    assert summarize._slug("ab-mp0-medium-nvt") == "mp0-medium-nvt"
    assert summarize._slug("ab-polar1-s-nvt") == "polar1-s-nvt"
