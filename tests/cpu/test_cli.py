"""1b-T22: ``fastmace md``, ``init`` and ``run`` (SPEC §7.5, §7.7; owner decision D5).

The commands only turn flags or a TOML file into ``fm.md``'s arguments and call it, so the tests
check that mapping: every §7.5 ``md`` flag reaches its ``fm.md`` argument, ``init`` writes every
``fm.md`` argument, ``run`` reads them back and refuses an unknown key by name, and
``fastmace init <flags> | fastmace run`` gives the run of ``fastmace md <flags>`` on a tiny model
(same plan, same ``metadata.json`` except timestamps). Refusals come through T19's ``main``.
"""

from __future__ import annotations

import inspect
import io
import json
import re

import pytest

from fastmace import cli
from fastmace.cli import init as init_command
from fastmace.cli import md as md_command
from fastmace.cli import run as run_command

NA_CL = (11, 17)

# SPEC §7.5's md flags, each with a sample argv and the fm.md argument and value it must give.
SPEC_FLAGS = {
    "--model": (["--model", "mh1"], "model", "mh1"),
    "--head": (["--head", "matpes_r2scan"], "head", "matpes_r2scan"),
    "--charge": (["--charge", "-1"], "charge", -1),
    "--spin-multiplicity": (["--spin-multiplicity", "2"], "spin_multiplicity", 2),
    "--structure": (["--structure", "NaCl.cif"], "structure", "NaCl.cif"),
    "--ensemble": (["--ensemble", "NPT"], "ensemble", "NPT"),
    "--temperature": (["--temperature", "450"], "temperature_K", 450.0),
    "--pressure": (["--pressure", "1.5"], "pressure_GPa", 1.5),
    "--cell-mode": (["--cell-mode", "full"], "cell_mode", "full"),
    "--timestep": (["--timestep", "1.0"], "timestep_fs", 1.0),
    "--steps": (["--steps", "1000"], "steps", 1000),
    "--repeat": (["--repeat", "3", "2", "1"], "repeat", (3, 2, 1)),
    "--dispersion": (["--dispersion", "d3bj"], "dispersion", "d3bj"),
    "--d3-functional": (["--dispersion", "d3bj", "--d3-functional", "pbe"], "dispersion", "D3:pbe"),
    "--seed": (["--seed", "7"], "seed", 7),
    "--check": (["--check", "gate"], "check", "gate"),
    "--output": (["--output", "run/"], "output", "run/"),
    "--sha256": (["--sha256", "ab" * 32], "sha256", "ab" * 32),
    "--trusted": (["--trusted"], "trusted", True),
}

REQUIRED = ["--model", "mh1", "--structure", "NaCl.cif"]


def _md_parameters():
    from fastmace.quick import md

    return list(inspect.signature(md).parameters)


def _parse(command, argv):
    return cli.build_parser().parse_args([command, *argv])


# --------------------------------------------------------------------------- the command table


def test_command_table_holds_md_init_and_run():
    assert cli.COMMANDS["md"] == "fastmace.cli.md"
    assert cli.COMMANDS["init"] == "fastmace.cli.init"
    assert cli.COMMANDS["run"] == "fastmace.cli.run"


def test_choices_follow_the_library():
    from fastmace import quick, simulation

    assert md_command.CHECK_CHOICES == simulation.CHECK_MODES
    assert md_command.ENSEMBLE_CHOICES == quick.ENSEMBLES


# --------------------------------------------------------------------------- md flags -> fm.md


@pytest.mark.parametrize("flag", sorted(SPEC_FLAGS))
def test_every_spec_flag_maps_to_an_md_argument(flag):
    argv, name, expected = SPEC_FLAGS[flag]
    assert name in _md_parameters()
    extra = {"--model": REQUIRED[2:], "--structure": REQUIRED[:2]}.get(flag, REQUIRED)
    structure, options = md_command.md_arguments(_parse("md", [*extra, *argv]))
    value = structure if name == "structure" else options[name]
    if expected == "D3:pbe":
        from fastmace.longrange.d3 import D3

        assert value == D3(functional="pbe")
    else:
        assert value == expected
        assert type(value) is type(expected)


def test_flags_not_given_are_left_to_md_defaults():
    structure, options = md_command.md_arguments(_parse("md", REQUIRED))
    assert structure == "NaCl.cif"
    assert options == {"model": "mh1"}


def test_md_has_no_flag_outside_spec():
    parser = cli.build_parser()
    md_parser = parser._subparsers._group_actions[0].choices["md"]
    flags = {option for action in md_parser._actions for option in action.option_strings
             if option.startswith("--")} - {"--help", "--debug"}
    assert flags == set(SPEC_FLAGS)


def test_dispersion_none_is_no_dispersion():
    _, options = md_command.md_arguments(_parse("md", [*REQUIRED, "--dispersion", "none"]))
    assert "dispersion" not in options


def test_d3_functional_needs_d3bj(capsys):
    assert cli.main(["md", *REQUIRED, "--d3-functional", "pbe"]) == 2
    assert "--d3-functional needs --dispersion d3bj" in capsys.readouterr().err


def test_md_value_error_is_a_usage_error(capsys):
    # NVT with a pressure: fm.md's ValueError, before anything is loaded.
    assert cli.main(["md", *REQUIRED, "--head", "matpes_r2scan", "--temperature", "300", "--steps", "10",
                     "--pressure", "1"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("fastmace md: error: ") and "pressure_GPa" in err
    assert "Traceback" not in err


def test_d3bj_is_expanded_then_refused_at_plan_time(capsys):
    assert cli.main(["md", *REQUIRED, "--head", "matpes_r2scan", "--temperature", "300", "--steps", "10",
                     "--dispersion", "d3bj"]) == 1
    err = capsys.readouterr().err
    assert "D3: phase 3b" in err and "functional=r2scan" in err
    assert "Traceback" not in err


def test_explicit_d3_functional_is_refused_with_the_record(capsys):
    assert cli.main(["md", *REQUIRED, "--head", "matpes_r2scan", "--temperature", "300", "--steps", "10",
                     "--dispersion", "d3bj", "--d3-functional", "pbe"]) == 1
    err = capsys.readouterr().err
    assert "D3: phase 3b" in err and "functional=pbe" in err


def test_nve_is_refused_through_main(capsys):
    assert cli.main(["md", *REQUIRED, "--head", "matpes_r2scan", "--ensemble", "NVE", "--temperature", "300",
                     "--steps", "10"]) == 1
    assert "NVE: phase 4" in capsys.readouterr().err


# --------------------------------------------------------------------------- init


def _init_text(argv=()):
    out = io.StringIO()
    args = _parse("init", list(argv))
    import contextlib

    with contextlib.redirect_stdout(out):
        assert init_command.run(args) == 0
    return out.getvalue()


def _toml(text):
    return run_command.parse_toml(text)


def test_init_names_every_md_argument():
    text = _init_text()
    lines = [line.lstrip("# ").split("=")[0].strip() for line in text.splitlines()]
    tables = {line.strip("#[] ") for line in text.splitlines() if line.lstrip("# ").startswith("[")}
    for name in _md_parameters():
        assert name in lines or name in tables, name


def test_init_example_is_the_spec_example():
    data = _toml(_init_text())
    assert data == {"structure": "NaCl.cif", "model": "mh1", "head": "matpes_r2scan", "ensemble": "NVT",
                    "temperature_K": 300.0, "steps": 1_000_000, "output": "run/"}
    structure, options = run_command.md_arguments(data)
    assert structure == "NaCl.cif" and options["model"] == "mh1"


def test_init_commented_examples_parse():
    """Every commented line, uncommented, is valid TOML that run accepts (one key or table at a time)."""
    text = _init_text()
    blocks, current = [], None
    for line in text.splitlines():
        body = line[2:] if line.startswith("# ") else ""
        if re.fullmatch(r"\[\[?[a-z_]+\]\]?", body):
            current = [body]
            blocks.append(current)
        elif re.match(r"[A-Za-z_]+ = ", body):
            if current is not None:
                current.append(body)
            else:
                blocks.append([body])
        else:
            current = None
    assert blocks
    for block in blocks:
        data = _toml("\n".join(block))
        if "mc" in data or "integrator" in data:
            data.setdefault("temperature_K", 300.0)
        run_command.md_arguments({"structure": "NaCl.cif", "model": "mh1", **data})


def test_init_with_flags_writes_only_those(tmp_path):
    data = _toml(_init_text(["--model", "x.model", "--structure", "a.xyz", "--temperature", "500",
                             "--repeat", "2", "2", "2", "--trusted", "--dispersion", "d3bj", "--d3-functional",
                             "pbe"]))
    assert data == {"model": "x.model", "structure": "a.xyz", "temperature_K": 500.0, "repeat": [2, 2, 2],
                    "trusted": True, "dispersion": {"functional": "pbe"}}


def test_init_flags_and_md_flags_give_equal_arguments():
    argv = ["--model", "mh1", "--head", "matpes_r2scan", "--structure", "NaCl.cif", "--ensemble", "NPT",
            "--temperature", "450", "--pressure", "1.5", "--cell-mode", "full", "--timestep", "1.0", "--steps", "20",
            "--repeat", "3", "2", "1", "--dispersion", "d3bj", "--d3-functional", "r2scan", "--seed", "7",
            "--check", "gate", "--output", "run/", "--sha256", "ab" * 32, "--charge", "0",
            "--spin-multiplicity", "1"]
    expected = md_command.md_arguments(_parse("md", argv))
    assert run_command.md_arguments(_toml(_init_text(argv))) == expected


def test_init_strings_are_escaped():
    path = "C:\\runs\\\"quoted\" é"
    data = _toml(_init_text(["--model", "mh1", "--structure", path]))
    assert data["structure"] == path


# --------------------------------------------------------------------------- run's TOML


def test_run_refuses_an_unknown_key_naming_it(tmp_path, capsys):
    config = tmp_path / "run.toml"
    config.write_text('structure = "NaCl.cif"\nmodel = "mh1"\ntemprature_K = 300.0\n', encoding="utf-8")
    assert cli.main(["run", str(config)]) == 2
    err = capsys.readouterr().err
    assert "temprature_K" in err and "fastmace run: error:" in err


def test_run_needs_structure_and_model():
    with pytest.raises(ValueError, match="structure"):
        run_command.md_arguments({"model": "mh1"})
    with pytest.raises(ValueError, match="model"):
        run_command.md_arguments({"structure": "NaCl.cif"})


def test_run_reads_a_profile_schedule():
    from fastmace.protocols import PROFILES

    _, options = run_command.md_arguments({"structure": "a.xyz", "model": "mh1", "schedule": "anneal"})
    assert options["schedule"] == PROFILES["anneal"].schedule
    with pytest.raises(ValueError, match="slow"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1", "schedule": "nope"})


def test_run_reads_schedule_segments():
    from fastmace.protocols import hold, ramp

    data = _toml('structure = "a.xyz"\nmodel = "mh1"\n'
                 '[[schedule]]\nkind = "ramp"\nstart_K = 0.0\nend_K = 600.0\nps = 2.0\n'
                 '[[schedule]]\nkind = "hold"\ntemperature_K = 600.0\nps = 1.0\n')
    _, options = run_command.md_arguments(data)
    assert options["schedule"] == ramp(0.0, 600.0, ps=2.0) + hold(600.0, ps=1.0)


def test_run_reads_an_integrator_table():
    from fastmace import integrators, quick

    data = _toml('structure = "a.xyz"\nmodel = "mh1"\n[integrator]\ntype = "LangevinBAOAB"\n'
                 'temperature_K = 300.0\nexternal_stress_GPa = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]\n')
    _, options = run_command.md_arguments(data)
    assert options["integrator"] == integrators.LangevinBAOAB(
        temperature_K=300.0, friction_per_fs=quick.default_friction_per_fs(),
        external_stress_GPa=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="frction"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1",
                                  "integrator": {"type": "Langevin", "temperature_K": 1.0, "frction": 1.0}})
    with pytest.raises(ValueError, match="NoseHoover"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1", "integrator": {"type": "NoseHoover"}})


def test_run_reads_dispersion_and_output_tables():
    from fastmace.io.output import Output
    from fastmace.longrange.d3 import D3

    _, options = run_command.md_arguments({"structure": "a.xyz", "model": "mh1",
                                           "dispersion": {"functional": "pbe", "three_body": True},
                                           "output": {"directory": "run", "thermo_every": 10}})
    assert options["dispersion"] == D3(functional="pbe", three_body=True)
    assert options["output"] == Output("run", thermo_every=10)
    _, options = run_command.md_arguments({"structure": "a.xyz", "model": "mh1", "dispersion": "none"})
    assert "dispersion" not in options
    with pytest.raises(ValueError, match="cutof"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1",
                                  "dispersion": {"functional": "pbe", "cutof": 1.0}})


def test_run_mc_tables_become_move_records():
    from fastmace import mc

    _, options = run_command.md_arguments(_toml(
        'structure = "a.xyz"\nmodel = "mh1"\n[[mc]]\nmove = "swap"\npairs = [["Na", "Cl"]]\nevery = 100\n'
        'attempts = 50\n[[mc]]\nmove = "transmute"\nspecies = ["Na", "K"]\nevery = 10\nattempts = 1\n'
        'mu = {Na = 0.0, K = -0.2}\n'))
    assert options["mc"] == (mc.Swap(pairs=[("Na", "Cl")], every=100, attempts=50),
                             mc.Transmute(species=["Na", "K"], every=10, attempts=1, mu={"Na": 0.0, "K": -0.2}))
    _, options = run_command.md_arguments(_toml(
        'structure = "a.xyz"\nmodel = "mh1"\n[mc]\nmove = "displace"\ndelta_A = 0.1\nevery = 5\nattempts = 2\n'))
    assert options["mc"] == (mc.Displace(delta_A=0.1, every=5, attempts=2),)
    with pytest.raises(ValueError, match="move"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1", "mc": {"kind": "swap"}})
    with pytest.raises(ValueError, match="hop"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1", "mc": {"move": "hop"}})
    with pytest.raises(ValueError, match="atempts"):
        run_command.md_arguments({"structure": "a.xyz", "model": "mh1",
                                  "mc": {"move": "swap", "pairs": [["Na", "Cl"]], "every": 1, "atempts": 1}})


def test_init_documents_mc_without_a_phase_refusal():
    text = _init_text()
    assert "phase 3" not in text.replace("phase 3b", "")
    assert "[[mc]]" in text and 'move = "swap"' in text


def test_run_reads_standard_input(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('structure = "a.xyz"\nmodel = "mh1"\nhead = "matpes_r2scan"\n'
                                                 'ensemble = "NVE"\ntemperature_K = 1.0\nsteps = 1\n'))
    assert cli.main(["run", "-"]) == 1
    assert "NVE: phase 4" in capsys.readouterr().err


def test_run_bad_toml_is_a_usage_error(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("structure = \n", encoding="utf-8")
    assert cli.main(["run", str(config)]) == 2
    assert "bad.toml" in capsys.readouterr().err


# --------------------------------------------------------------------------- init | run == md


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))  # no licence warning noise
    return str(path)


@pytest.fixture
def cpu_planner(monkeypatch):
    """The planner of every Simulation: the CPU without compilation (as test_quick.py)."""
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.planner import minimal

    def planner(force_field):
        return minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                      process=constraints.ProcessRecord())

    monkeypatch.setattr(minimal, "default_planner", planner)


def _without_times(value):
    """A metadata record without its timestamps and timings: the report's measured speed, and its text,
    which prints that speed (the report's other fields are compared one by one)."""
    if isinstance(value, dict):
        return {key: _without_times(item) for key, item in value.items()
                if not (key.endswith(("_utc", "_s")) or "seconds" in key or key == "text")}
    if isinstance(value, list):
        return [_without_times(item) for item in value]
    return value


def _differences(first, second, path=""):
    """The paths where two JSON values differ."""
    if isinstance(first, dict) and isinstance(second, dict):
        return [found for key in sorted(set(first) | set(second))
                for found in _differences(first.get(key), second.get(key), f"{path}/{key}")]
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        return [found for index, (a, b) in enumerate(zip(first, second))
                for found in _differences(a, b, f"{path}/{index}")]
    return [] if first == second else [(path, first, second)]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_init_then_run_equals_md(model_path, cpu_planner, tmp_path, monkeypatch, capsys):
    from _support.structures import nacl

    structure = tmp_path / "nacl.extxyz"
    nacl(repeat=1, rattle=0.05, seed=3).write(structure)
    argv = ["--model", model_path, "--trusted", "--structure", str(structure), "--temperature", "300",
            "--steps", "3", "--seed", "5", "--output", "run"]

    direct = tmp_path / "direct"
    direct.mkdir()
    monkeypatch.chdir(direct)
    assert cli.main(["md", *argv]) == 0
    printed = capsys.readouterr().out
    assert "run" in printed

    via_toml = tmp_path / "via_toml"
    via_toml.mkdir()
    monkeypatch.chdir(via_toml)
    assert cli.main(["init", *argv]) == 0
    (via_toml / "run.toml").write_text(capsys.readouterr().out, encoding="utf-8")
    assert cli.main(["run", "run.toml"]) == 0
    capsys.readouterr()

    first = json.loads((direct / "run" / "metadata.json").read_text(encoding="utf-8"))
    second = json.loads((via_toml / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert first["created_utc"] and second["created_utc"]
    differing = _differences(_without_times(first), _without_times(second))
    assert not differing
    assert first.get("plan") == second.get("plan")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_run_with_mc_tables_runs_mc(model_path, cpu_planner, tmp_path, monkeypatch, capsys):
    from _support import fake_mc
    from _support.structures import nacl

    drivers = fake_mc.spy(monkeypatch)
    structure = tmp_path / "nacl.extxyz"
    nacl(repeat=1, rattle=0.05, seed=3).write(structure)
    config = tmp_path / "mc.toml"
    config.write_text(f'structure = "{structure.as_posix()}"\nmodel = "{model_path.replace(chr(92), "/")}"\n'
                      'trusted = true\ntemperature_K = 300.0\nsteps = 4\nseed = 5\n'
                      f'output = "{(tmp_path / "run").as_posix()}"\n'
                      '[[mc]]\nmove = "swap"\npairs = [["Na", "Cl"]]\nevery = 2\nattempts = 3\n', encoding="utf-8")
    assert cli.main(["run", str(config)]) == 0
    assert "swap Na-Cl every 2, 3 attempts" in capsys.readouterr().out
    assert [step for step, _ in drivers[0].spied] == [2, 4]
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["counters"]["mc"]["swap"]["attempted"] == 6
