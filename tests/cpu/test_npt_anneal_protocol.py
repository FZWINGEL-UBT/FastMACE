"""3b-T07: the NPT melt-quench protocol as data (owner decision D1 = SPEC §18 decision 40).

NbCl5 3x3x3 on MH-1 matpes_r2scan with D3-BJ, hydrostatic NPT at 0 GPa along the anneal profile at
1 fs, with the library's barostat defaults, a fixed-cell relaxation first and the anneal quench.
"""

from __future__ import annotations

import pytest

from fastmace.cli import run as run_command
from fastmace.protocols import PROFILES, ScheduleState, runner

FULL, SHORT = "npt_anneal_NbCl5", "npt_anneal_NbCl5_short"


def _data(name):
    path, shipped = runner.protocol_file(name)
    assert shipped and path.parent == runner.DATA
    return run_command.parse_toml(path.read_text(encoding="utf-8"))


def test_the_shipped_protocol_is_decision_40():
    from fastmace.config import EngineConfig
    from fastmace.io.output import Output

    data = _data(FULL)
    stages, rest = runner.split_stages(data)
    # Frames every 100 fs (not main.py's 10 fs, decision 36) keep the 1944-atom trajectory to a few
    # hundred MB; thermo, with the volume, stays at 10 fs.
    assert rest["output"] == {"trajectory_every": 100, "thermo_every": 10, "restart_every": 10_000}
    rest["output"] = run_command._with_output(rest, None)
    structure, options = run_command.md_arguments(rest)
    assert (runner.DATA / structure).is_file()
    assert options["model"] == "mh1" and options["head"] == "matpes_r2scan"
    assert options["dispersion"] == "d3bj"
    assert options["repeat"] == (3, 3, 3) and options["timestep_fs"] == 1.0
    assert options["schedule"] == PROFILES["anneal"].schedule
    # Hydrostatic NPT at 0 GPa with the library's barostat defaults: no integrator table.
    assert (options["ensemble"], options["pressure_GPa"], options["cell_mode"]) == ("NPT", 0.0, "hydrostatic")
    assert "integrator" not in options and "mc" not in options
    assert options["engine"] == EngineConfig(backend="hybrid")
    assert options["output"] == Output("NbCl5_R2SCAN.fastmace", trajectory_every=100, thermo_every=10,
                                       restart_every=10_000)
    anneal = PROFILES["anneal"]
    assert stages["relax"] == {"fmax": anneal.relax_fmax, "optimizer": anneal.relax_optimizer, "max_steps": 5000}
    assert stages["quench"] == dict(anneal.quench)


def test_the_short_protocol_differs_only_in_its_schedule():
    full, short = _data(FULL), _data(SHORT)
    assert {key: value for key, value in short.items() if key != "schedule"} == \
        {key: value for key, value in full.items() if key != "schedule"}
    position = ScheduleState(run_command._schedule(short["schedule"]), 1.0)
    assert position.max_target() == 1000.0 and position.resolved.target(position.total_steps) == 250.0
    assert position.total_steps == 27_000  # 2 ps up, 25 ps down: a tenth of the full 270,000 steps


def test_the_quench_has_frames_in_every_bin_at_100_fs():
    from fastmace.protocols import quench

    resolved = ScheduleState(PROFILES["anneal"].schedule, 1.0).resolved
    picks = quench.picks(resolved, 100, PROFILES["anneal"].quench)
    assert len(picks) == 21  # 2 per 100 K bin from 500 K on each ramp, and the final frame


def test_the_shipped_cif_is_the_r2scan_benchmark():
    from ase.io import read

    atoms = read(runner.DATA / "NbCl5_R2SCAN.cif")
    assert atoms.get_chemical_formula() == "Cl60Nb12" and len(atoms) == 72
    assert len(atoms.repeat((3, 3, 3))) == 1944
    a, b, c, *_ = atoms.cell.cellpar()
    assert (a, b, c) == pytest.approx((18.9641, 18.1777, 6.0072), abs=1e-4)
    assert "G:/Margraf/ResearchModule/Structures/Benchmark/R2SCAN/NbCl5.cif" in \
        (runner.DATA / "SOURCES.md").read_text(encoding="utf-8")


def test_protocol_names():
    assert {FULL, SHORT} <= set(runner.protocol_names())
