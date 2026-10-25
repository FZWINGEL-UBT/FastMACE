"""The examples are runnable scripts: each imports, parses its arguments and prints --help (plan task 6-T07).

No model is loaded and no GPU is touched: ``main`` is only called with ``--help``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = ("01_nvt_anneal_nacl.py", "02_npt_d3_melt_quench_nbcl5.py", "03_swap_mc_anneal.py",
           "04_gpu_relaxations.py", "05_port_ase_script.py")


def _module(name):
    spec = importlib.util.spec_from_file_location(f"fastmace_example_{name[:2]}", EXAMPLES / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_is_listed():
    assert sorted(p.name for p in EXAMPLES.glob("[0-9]*.py")) == sorted(SCRIPTS)
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for name in SCRIPTS:
        assert name in readme


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_help(name, capsys):
    module = _module(name)
    before = set(sys.modules)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for option in ("--checkpoint", "--steps", "--output"):
        assert option in text
    assert "torch" not in set(sys.modules) - before  # --help loads nothing heavy


def test_swap_example_is_the_shipped_protocol():
    pytest.importorskip("tomli" if sys.version_info < (3, 11) else "tomllib")
    from fastmace.cli.run import md_arguments, parse_config
    from fastmace.protocols import runner

    path, _ = runner.protocol_file("swap_anneal_NaNbOCl4_short")
    data = parse_config(path.name, path.read_text(encoding="utf-8"))
    data["structure"] = str(runner.DATA / data["structure"])
    data["output"] = {**data["output"], "directory": "swap_anneal"}
    stages, rest = runner.split_stages(data)
    structure, options = md_arguments(rest)
    ex_structure, ex_options, ex_stages = _module("03_swap_mc_anneal.py").protocol()
    assert ex_structure == structure
    assert ex_options == options
    assert runner._relax(ex_stages["relax"]) == stages["relax"]
    from fastmace.protocols.profiles import _quench

    assert _quench(ex_stages["quench"]) == stages["quench"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_compiles_and_has_usage(name):
    source = (EXAMPLES / name).read_text(encoding="utf-8")
    compile(source, name, "exec")
    assert f"python examples/{name}" in source  # the docstring shows how to run it
