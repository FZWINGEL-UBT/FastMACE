"""The OEq extra must select a prebuilt extension on every supported CPython."""

import re
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
WHEELS = ROOT / "tools/release/oeq_wheels"


@pytest.mark.parametrize("extra", ["oeq", "full"])
@pytest.mark.parametrize("version, selected", [("3.10", True), ("3.12", True), ("3.13", True), ("3.14", False)])
def test_extra_selects_prebuilt_for_supported_python(extra, version, selected):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(value) for value in project["optional-dependencies"][extra]]
    extension, = [req for req in requirements if req.name == "fastmace-oeq-ext"]
    env = {**default_environment(), "sys_platform": "linux", "platform_machine": "x86_64",
           "python_version": version, "python_full_version": version + ".0"}
    assert extension.marker.evaluate(env) is selected


def test_extension_metadata_and_build_matrix_include_python_313():
    project = tomllib.loads((WHEELS / "package/pyproject.toml").read_text())["project"]
    supported = SpecifierSet(project["requires-python"])
    assert "3.13" in supported and "3.14" not in supported
    builder = (WHEELS / "build_in_container.sh").read_text()
    targets = re.search(r'PYTHONS=\$\{PYTHONS:-"([^"]+)"\}', builder).group(1).split()
    assert "cp313-cp313" in targets
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]
