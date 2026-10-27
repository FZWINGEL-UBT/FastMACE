"""fastmace.kernels.oeq_streaming without a GPU: vendored files, licence, lazy imports, template helpers.

The vendored OpenEquivariance PR 223 files keep their BSD 3-Clause LICENSE and NOTICE.md byte-identical
(SPEC §14). The hashes below are those of the research engine's copies at ResearchModule tag
research-engine-baseline-20261017; with FASTMACE_BASELINE set, the files are also compared with the
export byte for byte.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPOSITORY_ROOT / "fastmace" / "kernels" / "oeq_streaming"

# SHA-256 of the files that must stay byte-identical to the research engine's.
UNCHANGED = {
    "LICENSE": "65bb7f1dc061df3e82aa51dcc162b71cc9bbef2df43a799b6f0d14cefced0908",
    "NOTICE.md": "d29a77ed17266b6d9caf8b6a8503725baedc54cfec4ec5248e8d98f6deb09d22",
    "factorized_projected.cuh": "d37652d5b456fcedbd9bc07f40c9bcb3691fcc51675132928319f05dac5b2a3b",
}
# Vendored Python files whose only changes are the package path of their imports and the import
# grouping ruff's isort asks for: (old, new) replacements applied to the baseline copy.
IMPORT_PATH_ONLY = ("schedule.py", "jinja_env.py")
IMPORT_EDITS = (("langevin.oeq_streaming", "fastmace.kernels.oeq_streaming"),
                ("import numpy as np\n\nfrom openequivariance", "import numpy as np\nfrom openequivariance"))


def baseline_directory():
    root = os.environ.get("FASTMACE_BASELINE")
    if not root:
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    directory = Path(root).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin" / "oeq_streaming"
    assert directory.is_dir(), f"FASTMACE_BASELINE={root} has no langevin/oeq_streaming"
    return directory


@pytest.mark.parametrize("name", sorted(UNCHANGED))
def test_vendored_file_is_present_and_unchanged(name):
    data = (PACKAGE / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == UNCHANGED[name]
    assert b"\r\n" not in data  # LF, as in git


def test_licence_is_bsd_3_clause_of_lbnl():
    text = (PACKAGE / "LICENSE").read_text(encoding="utf-8")
    assert "Redistribution and use in source and binary forms" in text
    assert "Neither the name" in text  # the third clause
    notice = (PACKAGE / "NOTICE.md").read_text(encoding="utf-8")
    assert "OpenEquivariance" in notice and "BSD 3-Clause" in notice


@pytest.mark.parametrize("name", sorted(UNCHANGED))
def test_vendored_file_equals_baseline(name):
    assert (PACKAGE / name).read_bytes() == (baseline_directory() / name).read_bytes()


@pytest.mark.parametrize("name", IMPORT_PATH_ONLY)
def test_vendored_python_differs_from_baseline_only_in_the_import_path(name):
    old = (baseline_directory() / name).read_text(encoding="utf-8")
    new = (PACKAGE / name).read_text(encoding="utf-8")
    for before, after in IMPORT_EDITS:
        old = old.replace(before, after)
    assert new == old


def test_template_is_found_next_to_the_package():
    from fastmace.kernels.oeq_streaming import jinja_env

    environment = jinja_env.get_jinja_environment()
    assert environment.get_template("factorized_projected.cuh") is not None
    with pytest.raises(ValueError):
        jinja_env.get_jinja_environment("hip")


def test_scalar_helpers():
    from fastmace.kernels.oeq_streaming.jinja_env import cpp_scalar_literal, cpp_scalar_type

    assert cpp_scalar_type(np.float32) == "float" and cpp_scalar_type(np.float64) == "double"
    with pytest.raises(ValueError):
        cpp_scalar_type(np.float16)
    assert cpp_scalar_literal(0.5, "float") == "static_cast<scalar_t>(0x1.0000000000000p-1f)"
    assert cpp_scalar_literal(0.5, "double") == "static_cast<scalar_t>(0x1.0000000000000p-1)"


def test_custom_ops_are_in_the_fastmace_namespace():
    import torch

    import fastmace.kernels.oeq_streaming  # noqa: F401

    assert hasattr(torch.ops.fastmace, "streaming_conv_forward")
    assert hasattr(torch.ops.fastmace, "streaming_conv_backward")


def test_importing_the_package_does_not_import_openequivariance():
    # (cuda.bindings is not checked: torch 2.14 with CUDA imports it itself.)
    code = ("import fastmace, sys\n"
            "import fastmace.kernels.oeq_streaming\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('openequivariance', 'langevin')))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_l2_chunk_edges_float32_and_float64(monkeypatch):
    """chunked_conv_backward's chunk: a quarter of the L2 for float32 weights (OPT6-T01); for float64 weights
    UNCHUNKED_L2_MULTIPLE L2s (f64-T03: the emulated input-gradient products need large chunks)."""
    from fastmace.kernels.oeq_streaming import runtime

    monkeypatch.setattr(runtime, "l2_bytes", lambda device: 32 * 2**20)
    assert runtime.l2_chunk_edges("cuda:0", 1280, 4) == 1536
    want = runtime.UNCHUNKED_L2_MULTIPLE * 32 * 2**20 // (1280 * 8) // 128 * 128
    assert runtime.l2_chunk_edges("cuda:0", 1280, 8) == want
    assert runtime.l2_chunk_edges("cuda:0", 1280, 8) == 13056
