"""fastmace.kernels.triton.cache: Triton and Inductor caches under the FastMACE cache root (SPEC §14)."""

import sys
from pathlib import Path

import pytest

from fastmace._cache import CacheLocationError
from fastmace.kernels.triton import cache


def test_settings_default_to_the_cache_root(tmp_path):
    environ = {"FASTMACE_CACHE": str(tmp_path / "root")}
    settings = cache.cache_settings(environ=environ)
    root = (tmp_path / "root").resolve()
    assert settings["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")
    assert settings["TRITON_CACHE_DIR"] == str(root / "triton")
    assert {settings[k] for k in ("TORCHINDUCTOR_FX_GRAPH_CACHE", "ENABLE_AOT_AUTOGRAD_CACHE",
                                  "TORCHINDUCTOR_AUTOGRAD_CACHE")} == {"1"}
    assert set(settings) == set(cache.VARIABLES)
    assert not root.exists()  # settings alone create nothing


def test_settings_for_an_explicit_directory(tmp_path):
    settings = cache.cache_settings(tmp_path / "run")
    base = (tmp_path / "run").resolve()
    assert settings["TORCHINDUCTOR_CACHE_DIR"] == str(base / "inductor")
    assert settings["TRITON_CACHE_DIR"] == str(base / "triton")


def test_a_directory_inside_the_environment_is_refused():
    with pytest.raises(CacheLocationError):
        cache.cache_settings(Path(sys.prefix) / "compile-cache")
    with pytest.raises(CacheLocationError):
        cache.cache_settings(environ={"FASTMACE_CACHE": str(Path(sys.prefix) / "cache")})


def test_configure_sets_the_variables_and_creates_the_directories(tmp_path):
    environ = {"FASTMACE_CACHE": str(tmp_path / "root")}
    metadata = cache.configure(environ=environ)
    root = (tmp_path / "root").resolve()
    assert metadata == {"inductor": str(root / "inductor"), "triton": str(root / "triton"), "shared": True}
    assert environ["TRITON_CACHE_DIR"] == str(root / "triton")
    assert (root / "inductor").is_dir() and (root / "triton").is_dir()
    assert cache.configure(tmp_path / "run", environ={})["shared"] is False


def test_cache_stats_returns_counts():
    stats = cache.cache_stats()
    assert all(isinstance(v, int) for v in stats.values())


def test_configure_defaults_fills_in_the_cache_root_and_creates_nothing(tmp_path):
    """OPT4-T05b: the production default puts Inductor and Triton under FASTMACE_CACHE (SPEC §14), not
    under Inductor's ``$TMPDIR/torchinductor_$USER`` or Triton's ``~/.triton``."""
    environ = {"FASTMACE_CACHE": str(tmp_path / "root")}
    assert set(cache.configure_defaults(environ)) == set(cache.VARIABLES)
    root = (tmp_path / "root").resolve()
    assert environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")
    assert environ["TRITON_CACHE_DIR"] == str(root / "triton")
    assert environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert not root.exists()  # Inductor and Triton create their directories when they first write
    assert cache.configure_defaults(environ) == ()  # idempotent


def test_configure_defaults_keeps_explicit_variables(tmp_path):
    environ = {"FASTMACE_CACHE": str(tmp_path / "root"), "TORCHINDUCTOR_CACHE_DIR": "/elsewhere/inductor",
               "TORCHINDUCTOR_FX_GRAPH_CACHE": "0"}
    assert "TORCHINDUCTOR_CACHE_DIR" not in cache.configure_defaults(environ)
    assert environ["TORCHINDUCTOR_CACHE_DIR"] == "/elsewhere/inductor"
    assert environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "0"
    assert environ["TRITON_CACHE_DIR"] == str((tmp_path / "root").resolve() / "triton")
    flags_only = {key: "x" for key in cache.DIRECTORIES}
    cache.configure_defaults(flags_only)  # no cache root needed when both directories are set
    assert flags_only["ENABLE_AOT_AUTOGRAD_CACHE"] == "1" and flags_only["TRITON_CACHE_DIR"] == "x"


def test_planner_process_setup_puts_the_compiler_caches_under_the_cache_root(tmp_path):
    from fastmace.config import EngineConfig
    from fastmace.planner import fixed

    environ = {"FASTMACE_CACHE": str(tmp_path / "root")}
    fixed.FixedPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ=environ).configure_process()
    root = (tmp_path / "root").resolve()
    assert environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")
    assert environ["TRITON_CACHE_DIR"] == str(root / "triton")


def test_configure_defaults_replaces_inductors_own_default(tmp_path):
    """Importing ``torch._dynamo`` (MACE does, while loading) writes Inductor's default
    ``$TMPDIR/torchinductor_$USER`` into the environment (``torch/_dynamo/package.py``). That value is torch's,
    not the caller's, so it is replaced; any other value is the caller's and is kept."""
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    environ = {"FASTMACE_CACHE": str(tmp_path / "root"), "TORCHINDUCTOR_CACHE_DIR": default_cache_dir()}
    assert "TORCHINDUCTOR_CACHE_DIR" in cache.configure_defaults(environ)
    assert environ["TORCHINDUCTOR_CACHE_DIR"] == str((tmp_path / "root").resolve() / "inductor")


def test_importing_dynamo_does_not_keep_the_cache_in_tmp(tmp_path):
    """In a fresh process: import fastmace, then torch._dynamo, then the planner's setup."""
    import os
    import subprocess

    code = ("import fastmace, os, torch._dynamo\n"
            "from fastmace.kernels.triton import cache\n"
            "before = os.environ.get('TORCHINDUCTOR_CACHE_DIR')\n"
            "cache.configure_defaults()\n"
            "print(before, os.environ['TORCHINDUCTOR_CACHE_DIR'], os.environ['TRITON_CACHE_DIR'])\n")
    environ = {key: value for key, value in os.environ.items() if key not in cache.VARIABLES}
    environ.update(FASTMACE_CACHE=str(tmp_path / "root"), TMPDIR=str(tmp_path / "tmp"), CUDA_VISIBLE_DEVICES="")
    (tmp_path / "tmp").mkdir()
    out = subprocess.run([sys.executable, "-c", code], env=environ, capture_output=True, text=True, timeout=300,
                         check=True).stdout.split()
    root = (tmp_path / "root").resolve()
    assert out[0] != "None"  # torch set its default at import (otherwise this test proves nothing)
    assert out[1:] == [str(root / "inductor"), str(root / "triton")]
