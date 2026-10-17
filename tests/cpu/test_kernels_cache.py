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
