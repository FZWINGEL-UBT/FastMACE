"""``fm.load`` configures the expandable-segments allocator before anything can touch CUDA (5-T07 finding).

The allocator reads ``PYTORCH_CUDA_ALLOC_CONF`` once, at the first CUDA allocation. Preparing the hybrid
(``prepare_hybrid_if_possible``) can touch CUDA, so a script that loads by label before it builds a
``Simulation`` hit the engine's process constraint ("set after the first CUDA allocation").
"""

import dataclasses

from fastmace.model import compile as compiling
from fastmace.model import loading


@dataclasses.dataclass
class _Loaded:
    oeq_runtime: object = None


def _record_order(monkeypatch, environ):
    seen = []
    monkeypatch.setattr(loading, "prepare_hybrid_if_possible",
                        lambda: seen.append(("prepare", environ.get("PYTORCH_CUDA_ALLOC_CONF"))) or "runtime")
    monkeypatch.setattr(loading, "load",
                        lambda *args, **kwargs: seen.append(("load", environ.get("PYTORCH_CUDA_ALLOC_CONF")))
                        or _Loaded())
    return seen


def test_load_sets_the_allocator_before_preparing_the_hybrid(monkeypatch):
    environ = {}
    monkeypatch.setattr(compiling, "configure_allocator",
                        lambda environ=None, **kw: environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                                                                      compiling.ALLOCATOR_SETTINGS))
    seen = _record_order(monkeypatch, environ)
    loaded = loading.load_pristine("mh1-r2scan", environ=environ)
    assert seen == [("prepare", compiling.ALLOCATOR_SETTINGS), ("load", compiling.ALLOCATOR_SETTINGS)]
    assert loaded.oeq_runtime == "runtime"


def test_load_keeps_a_user_allocator_setting(monkeypatch):
    environ = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    seen = _record_order(monkeypatch, environ)
    loading.load_pristine("mh1-r2scan", environ=environ)
    assert seen[0] == ("prepare", "max_split_size_mb:128")
    assert compiling.configure_allocator(environ=dict(environ), torch_version="2.14.1") == "max_split_size_mb:128"
