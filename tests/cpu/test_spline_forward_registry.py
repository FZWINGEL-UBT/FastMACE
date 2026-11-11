"""``spline_forward.shared``: equal ``SplineForward`` builds share the first registered object (ada-5060-screen,
10 November 2026).

The custom ops launch ``_REGISTRY[spline_id]``, and that object carries per-call state (the BLOCK_ORDER ``order`` its
force field binds before each call, the cached identity orders). A later equal build (another force field in the
process) used to replace the entry, so earlier force fields' captured graphs read the newer object's tensors, and a
replay after that object was replaced and freed read freed memory (CUDA illegal address in a five-engine A/B process
on the RTX 4000 Ada). The entry is never replaced, and every build uses the registered object, so each force field
binds its own order on the object its launches go through.
"""

import inspect
import types

import pytest
import torch

from fastmace.kernels.oeq_streaming import spline_forward, wprime_backward


def _kernel(content):
    return types.SimpleNamespace(id=content, order=None)


def test_the_first_build_stays_registered_and_later_equal_builds_use_it(monkeypatch):
    monkeypatch.setattr(spline_forward, "_REGISTRY", {})
    first, second, other = _kernel(7), _kernel(7), _kernel(8)
    assert spline_forward.shared(first) is first
    assert spline_forward.shared(second) is first      # not replaced: the second build is dropped
    assert spline_forward._REGISTRY[7] is first
    assert spline_forward.shared(other) is other       # another content id has its own entry
    assert spline_forward._REGISTRY == {7: first, 8: other}


def test_an_order_bound_through_a_later_build_reaches_the_launched_object(monkeypatch):
    monkeypatch.setattr(spline_forward, "_REGISTRY", {})
    engine_k = spline_forward.shared(_kernel(3))
    engine_j = spline_forward.shared(_kernel(3))
    engine_j.order = "order of engine j"               # what forces._bind_streaming_count does before a call
    assert spline_forward._REGISTRY[3].order == "order of engine j"
    assert engine_k is engine_j


def test_the_constructor_never_replaces_an_entry():
    source = inspect.getsource(spline_forward.SplineForward.__init__)
    assert "_REGISTRY.setdefault(self.id, self)" in source
    assert "_REGISTRY[self.id] = self" not in source


def test_the_wprime_build_binds_the_registered_forward():
    source = inspect.getsource(wprime_backward)
    assert "forward = registered_forward(SplineForward(" in source



def _shared_kernel():
    """A stand-in with ``SplineForward``'s per-call state (its ``launch_order`` runs on the CPU)."""
    return types.SimpleNamespace(id=5, order=None, _identity={})


def _launch_order(kernel, *args):
    return spline_forward.SplineForward.launch_order(kernel, *args)


def test_two_force_fields_of_different_sizes_each_launch_their_own_order(monkeypatch):
    monkeypatch.setattr(spline_forward, "_REGISTRY", {})
    kernel = spline_forward.shared(_shared_kernel())
    assert spline_forward.shared(_shared_kernel()) is kernel   # force field j's build: the same object
    small = torch.tensor([2, 0, 1], dtype=torch.int32)        # force field k: 3 receivers
    large = torch.tensor([4, 3, 0, 1, 2], dtype=torch.int32)  # force field j: 5 receivers
    kernel.order = small                                      # k binds, then calls (or captures)
    assert _launch_order(kernel, 3, "cpu") is small
    kernel.order = large                                      # j binds before its own call
    assert _launch_order(kernel, 5, "cpu") is large
    # a call with another row count than the bound order (k's rows while j's order is bound) never reads it
    identity = _launch_order(kernel, 3, "cpu")
    assert identity.tolist() == [0, 1, 2] and _launch_order(kernel, 3, "cpu") is identity  # cached per row count
    assert _launch_order(kernel, 7, "cpu").tolist() == list(range(7))                     # never too short
    explicit = torch.tensor([1, 0], dtype=torch.int32)
    assert _launch_order(kernel, 2, "cpu", explicit) is explicit


def test_an_order_of_another_dtype_is_refused(monkeypatch):
    monkeypatch.setattr(spline_forward, "_REGISTRY", {})
    kernel = spline_forward.shared(_shared_kernel())
    kernel.order = torch.arange(4)  # int64
    with pytest.raises(ValueError, match="BLOCK_ORDER"):
        _launch_order(kernel, 4, "cpu")
