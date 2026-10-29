"""Capture snapshots in pinned host memory on CUDA (plans/capacity.md, CAP-T05)."""

import pytest
import torch

from fastmace.engine.registry import Registry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_large_buffers_are_pinned_on_the_host_and_restore_bitwise():
    big = torch.randn(1 << 20, device="cuda", dtype=torch.float64)
    small = torch.ones(8, device="cuda", dtype=torch.int64)
    reg = Registry(lambda: {"state.rng.batch": big, "engine.mid.0": small})
    saved = reg.snapshot(offload_bytes=2**20)
    copies = {name: value for name, (_, value) in saved.values.items()}
    assert copies["state.rng.batch"].device.type == "cpu" and copies["state.rng.batch"].is_pinned()
    assert copies["engine.mid.0"].device.type == "cuda"
    original = big.clone()
    big.mul_(-2.0)
    small.add_(3)
    reg.restore(saved)
    torch.cuda.synchronize()
    assert torch.equal(big, original) and torch.equal(small, torch.ones(8, device="cuda", dtype=torch.int64))
