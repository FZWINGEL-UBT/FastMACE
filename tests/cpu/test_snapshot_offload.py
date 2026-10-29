"""Capture snapshots in host memory (plans/capacity.md, CAP-T05; ``Registry.snapshot(offload_bytes=...)``).

Around each capture and warm-up the graph engine copies every captured buffer and restores it afterwards. On the
device those copies sit beside the warm-up's activations, at the force stage's memory peak (0.5 GiB at 27,000 NaCl
atoms: the neighbour list and the noise batch). With ``offload_bytes`` the snapshot keeps buffers of at least that
size in (pinned, on CUDA) host memory; restoring copies them back bit for bit.
"""

import torch

from fastmace.engine import graph
from fastmace.engine.registry import Registry


def registry(tensors):
    return Registry(lambda: tensors)


def test_large_buffers_go_to_the_host_and_come_back_unchanged():
    big = torch.arange(4096, dtype=torch.float64)
    small = torch.ones(4, dtype=torch.int64)
    reg = registry({"engine.mid.0": small, "state.rng.batch": big})
    saved = reg.snapshot(offload_bytes=1024)
    copies = {name: value for name, (_, value) in saved.values.items()}
    assert copies["state.rng.batch"].device.type == "cpu" and copies["state.rng.batch"] is not big
    assert copies["engine.mid.0"].device == small.device
    original = big.clone()
    big.mul_(-3.0)
    small.add_(7)
    reg.restore(saved)
    assert torch.equal(big, original) and torch.equal(small, torch.ones(4, dtype=torch.int64))


def test_without_offload_the_snapshot_stays_on_the_device():
    big = torch.arange(4096, dtype=torch.float64)
    saved = registry({"state.rng.batch": big}).snapshot()
    (_, value), = saved.values.values()
    assert value.device == big.device and value.data_ptr() != big.data_ptr()


def test_the_engine_offloads_capture_snapshots_above_a_mebibyte():
    assert graph.CAPTURE_SNAPSHOT_OFFLOAD_BYTES == 2**20
