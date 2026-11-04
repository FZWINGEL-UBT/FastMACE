"""Per-GPU default sets (kernels/gpu_models.py): an A100 entry stands for every A100 model (owner, 4 November 2026);
TABLE_W's per-GPU default for the generic family."""

import torch

from fastmace.kernels.gpu_models import GPUModels, is_a100
from fastmace.kernels.oeq_streaming import fused_forward, wprime_backward
from fastmace.planner import fixed

A100S = ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB PCIe", "NVIDIA A100-PCIE-40GB")


def test_a100_names():
    assert all(is_a100(name) for name in A100S)
    assert not any(is_a100(name) for name in ("NVIDIA A10", "NVIDIA A100X1", "NVIDIA RTX A1000", "NVIDIA A40", None))


def test_an_a100_entry_covers_every_a100():
    models = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti"))
    assert all(name in models for name in A100S)
    assert "NVIDIA GeForce RTX 5060 Ti" in models
    assert "NVIDIA RTX 4000 Ada Generation" not in models
    assert "NVIDIA A100-SXM4-40GB" not in GPUModels(("NVIDIA GeForce RTX 5060 Ti",))
    assert "NVIDIA A100-PCIE-40GB" in GPUModels(("NVIDIA L4",)) + GPUModels(("NVIDIA A100 80GB PCIe",))
    assert models == ("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti")  # still a tuple


def test_a100_defaults_cover_every_model():
    for models in (fixed.DECISION_66_WPRIME_GPUS, fixed.DECISION_66_TF32X3_GPUS,
                   fixed.DECISION_66_GENERIC_WPRIME_GPUS, fixed.DECISION_66_GENERIC_RADIAL_TABLES_GPUS,
                   fused_forward.SKIP_ZERO_COLUMN_GPUS, fused_forward.TENSOR_CORES_GPUS,
                   wprime_backward.TABLE_W_GPUS, wprime_backward.BLOCK_MESSAGE_GPUS):
        assert all(name in models for name in A100S)
    assert not any(name in fixed.DECISION_66_GATE_EPILOGUE_GPUS for name in A100S)  # no A100 measured there


def test_table_w_default(monkeypatch):
    assert wprime_backward.TABLE_W is None
    assert "NVIDIA GeForce RTX 5060 Ti" in wprime_backward.TABLE_W_GPUS
    assert "NVIDIA RTX 4000 Ada Generation" in wprime_backward.TABLE_W_GPUS
    assert "NVIDIA L4" not in wprime_backward.TABLE_W_GPUS
    assert not wprime_backward.table_w_default(torch.device("cpu"), generic=True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100-PCIE-40GB")
    assert wprime_backward.table_w_default(torch.device("cuda", 0), generic=True)
    assert not wprime_backward.table_w_default(torch.device("cuda", 0), generic=False)  # MH-1: not measured
    monkeypatch.setattr(wprime_backward, "TABLE_W", False)
    assert not wprime_backward.table_w_default(torch.device("cuda", 0), generic=True)
    monkeypatch.setattr(wprime_backward, "TABLE_W", True)
    assert wprime_backward.table_w_default(torch.device("cpu"), generic=False)


def test_block_message_default(monkeypatch):
    """BLOCK_MESSAGE (owner, 4 November 2026: "Yes, Ada and A100"): on for the RTX 4000 Ada and every A100, off on the
    RTX 5060 Ti (measured slower) and elsewhere; the constant overrides."""
    assert wprime_backward.BLOCK_MESSAGE is None
    assert "NVIDIA RTX 4000 Ada Generation" in wprime_backward.BLOCK_MESSAGE_GPUS
    assert "NVIDIA GeForce RTX 5060 Ti" not in wprime_backward.BLOCK_MESSAGE_GPUS
    assert not wprime_backward.block_message_default(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100 80GB PCIe")
    assert wprime_backward.block_message_default(torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 5060 Ti")
    assert not wprime_backward.block_message_default(torch.device("cuda", 0))
    monkeypatch.setattr(wprime_backward, "BLOCK_MESSAGE", True)
    assert wprime_backward.block_message_default(torch.device("cuda", 0))

