"""Per-GPU default sets (kernels/gpu_models.py): an A100 entry stands for every A100 model (owner, 4 November 2026);
TABLE_W's per-GPU default for the generic family."""

import torch

from fastmace.kernels.gpu_models import GPUModels, is_a100
from fastmace.kernels.oeq_streaming import fused_forward, wprime_backward
from fastmace.planner import fixed
from fastmace.rewrites import radial

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
                   wprime_backward.TABLE_W_GPUS, wprime_backward.BLOCK_MESSAGE_GPUS,
                   fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS):
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
    """BLOCK_MESSAGE: on for the RTX 5060 Ti, the RTX 4000 Ada and every A100, off elsewhere; the constant overrides."""
    assert wprime_backward.BLOCK_MESSAGE is None
    assert "NVIDIA RTX 4000 Ada Generation" in wprime_backward.BLOCK_MESSAGE_GPUS
    assert "NVIDIA GeForce RTX 5060 Ti" in wprime_backward.BLOCK_MESSAGE_GPUS
    assert not wprime_backward.block_message_default(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100 80GB PCIe")
    assert wprime_backward.block_message_default(torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA RTX A4000")
    assert not wprime_backward.block_message_default(torch.device("cuda", 0))
    monkeypatch.setattr(wprime_backward, "BLOCK_MESSAGE", True)
    assert wprime_backward.block_message_default(torch.device("cuda", 0))


def test_block_ldg_default(monkeypatch):
    """BLOCK_LDG: no GPU since round 2 (the RTX 5060 Ti's factored kernel is faster with plain loads); the constant
    overrides."""
    assert wprime_backward.BLOCK_LDG is None
    assert "__ldg(dout{{ bk[0] }} + int64_t(receiver)" in wprime_backward.TEMPLATE
    assert not wprime_backward.block_ldg_default(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 5060 Ti")
    assert not wprime_backward.block_ldg_default(torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA RTX 4000 Ada Generation")
    assert not wprime_backward.block_ldg_default(torch.device("cuda", 0))
    monkeypatch.setattr(wprime_backward, "BLOCK_LDG", True)
    assert wprime_backward.block_ldg_default(torch.device("cuda", 0))


def test_factored_and_symmetric_pairs_defaults(monkeypatch):
    """FACTORED and SYMMETRIC_PAIRS: on for the RTX 4000 Ada and every A100 (decision 66), FACTORED also on the RTX
    5060 Ti (round 2), off elsewhere; the constants override."""
    for module, constant, default, on_5060 in (
            (wprime_backward, "FACTORED", wprime_backward.factored_default, True),
            (radial, "SYMMETRIC_PAIRS", radial.symmetric_pairs_default, False)):
        assert getattr(module, constant) is None
        assert not default(torch.device("cpu"))
        for name, on in (("NVIDIA RTX 4000 Ada Generation", True), ("NVIDIA A100 80GB PCIe", True),
                         ("NVIDIA A100-SXM4-40GB", True), ("NVIDIA GeForce RTX 5060 Ti", on_5060),
                         ("NVIDIA RTX A4000", False)):
            monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None, name=name: name)
            assert default(torch.device("cuda", 0)) is on, (constant, name)
        monkeypatch.setattr(module, constant, True)
        assert default(torch.device("cuda", 0))
        monkeypatch.setattr(module, constant, False)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA RTX 4000 Ada Generation")
        assert not default(torch.device("cuda", 0))


def test_round2_defaults(monkeypatch):
    """Round 2 (5 November 2026, decision 66): HORNER on the A100s and the RTX 5060 Ti, SCALAR_PATHS and
    BACKWARD_WALK on the A100s, 80 registers on the RTX 5060 Ti, the fused first interaction (FUSED_FIRST_BACKWARD,
    TABLED_FORWARD) on all three measured models, BUTTERFLY (round 5) on the RTX 5060 Ti; off elsewhere; the constants
    override."""
    from fastmace.kernels.triton import edge_outer_sum, tabled_outer_sum
    from fastmace.rewrites.generic import modules as generic

    def walk(device):
        edge_outer_sum._walk_on.cache_clear()
        return edge_outer_sum.backward_walk(device)

    checks = (
        (wprime_backward, "HORNER", wprime_backward.horner_default, {"a100", "5060"}),
        (wprime_backward, "SCALAR_PATHS", wprime_backward.scalar_paths_default, {"a100"}),
        (wprime_backward, "BUTTERFLY", wprime_backward.butterfly_default, {"5060"}),
        (wprime_backward, "REDUX", wprime_backward.redux_default, set()),
        (edge_outer_sum, "BACKWARD_WALK", walk, {"a100"}),
        (generic, "FUSED_FIRST_BACKWARD", generic.fused_first_backward, {"a100", "5060", "ada"}),
        (tabled_outer_sum, "TABLED_FORWARD", tabled_outer_sum.tabled_forward, {"a100", "5060", "ada"}),
    )
    names = {"a100": "NVIDIA A100 80GB PCIe", "5060": "NVIDIA GeForce RTX 5060 Ti",
             "ada": "NVIDIA RTX 4000 Ada Generation", "a4000": "NVIDIA RTX A4000"}
    for module, constant, default, on in checks:
        assert getattr(module, constant) is None, constant
        assert not default(torch.device("cpu")), constant
        for key, name in names.items():
            monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None, name=name: name)
            assert bool(default(torch.device("cuda", 0))) is (key in on), (constant, name)
        monkeypatch.setattr(module, constant, True)
        assert default(torch.device("cuda", 0)), constant
        monkeypatch.setattr(module, constant, None)
    for name, registers in (("NVIDIA GeForce RTX 5060 Ti", 80), ("NVIDIA A100-SXM4-80GB", 0)):
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None, name=name: name)
        assert wprime_backward.max_registers_default(torch.device("cuda", 0)) == registers


def test_round4_defaults(monkeypatch):
    """Round 4 (5 November 2026, decision 66): VEC_SH on the A100s and the RTX 5060 Ti, VEC_A and TC_WFRAG on the
    A100s (where the tensor-core kernel runs); off elsewhere and on the CPU; the constants override."""
    names = {"a100": "NVIDIA A100 80GB PCIe", "5060": "NVIDIA GeForce RTX 5060 Ti",
             "ada": "NVIDIA RTX 4000 Ada Generation", "a4000": "NVIDIA RTX A4000"}
    checks = (
        ("VEC_SH", lambda name: fused_forward.vec_sh_default(16, name), {"a100", "5060"}),
        ("VEC_A", fused_forward.vec_a_default, {"a100"}),
        ("TC_WFRAG", fused_forward.tc_wfrag_default, {"a100"}),
    )
    for constant, default, on in checks:
        assert getattr(fused_forward, constant) is None, constant
        assert not default(None), constant
        for key, name in names.items():
            assert bool(default(name)) is (key in on), (constant, name)
        monkeypatch.setattr(fused_forward, constant, True)
        assert default(names["ada"]), constant
        monkeypatch.setattr(fused_forward, constant, False)
        assert not default(names["a100"]), constant
        monkeypatch.setattr(fused_forward, constant, None)
    assert not fused_forward.vec_sh_default(18, names["a100"])  # float4 staging needs ED % 4 == 0


def test_round5_edge_sh_default(monkeypatch):
    """Round 5 (5 November 2026, decision 66): EDGE_SH on the RTX 5060 Ti, the RTX 4000 Ada and every A100; off on
    other GPUs, the CPU and without a device; the constant overrides."""
    import torch

    from fastmace.rewrites.generic import modules as gm
    assert gm.EDGE_SH is None
    for name in ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB",
                 "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB"):
        assert gm.edge_sh_default(name), name
    for name in ("NVIDIA RTX A4000", "NVIDIA L4", None):
        assert not gm.edge_sh_default(name), name
    assert not gm.edge_sh_default(torch.device("cpu"))
    monkeypatch.setattr(gm, "EDGE_SH", True)
    assert gm.edge_sh_default("NVIDIA L4")
    monkeypatch.setattr(gm, "EDGE_SH", False)
    assert not gm.edge_sh_default("NVIDIA A100 80GB PCIe")


def test_energy_source_edit_defaults(monkeypatch):
    """Round 5 (5 November 2026, decision 66): HEAD_SLICE on the RTX 5060 Ti, the RTX 4000 Ada and every A100;
    GRAPH_SUM per its A/Bs; off on the CPU and other GPUs; the constants override."""
    from fastmace.model import energy

    class Model(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device_ = device

        def parameters(self, recurse=True):
            yield type("P", (), {"device": self.device_})()

    names = {"a100": "NVIDIA A100 80GB PCIe", "5060": "NVIDIA GeForce RTX 5060 Ti",
             "ada": "NVIDIA RTX 4000 Ada Generation", "a4000": "NVIDIA RTX A4000"}
    for constant, default, on in (("HEAD_SLICE", energy.head_slice_default, {"a100", "5060", "ada"}),
                                  ("GRAPH_SUM", energy.graph_sum_default, set())):
        assert getattr(energy, constant) is None, constant
        assert not default(Model(torch.device("cpu"))), constant
        for key, name in names.items():
            monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None, name=name: name)
            assert default(Model(torch.device("cuda", 0))) is (key in on), (constant, name)
        monkeypatch.setattr(energy, constant, True)
        assert default(Model(torch.device("cpu"))), constant
        monkeypatch.setattr(energy, constant, None)
