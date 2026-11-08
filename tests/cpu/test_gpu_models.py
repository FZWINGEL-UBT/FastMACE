"""Per-GPU default sets (kernels/gpu_models.py): an A100 entry stands for every A100 model (owner, 4 November 2026);
TABLE_W's per-GPU default for the generic family."""

import torch

from fastmace.kernels.cutlass import linear_div, skip_select
from fastmace.kernels.gpu_models import GPUModels, family, is_a100, lookup
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


def test_families(monkeypatch):
    """Owner, 7 November 2026: a GPU tested on one variant counts for every variant of its family, unless the variants
    differ significantly in performance (the H100 SXM and PCIe cards stay separate)."""
    from fastmace.kernels.triton import radial_table as rt

    assert family("NVIDIA H200 NVL") == family("NVIDIA H200") == "H200"
    assert family("NVIDIA H20") == "NVIDIA H20" and family(None) is None
    assert "NVIDIA H200 NVL" in GPUModels(("NVIDIA H200",))
    assert "NVIDIA H100 PCIe" not in GPUModels(("NVIDIA H100 80GB HBM3",))
    assert "NVIDIA H100 80GB HBM3" not in GPUModels(("NVIDIA H200",))
    assert lookup({"NVIDIA H200": 8}, "NVIDIA H200 NVL") == 8
    assert lookup({"NVIDIA H200": 8}, "NVIDIA A100-SXM4-80GB") is None
    assert lookup({"NVIDIA A100-SXM4-80GB": 1, "NVIDIA A100-PCIE-40GB": 2}, "NVIDIA A100-PCIE-40GB") == 2
    assert rt.WARPS_GPUS == {"NVIDIA H200": 8}
    assert all("NVIDIA H200 NVL" in models for models in (fused_forward.WS_FORWARD_GPUS, fixed.DECISION_66_TF32X3_OMAT0_GPUS))


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
        (wprime_backward, "REDUX", wprime_backward.redux_default, {"5060"}),
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
    GRAPH_SUM on every A100; off on the CPU and other GPUs; the constants override."""
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
                                  ("GRAPH_SUM", energy.graph_sum_default, {"a100"})):
        assert getattr(energy, constant) is None, constant
        assert not default(Model(torch.device("cpu"))), constant
        for key, name in names.items():
            monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None, name=name: name)
            assert default(Model(torch.device("cuda", 0))) is (key in on), (constant, name)
        monkeypatch.setattr(energy, constant, True)
        assert default(Model(torch.device("cpu"))), constant
        monkeypatch.setattr(energy, constant, None)


def test_segment_rows_default(monkeypatch):
    """SEGMENT_ROWS (round 5, decision 66): on for every A100, off elsewhere and on the CPU; the constant overrides."""
    from fastmace.kernels.cutlass import segment_rows

    assert segment_rows.SEGMENT_ROWS is None and not segment_rows.segment_rows_default("cpu")
    for name, on in (("NVIDIA A100 80GB PCIe", True), ("NVIDIA A100-SXM4-40GB", True),
                     ("NVIDIA GeForce RTX 5060 Ti", False), ("NVIDIA RTX 4000 Ada Generation", False)):
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None, name=name: name)
        assert segment_rows.segment_rows_default(torch.device("cuda", 0)) is on, name
    monkeypatch.setattr(segment_rows, "SEGMENT_ROWS", False)
    assert not segment_rows.segment_rows_default(torch.device("cuda", 0))
    assert segment_rows.SEGMENT_GROUPED is None and not segment_rows.segment_grouped_default("cpu")
    monkeypatch.setattr(segment_rows, "SEGMENT_GROUPED", True)
    assert segment_rows.segment_grouped_default(torch.device("cuda", 0))


def test_h200_defaults():
    """The H200's decision-66 sets (5 November 2026, results/h200-defaults-20261105): every switch its MD-step A/Bs
    won in both build orders, by exact name; the backward walk (1.0000 / 1.0001) stays off. The tabled forward joined
    after its recapture fix (0.984 / 0.987, results/h200-wgmma-20261105); the wgmma product stays off."""
    from fastmace.kernels.triton import edge_outer_sum, tabled_outer_sum
    from fastmace.model import energy
    from fastmace.rewrites.generic import modules

    for models in (fixed.DECISION_66_GENERIC_RADIAL_TABLES_GPUS, fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS,
                   fixed.DECISION_66_TF32X3_OMAT0_GPUS, fixed.DECISION_66_GENERIC_WPRIME_GPUS,
                   fused_forward.SKIP_ZERO_COLUMN_GPUS, fused_forward.TENSOR_CORES_GPUS, fused_forward.VEC_SH_GPUS,
                   fused_forward.VEC_A_GPUS,
                   fused_forward.TC_WFRAG_GPUS, wprime_backward.TABLE_W_GPUS, wprime_backward.BLOCK_MESSAGE_GPUS,
                   wprime_backward.FACTORED_GPUS, wprime_backward.HORNER_GPUS, wprime_backward.SCALAR_PATHS_GPUS,
                   energy.HEAD_SLICE_GPUS, energy.GRAPH_SUM_GPUS,
                   modules.FUSED_FIRST_BACKWARD_GPUS, modules.EDGE_SH_GPUS, radial.SYMMETRIC_PAIRS_GPUS,
                   tabled_outer_sum.TABLED_FORWARD_GPUS, fused_forward.DEGREE_PARTS_GPUS):
        assert "NVIDIA H200" in models
    assert "NVIDIA A100-SXM4-80GB" in fused_forward.DEGREE_PARTS_GPUS  # 0.9920 / 0.9932
    assert "NVIDIA GeForce RTX 5060 Ti" not in fused_forward.DEGREE_PARTS_GPUS
    assert "NVIDIA H200" in fused_forward.LANE_INDICES_GPUS  # 0.9897 / 0.9913 (idea 15a, 6 November 2026)
    assert "NVIDIA GeForce RTX 5060 Ti" not in fused_forward.LANE_INDICES_GPUS  # kernel lab 1.024
    assert "NVIDIA H200" in fused_forward.W_GROUP_GPUS  # 0.9946 / 0.9942 (idea 17 at G = 2, 6 November 2026)
    # A100-SXM4: 12 warps with WFRAG_REGION, then W_GROUP (6 November 2026): 0.9718 / 0.9734 together
    assert "NVIDIA A100-SXM4-80GB" in fused_forward.W_GROUP_GPUS
    assert "NVIDIA A100-SXM4-80GB" in fused_forward.WFRAG_REGION_GPUS  # alone 0.9871 / 0.9887
    assert "NVIDIA H200" not in fused_forward.WFRAG_REGION_GPUS  # a tie: 1.0014 / 1.0010
    assert not fused_forward.A_SPLIT_GPUS  # H200 1.0311 / 1.0346 (slower)
    # Idea 27 (6 November 2026): X_LATE on both, MD H200 0.9884 / 0.9882, A100-SXM4 0.9937 / 0.9937
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(fused_forward.X_LATE_GPUS)
    assert "NVIDIA H200" not in fused_forward.K_PIPE_GPUS  # with X_LATE 0.9934 / 0.9927, worse than X_LATE alone
    # Idea 24 (6 November 2026): CG_FOLD on both, MD H200 0.9938 / 0.9959, A100-SXM4 0.9877 / 0.9898
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(fused_forward.CG_FOLD_GPUS)
    # Revision 13 (6 November 2026): A_LD128 on both (H200 0.9925 / 0.9956; A100-SXM4 with TP_NOFLUSH 0.9909 /
    # 0.9923); TP_NOFLUSH only on the A100-SXM4 (H200 0.9992 / 1.0018, noise)
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(fused_forward.A_LD128_GPUS)
    assert "NVIDIA A100-SXM4-80GB" in fused_forward.TP_NOFLUSH_GPUS
    assert "NVIDIA H200" not in fused_forward.TP_NOFLUSH_GPUS
    # Idea 31 (6 November 2026): LINEAR_DIV on both, MD H200 0.9912 / 0.9934, A100-SXM4 0.9896 / 0.9891
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(linear_div.LINEAR_DIV_GPUS)
    # Idea 30 (6 November 2026): WS_FORWARD on the H200, MD 0.9584 / 0.9610 (sm_90 only)
    assert "NVIDIA H200" in fused_forward.WS_FORWARD_GPUS
    # Idea 38 (6 November 2026): STAGE_DOUT on both, MD H200 0.9616 / 0.9638, A100-SXM4 0.9896 / 0.9920 (bitwise)
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(wprime_backward.STAGE_DOUT_GPUS)
    # Idea 31 step 3 (6 November 2026): SKIP_SELECT on both, MD H200 0.9876 / 0.9919, A100-SXM4 0.9843 / 0.9858
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(skip_select.SKIP_SELECT_GPUS)
    # Idea 39 (6 November 2026): ESTRIN off everywhere: A100-SXM4 0.9961 / 0.9963 alone, but FOLD_ALPHA alone is faster
    # (0.9967 / 0.9902 vs ESTRIN) and the two together gain nothing; H200 1.0030 / 1.0012
    assert "NVIDIA A100-SXM4-80GB" not in wprime_backward.ESTRIN_GPUS
    assert "NVIDIA H200" not in wprime_backward.ESTRIN_GPUS
    # Idea 34 (6 November 2026): FOLD_ALPHA on both, MD H200 0.9963 / 0.9954, A100-SXM4 0.9895 / 0.9893;
    # gates as the default
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} <= set(wprime_backward.FOLD_ALPHA_GPUS)
    # Idea 40 (6 November 2026): POWER_BACKWARD on the H200 only, MD 0.9954 / 0.9957, gate as the default; A100 a tie
    from fastmace.kernels.triton import tabled_outer_sum as tos
    assert "NVIDIA H200" in tos.POWER_BACKWARD_GPUS
    assert "NVIDIA A100-SXM4-80GB" not in tos.POWER_BACKWARD_GPUS
    # Idea 5(b) (6 November 2026): RECEIVER_SUM on the A100 (both 80GB names), MD 0.9998 / 0.9918 and 0.9990 / 0.9985,
    # gate as the default; H200 a tie (0.9992 / 1.0012)
    from fastmace.kernels.triton import receiver_sum
    assert {"NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"} <= set(receiver_sum.RECEIVER_SUM_GPUS)
    assert "NVIDIA H200" not in receiver_sum.RECEIVER_SUM_GPUS
    # Neighbour-list follow-up (6 November 2026): BINNED_SCREEN, bitwise, from BINNED_MIN_ATOMS atoms; SrTiO3
    # 16x16x16 MD H200 0.9767 / 0.9833, A100-SXM4 0.9765 / 0.9776 (10x10x10: H200 a tie, A100 1.0038 / 1.0044)
    from fastmace.neighbors import pair_search
    assert {"NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"} <= set(pair_search.BINNED_SCREEN_GPUS)
    assert 5_000 < pair_search.BINNED_MIN_ATOMS <= 20_480
    assert "NVIDIA H200" not in edge_outer_sum.BACKWARD_WALK_GPUS
    assert "NVIDIA H200" not in fused_forward.WGMMA_GPUS  # 1.024 / 1.025, pipelined 1.035 / 1.033
    from fastmace.kernels.cutlass import segment_rows
    assert "NVIDIA H200" not in segment_rows.SEGMENT_ROWS_GPUS  # 1.0006 / 1.0000: no gain
    assert "NVIDIA H200" not in fixed.DECISION_66_TF32X3_GPUS  # OMAT-0 only (TF32X3_OMAT0); MH-1 not measured
    assert "NVIDIA H200" not in wprime_backward.BUTTERFLY_GPUS  # on 5 Nov; 0.9659 / 0.9677 off at d599cc70 (6 Nov)
    assert "NVIDIA H100 80GB HBM3" not in wprime_backward.BUTTERFLY_GPUS  # exact names: the H100 is not measured
    # SPLINE_W (7 November 2026, results/spline-w-h200-20261107): MD H200 0.8789 / 0.8775, gate passes; R128 on top
    # 0.9838 / 0.9848; PARTS off (0.9938 / 0.9904, under 1 %). The A100 family from 8 November 2026 (Alex, results/
    # a100-omat0-best-20261108): all ten H200 switches 0.7471 / 0.7469, each faster in both orders, gate passes
    from fastmace.kernels.oeq_streaming import spline_forward
    assert "NVIDIA H200" in spline_forward.SPLINE_W_GPUS and "NVIDIA H200 NVL" in spline_forward.SPLINE_W_GPUS
    assert "NVIDIA H200" in spline_forward.R128_GPUS
    a100 = ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe")
    assert all(g in spline_forward.SPLINE_W_GPUS and g in spline_forward.R128_GPUS for g in a100)
    assert all(g in spline_forward.STAGE_EDGES_GPUS and g in wprime_backward.STAGE_SH_GPUS for g in a100)
    assert spline_forward.PARTS is False
    assert spline_forward.MIN_BLOCKS is None  # R128's cap stays 6; cap 5 0.9925 / 0.9937, not combined with vec4
    # SPLINE_VECTOR (7 November 2026): the spline w' table as one float4 per weight, MD H200 0.9739 / 0.9793
    assert "NVIDIA H200" in wprime_backward.SPLINE_VECTOR_GPUS
    from fastmace.kernels.triton import compose_skip, tabled_outer_sum as tos

    assert "NVIDIA H200" in compose_skip.COMPOSE_SKIP_GPUS  # 0.9698 / 0.9738, gate passes (7 November 2026)
    assert all(g in compose_skip.COMPOSE_SKIP_GPUS and g in compose_skip.COUNT_PLAN_GPUS for g in a100)
    # measured and closed on the H200 (results/h200-compose-quads-20261107): none clears 1 %
    assert not wprime_backward.STAGE_QUADS_GPUS and not tos.PACKED_COEF_GPUS
    assert not spline_forward.FACTOR_W_GPUS and not spline_forward.CURSOR32_GPUS
    assert not spline_forward.SHARED_LOCATOR_GPUS
    assert spline_forward.FACTOR_W is False and spline_forward.CURSOR32 is False
    from fastmace.rewrites import cueq_projection

    # h200-fastpath (8 November 2026): COMPOSE_FAST 0.9614 / 0.9656, CUEQ_PROJECTED 0.9902 / 0.9954 (gate passes);
    # LINEAR_DIV_SPLIT 0.9972 / 1.0025 stays off
    assert "NVIDIA H200" in compose_skip.COMPOSE_FAST_GPUS and "NVIDIA H200" in cueq_projection.CUEQ_PROJECTED_GPUS
    assert not linear_div.LINEAR_DIV_SPLIT_GPUS
    assert all(g in wprime_backward.SPLINE_VECTOR_GPUS and g in compose_skip.COMPOSE_FAST_GPUS for g in a100)
    assert all(g in cueq_projection.CUEQ_PROJECTED_GPUS for g in a100)
    # POWER_BACKWARD on the A100: alone 0.9919 / 0.9969, on top of the ten 1.0006 / 1.0028 (b = on): stays H200-only
    assert "NVIDIA A100-SXM4-80GB" not in tos.POWER_BACKWARD_GPUS
    assert wprime_backward.SPLINE_VECTOR is None and wprime_backward.spline_vector_default("cpu") == 2


def test_radial_table_warps(monkeypatch):
    """The radial table forward's num_warps: WARPS when set, else the per-GPU entry (H200: 8), else Triton's."""
    from fastmace.kernels.triton import radial_table as rt

    assert rt.WARPS is None and rt.WARPS_GPUS == {"NVIDIA H200": 8}
    monkeypatch.setattr(rt, "_device_warps", lambda index: rt.WARPS_GPUS.get("NVIDIA H200"))
    assert rt.forward_warps("cuda:0") == 8
    monkeypatch.setattr(rt, "_device_warps", lambda index: rt.WARPS_GPUS.get("NVIDIA A100-SXM4-80GB"))
    assert rt.forward_warps("cuda:0") is None
    monkeypatch.setattr(rt, "WARPS", 2)
    assert rt.forward_warps("cuda:0") == 2


def test_dynamo_constant_is_torchs_assume_constant_result():
    """``dynamo_constant`` sets the same attributes as ``torch.compiler.assume_constant_result`` (a torch upgrade
    that changes the decorator fails here) and returns the function itself."""
    import torch

    from fastmace.kernels.gpu_models import dynamo_constant

    def ours(index):
        return index

    def torchs(index):
        return index

    assert dynamo_constant(ours) is ours
    assert torch.compiler.assume_constant_result(torchs) is torchs
    assert vars(ours) == vars(torchs) == {"_dynamo_marked_constant": True}


def test_dynamo_calls_a_dynamo_constant_at_trace_time():
    """A marked function runs once while tracing and becomes a constant: under ``fullgraph=True`` its body (a
    ``print``, which Dynamo cannot trace) causes no graph break."""
    import torch

    from fastmace.kernels.gpu_models import dynamo_constant

    calls = []

    @dynamo_constant
    def listed(index: int) -> bool:
        calls.append(index)
        print("not traceable")
        return index == 0

    @torch.compile(fullgraph=True, backend="eager")
    def scaled(x):
        return x * 2 if listed(0) else x

    x = torch.ones(3)
    assert torch.equal(scaled(x), 2 * x) and torch.equal(scaled(x), 2 * x)
    assert calls == [0]
