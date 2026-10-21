"""The folded last layer's input gradient tiled by the waves rule (opt-2 OPT2-T03).

- ``skinny_linear.tile_choice``: the fixed tiles unless ``tile_rule="waves"`` and the 64-row grid has 2 to 8
  waves over the SMs, then 128-row tiles with 8 warps; BLOCK_K is the same in both, so the reduction order
  is the same (bitwise results; the GPU test checks it on the kernel);
- the record ``folded_backward_tiles``: exact, opt-in (``never_automatic``), requires
  ``fused_folded_backward``, sets the waves rule on every folded radial network; on the CPU the input
  gradient is PyTorch's either way, so energy and forces are bitwise unchanged.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.kernels.triton import skinny_linear
from fastmace.rewrites import mh1

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, registry, tiny  # noqa: E402

FIXED = (skinny_linear.BLOCK_M, skinny_linear.BLOCK_K, skinny_linear.NUM_WARPS, skinny_linear.NUM_STAGES)


@pytest.mark.parametrize(("rows", "sms", "expected"), [
    (8203, 36, skinny_linear.WAVES_TILES),   # NaCl 3x3x3: 129 tiles, 3.6 waves
    (16384, 36, skinny_linear.WAVES_TILES),  # 256 tiles, 7.1 waves
    (64 * 72, 36, skinny_linear.WAVES_TILES),   # exactly 2 waves
    (64 * 288, 36, skinny_linear.WAVES_TILES),  # exactly 8 waves
    (64 * 71, 36, FIXED),                    # just below 2 waves
    (64 * 288 + 1, 36, FIXED),               # just above 8 waves
    (56250, 36, FIXED),                      # NbCl5: 24 waves
    (82688, 48, FIXED),                      # NaNbO3 on 48 SMs: 27 waves
    (8203, 48, skinny_linear.WAVES_TILES),   # NaCl on 48 SMs: 2.7 waves
    (8203, None, FIXED),                     # not a CUDA device
])
def test_the_waves_rule(rows, sms, expected):
    assert skinny_linear.tile_choice(rows, sms, "waves") == expected
    assert skinny_linear.tile_choice(rows, sms, "fixed") == FIXED
    assert skinny_linear.tile_choice(rows, sms) == FIXED
    assert expected[1] == FIXED[1]  # BLOCK_K, the reduction order, never changes


def test_an_unknown_tile_rule_is_refused():
    with pytest.raises(ValueError, match="tile_rule"):
        skinny_linear.tile_choice(100, 36, "auto")
    with pytest.raises(ValueError, match="tile_rule"):
        skinny_linear.folded_linear(torch.zeros(3, 4), torch.zeros(5, 4), tile_rule="fast")


def test_folded_linear_with_the_waves_rule_on_the_cpu_is_unchanged():
    a = torch.randn(37, 65, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1280, 65, dtype=torch.float64)
    grads = []
    for rule in ("fixed", "waves"):
        out = skinny_linear.folded_linear(a, weight, tile_rule=rule)
        (grad,) = torch.autograd.grad((out * out).sum(), a)
        grads.append((out.detach(), grad))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_the_record_is_exact_opt_in_and_needs_the_fused_backward():
    record = mh1.RECORDS["folded_backward_tiles"]
    assert (record.default, record.exactness, record.deterministic) == ("off", "exact", True)
    assert record.requires == ("fused_folded_backward",)
    names = list(mh1.RECORDS)
    assert names.index("folded_backward_tiles") == names.index("fused_folded_backward") + 1
    model, _ = tiny("mh1", backend="cueq")
    regs = registry()
    asked = {name: "on" for name in PACKAGE} | {"fused_folded_backward": "off", "folded_backward_tiles": "on"}
    assert not regs.resolve(copy.deepcopy(model), asked, strict=False).decision("folded_backward_tiles").enabled
    auto = {name: "on" for name in PACKAGE}
    assert not regs.resolve(copy.deepcopy(model), auto, strict=False).decision("folded_backward_tiles").enabled


def test_the_record_sets_the_waves_rule_and_changes_nothing_on_the_cpu():
    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    results = {}
    for tiles in ("off", "on"):
        out = copy.deepcopy(model)
        asked = {name: "on" for name in PACKAGE} | {"folded_backward_tiles": tiles}
        resolution = regs.resolve(out, asked, strict=False)
        applied = regs.apply(out, resolution, context(batch), original=model)
        assert resolution.decision("folded_backward_tiles").enabled is (tiles == "on")
        nets = [interaction.conv_tp_weights for interaction in out.interactions
                if getattr(interaction.conv_tp_weights, "fused_backward", False)]
        assert nets and all(net.folded_tiles == ("waves" if tiles == "on" else "fixed") for net in nets)
        if tiles == "on":
            entry = next(entry for entry in applied if entry.name == "folded_backward_tiles")
            assert entry.result.passed and entry.detail
        results[tiles] = energy_and_forces(out, batch)
    assert torch.equal(results["on"][0], results["off"][0]) and torch.equal(results["on"][1], results["off"][1])


def test_the_planner_turns_it_on_with_the_fused_backward():
    """Owner decision, 21 October 2026 (OPT-DEF-2): auto = on wherever fused_folded_backward is (MH-1 class)."""
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.folded_backward_tiles"] == "on_with_fused_folded_backward"
    assert fixed.REWRITE_FIELDS.index("folded_backward_tiles") == fixed.REWRITE_FIELDS.index(
        "fused_folded_backward") + 1
