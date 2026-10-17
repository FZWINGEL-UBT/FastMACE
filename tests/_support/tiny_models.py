"""Tiny randomly initialised MACE models, one per interaction family of SPEC §4.3.

They validate algorithms and plumbing (rewrite algebra, engines against MACE's own
calculator), not accuracy. Every builder seeds torch, so a family always gives the
same weights.

Building a model sets torch's default dtype to float64, as MACE's calculator does
when it is constructed; ``tests/conftest.py`` restores the default after each test.
"""

import numpy as np
import torch

# MACE 0.3.16 builds every family below; PolarMACE also needs graph_longrange
# (the ``polar`` extra), see ``polar_available``.
FAMILIES = {
    # MH-1 (all heads): the interactions receive the cutoff (apply_cutoff=False), so
    # padded edges beyond it contribute exactly zero.
    "mh1": {"interaction": "RealAgnosticResidualNonLinearInteractionBlock", "apply_cutoff": False},
    # MACE-OMOL-0: MH-1's class plus the per-graph joint embedding of total charge and spin.
    "omol": {"interaction": "RealAgnosticResidualNonLinearInteractionBlock", "apply_cutoff": False,
             "embedding_specs": {
                 "total_charge": {"type": "categorical", "per": "graph", "num_classes": 201,
                                  "offset": 100, "emb_dim": 8},
                 "total_spin": {"type": "categorical", "per": "graph", "num_classes": 101, "emb_dim": 8},
             }},
    # MACE-POLAR-1: PolarMACE on MH-1's class.
    "polar": {"interaction": "RealAgnosticResidualNonLinearInteractionBlock", "polar": True},
    # MP-0 medium/large.
    "residual": {"interaction": "RealAgnosticResidualInteractionBlock"},
    # MP-0b3, MPA-0, OMAT-0, MatPES.
    "density": {"interaction": "RealAgnosticDensityResidualInteractionBlock",
                "interaction_first": "RealAgnosticDensityInteractionBlock"},
    # MACE-OFF23/24: a non-residual first interaction, residual ones after it.
    "off": {"interaction": "RealAgnosticResidualInteractionBlock",
            "interaction_first": "RealAgnosticInteractionBlock"},
}


def polar_available():
    """True when graph_longrange is installed, which PolarMACE needs."""
    try:
        import graph_longrange  # noqa: F401
    except ImportError:
        return False
    return True


def tiny_mace_model(z_values, interaction="RealAgnosticResidualInteractionBlock", interaction_first=None,
                    polar=False, **options):
    """A tiny random ScaleShiftMACE (or PolarMACE) in float64; options go to its constructor."""
    # e3nn ships a constants file that torch>=2.6 refuses to load by default
    # (it contains `slice`); allow it for this test process only.
    torch.serialization.add_safe_globals([slice])
    from e3nn import o3
    from mace import modules
    from mace.tools import utils
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    table = utils.AtomicNumberTable(sorted(z_values))
    config = {
        "r_max": 4.0, "num_bessel": 6, "num_polynomial_cutoff": 5, "max_ell": 2,
        "interaction_cls": modules.interaction_classes[interaction],
        "interaction_cls_first": modules.interaction_classes[interaction_first or interaction],
        "num_interactions": 2, "num_elements": len(z_values), "hidden_irreps": o3.Irreps("8x0e + 8x1o"),
        "MLP_irreps": o3.Irreps("8x0e"), "gate": torch.nn.functional.silu,
        "atomic_energies": np.zeros(len(z_values)), "avg_num_neighbors": 10.0,
        "atomic_numbers": table.zs, "correlation": 2, "radial_type": "bessel",
        "atomic_inter_scale": 1.0, "atomic_inter_shift": 0.0, "pair_repulsion": True}
    config.update(options)
    model_cls = modules.PolarMACE if polar else modules.ScaleShiftMACE
    return model_cls(**config)


def tiny_mace_calculator(z_values, interaction="RealAgnosticResidualInteractionBlock", interaction_first=None,
                         device="cpu", **options):
    """A tiny random MACE calculator; options go to ScaleShiftMACE (e.g. apply_cutoff=False, as MH-1)."""
    from mace.calculators import MACECalculator
    model = tiny_mace_model(z_values, interaction, interaction_first, **options)
    return MACECalculator(models=[model], device=device, default_dtype="float64")


def tiny_family_calculator(family, z_values, device="cpu"):
    """The tiny calculator of one family in ``FAMILIES``."""
    return tiny_mace_calculator(z_values, device=device, **FAMILIES[family])
