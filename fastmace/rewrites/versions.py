"""MACE versions and the always-on treatments (SPEC §6.3, §18 decision 2; plan task 2-T04).

FastMACE is qualified on MACE 0.3.16 only. On another version the rewrite records follow the
versioned registry (``fastmace.rewrites.registry.qualify``: a record whose pinned source does not
hash is off with its reason). The treatments here cannot simply be switched off, so each has its
own rule, decided by the same source hashes (never by the version string):

- ``float64_energy`` (``fastmace.model.energy``): the patched forward when it hashes; otherwise
  the force precision is promoted to float64, whose forward sums the energy in float64 natively
  (``effective_precision``), with the reference tier as the last resort;
- ``polar_forward`` (``fastmace.longrange.polar``): PolarMACE is refused when its sources do not
  hash, because its stress fix (SPEC §12.1) needs the patched code (``polar_support``);
- ``capture_safe_mask_head``: without it a multi-head model cannot be captured in CUDA graphs, so
  it runs on the eager tier (``capture_support``);
- ``first_derivative_only``, ``single_graph_strain``, ``unfuse_full_addmm``: not installed, MACE's
  or torch's own function stays;
- ``cueq_conv_fusion``: not installed, MACE's own conversion rule stays.

``treatments()`` lists the decisions with their reasons, for the plan and the report.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmace.errors import Reason
from fastmace.model.capabilities import RewriteSupport

QUALIFIED_MACE_VERSIONS = ("0.3.16",)

# Treatment -> (decision when its sources hash, decision when they do not).
TREATMENTS = {
    "float64_energy": ("patched", "native float64"),
    "polar_forward": ("patched", "refuse PolarMACE"),
    "capture_safe_mask_head": ("installed", "no capture for multi-head models"),
    "first_derivative_only": ("installed", "not installed"),
    "single_graph_strain": ("installed", "not installed"),
    "unfuse_full_addmm": ("installed", "not installed"),
    "cueq_conv_fusion": ("installed", "MACE's own rule"),
}

PROMOTION_CODE = "precision-promoted-unqualified-energy"
# Planner decision codes of 2-T04b.
CAPTURE_CODE = "eager-unqualified-mask"
CONV_FUSION_CODE = "conv-fusion-mace-rule"
COMPILE_CODE = "compile-unavailable"

# What fastmace.model.compile needs from MACE (0.3.16's compile API).
COMPILE_API = {"mace.tools.compile": ("configure_autograd_for_compile", "disable_e3nn_codegen", "simplify",
                                      "prepare"),
               "mace.tools.scripts_utils": ("extract_model",)}


@dataclass(frozen=True)
class Treatment:
    """The decision on one always-on treatment for the installed sources."""

    name: str
    qualified: bool
    decision: str
    reason: str


def _problems(name):
    if name == "float64_energy":
        from fastmace.model import energy

        return energy.source_problems()
    if name == "polar_forward":
        from fastmace.longrange import polar

        # Without graph_longrange PolarMACE is refused at build ("not installed", polar.install), as
        # before; only sources that exist and differ are this rule's business.
        return [problem for problem in polar.source_problems() if "No module named 'graph_longrange" not in problem]
    from fastmace import _patches

    return _patches.patch_problems(name)


# The treatments that are patches in fastmace._patches.
_PATCHES = ("capture_safe_mask_head", "first_derivative_only", "single_graph_strain", "unfuse_full_addmm",
            "cueq_conv_fusion")

_CONSEQUENCES = {
    "float64_energy": "the force precision is promoted to float64, whose forward sums the energy in float64 "
                      "natively; the reference tier is the last resort",
    "polar_forward": "PolarMACE is refused: its stress fix (MACE issue 1642, SPEC §12.1) needs the patched code",
    "capture_safe_mask_head": "MACE's mask is kept; a multi-head model cannot be captured and runs on the eager tier",
    "first_derivative_only": "MACE's force functions are kept",
    "single_graph_strain": "MACE's strain function is kept",
    "unfuse_full_addmm": "Inductor's own addmm rule is kept",
    "cueq_conv_fusion": "MACE's own cuEquivariance conversion rule is kept",
}


def treatment(name):
    """The ``Treatment`` of ``name`` (a key of ``TREATMENTS``) for the installed sources."""
    good, bad = TREATMENTS[name]
    if name in _PATCHES:
        from fastmace import _patches

        applicable, why = _patches.patch_applicable(name)
        if not applicable:  # nothing to patch on this torch: not needed, not unqualified
            return Treatment(name, True, "not needed", why)
    problems = _problems(name)
    if not problems:
        return Treatment(name, True, good, "every pinned source hashes")
    return Treatment(name, False, bad, f"unqualified source ({'; '.join(problems)}): {_CONSEQUENCES[name]}")


def treatments():
    """Every always-on treatment's decision, in ``TREATMENTS`` order."""
    return tuple(treatment(name) for name in TREATMENTS)


def effective_precision(precision, *, polar_model, float64_energy=True):
    """``(precision, reason)``: the force precision a plan runs at, and why it differs from ``precision``.

    With ``float64_energy`` on, a non-polar model whose float64 forward does not hash runs at
    float64. SPEC §18 decision 2 (owner) sets the rule's frame: 0.3.16 is the only qualified version
    and the treatments follow 2-T04's rule. Promoting even when float32 was asked for, recorded, is
    the conductor's shared decision implementing it (``plans/phase-2.md``, "Shared decisions").
    ``reason`` is a ``Reason`` with code ``PROMOTION_CODE``, or None when nothing changes.
    """
    if precision == "float64" or not float64_energy or polar_model:
        return precision, None
    problems = _problems("float64_energy")
    if not problems:
        return precision, None
    return "float64", Reason(
        f"float64 (promoted from the requested {precision}): the float64_energy forward does not hash on this MACE "
        f"({'; '.join(problems)}), so the energy is summed by a float64 model instead (SPEC §18 decision 2, 2-T04)",
        code=PROMOTION_CODE)


def polar_support():
    """``(supported, reason)`` of PolarMACE on the installed sources."""
    problems = _problems("polar_forward")
    if not problems:
        return RewriteSupport(True, "ok")
    return RewriteSupport(False, "PolarMACE needs its hashed forward for the stress fix (MACE issue 1642, "
                                 f"SPEC §12.1): unqualified source ({'; '.join(problems)})")


def head_count(model):
    """Number of heads of a MACE model (or of anything with ``heads``)."""
    heads = getattr(model, "heads", None)
    return 1 if heads is None else len(heads)


def capture_support(model):
    """``(supported, reason)`` of CUDA graph capture for ``model`` on the installed sources.

    MACE masks the readout only for several heads, with indexed assignment that capture rejects;
    the capture-safe mask replaces it. Without a hashed mask a multi-head model runs eagerly.
    """
    if head_count(model) <= 1:
        return RewriteSupport(True, "ok")
    problems = _problems("capture_safe_mask_head")
    if not problems:
        return RewriteSupport(True, "ok")
    return RewriteSupport(False, "multi-head model without the capture-safe readout mask: eager tier "
                                 f"(unqualified source: {'; '.join(problems)})")


def compile_support():
    """``(supported, reason)`` of compilation: whether MACE has the compile API FastMACE uses (``COMPILE_API``).

    A capability check, not a hash: MACE 0.3.15 has no ``configure_autograd_for_compile`` (2-T04b).
    """
    import importlib

    missing = []
    for module_name, names in COMPILE_API.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            missing.append(f"{module_name} ({error})")
            continue
        missing += [f"{module_name}.{name}" for name in names if not hasattr(module, name)]
    if not missing:
        return RewriteSupport(True, "ok")
    return RewriteSupport(False, f"this MACE lacks the compile API FastMACE uses ({', '.join(missing)}): no "
                                 "compilation, the model runs uncompiled (SPEC §18 decision 2)")


def mace_conv_fusion(device):
    """MACE's own fusion rule for a plain cuEq conversion on ``device`` when the converter does not hash.

    ``None`` when it hashes (FastMACE's ``cueq_conv_fusion`` patch applies); else whether MACE's own
    rule fuses there (only for the device string "cuda"), which the planner then follows (2-T04b).
    """
    if not _problems("cueq_conv_fusion"):
        return None
    from fastmace import _patches

    return bool(_patches.conv_fusion_for(None, device))


__all__ = [
    "CAPTURE_CODE", "COMPILE_API", "COMPILE_CODE", "CONV_FUSION_CODE", "PROMOTION_CODE",
    "QUALIFIED_MACE_VERSIONS", "TREATMENTS", "Treatment", "capture_support", "compile_support",
    "effective_precision", "head_count", "mace_conv_fusion", "polar_support", "treatment", "treatments",
]
