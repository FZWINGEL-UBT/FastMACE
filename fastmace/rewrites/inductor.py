"""Inductor options of the compiled force model, as records (optimisation round 3, OPT3-T04; ``plans/opt-3.md``).

``RECORDS`` holds ``combo_kernels``: Inductor's combo kernels for the compiled model, pointwise kernels only
(``torch._inductor.config.combo_kernels`` with ``combo_kernels_pointwise_only``). Inductor then launches
independent adjacent pointwise kernels of the forward and backward as one kernel. On NaCl 3×3×3 (MH-1,
hybrid float32, local RTX 5060 Ti) the force graph has 211 kernels instead of 264 and 128 µs of gaps
instead of 157 µs (``results/opt3-T04-graph-gaps/``).

- **Rounding, not exact** (measured, ``docs/validation/opt3-T04.md``). Reductions are left out on purpose (a
  combo could tile them differently), and in the reproducible mode positions, momenta and forces stay bitwise
  equal over 200 NaCl steps; but the potential energy differs by 2.1e-6 eV (of 1474 eV), so the combined
  pointwise code is not bit-for-bit the separate kernels' on the energy path.
- **Not a default candidate:** NaCl gains about 1 % (local RTX 5060 Ti and Modal L4), NbCl5 loses about 2 %
  on both, also with ``combo_kernel_per_subkernel_blocks``.
- **How it acts.** ``apply`` marks the model with the options (``INDUCTOR_OPTIONS``).
  ``fastmace.model.compile.compile_model`` passes them to ``torch.compile(options=...)`` when it compiles that
  model. The record's name is in the force field's compile-cache key (the enabled records), so a model
  compiled with and one without the option are never shared. Nothing is patched process-wide, and an
  uncompiled plan (the CPU, ``compile_mode="none"``) is not affected.
- **Opt-in** (default off, the planner's ``never_automatic``); the owner decides defaults.
"""

from __future__ import annotations

from fastmace.rewrites.registry import CheckResult, Rewrite, Support

MACE_VERSIONS = ">=0.3.16,<0.3.17"
# The attribute compile_model reads: Inductor options for this model's torch.compile.
INDUCTOR_OPTIONS = "_fastmace_inductor_options"
COMBO_OPTIONS = {"combo_kernels": True, "combo_kernels_pointwise_only": True}


def inductor_options(model) -> dict:
    """The Inductor options a record set on ``model`` (empty when none)."""
    return dict(model.__dict__.get(INDUCTOR_OPTIONS, {}))


def set_options(model, options) -> dict:
    """Merge ``options`` into the model's Inductor options; returns them."""
    merged = {**inductor_options(model), **options}
    object.__setattr__(model, INDUCTOR_OPTIONS, merged)
    return merged


def _always(model):
    return Support(True, "an Inductor option of the compiled model; no module is changed")


def _apply_combo(model, context):
    return sorted(set_options(model, COMBO_OPTIONS))


def _combo_selfcheck(original, rewritten, context):
    """The model's modules are unchanged (only the compile option is set): nothing to compare numerically.

    How far the option moves values is measured on the GPU (``tests/gpu/test_combo_kernels_cuda.py`` and the
    reproducible-mode check in ``results/opt3-T04-graph-gaps/``)."""
    passed = inductor_options(rewritten).get("combo_kernels_pointwise_only") is True
    return CheckResult(passed, 0.0, 0.0, 0, "compile option only; pointwise combo kernels" if passed
                       else "combo kernels without the pointwise-only restriction")


RECORDS = {
    "combo_kernels": Rewrite(
        name="combo_kernels", applies_to=_always, requires=(), conflicts=(), mace_versions=MACE_VERSIONS,
        source_hashes={}, exactness="rounding", default="off", deterministic=True, workload_constraints=(),
        apply=_apply_combo, selfcheck=_combo_selfcheck),
}

__all__ = ["COMBO_OPTIONS", "INDUCTOR_OPTIONS", "RECORDS", "inductor_options", "set_options"]
