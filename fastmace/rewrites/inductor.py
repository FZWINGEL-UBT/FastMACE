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

``save_transcendentals`` (ideas-20261027 D3) gives the model's compile a custom AOTAutograd partitioner
(``torch._inductor.config.custom_partitioner_fn``, keyed by its uuid in the compiler caches): PyTorch's min-cut
partitioner, with exp, tanh, sigmoid, log and the other transcendental ops removed from its recomputable list for
that one call, so their forward outputs are saved for the backward instead of recomputed there. On reduced-rate
float64 GPUs those ops run on the FP64 pipe (84-89 % busy in the float64 profiles, f64-T03); there the float64
package's MD step took 0.970 (NbCl5) and 0.979 (NaCl) of its time on the RTX 5060 Ti. Exact: the same operations,
computed once in the forward instead of again in the backward. Opt-in.
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


# The ops ``save_transcendentals`` keeps out of the partitioner's recomputable list.
TRANSCENDENTAL_OPS = ("exp", "expm1", "exp2", "log", "log1p", "log2", "tanh", "sigmoid", "sin", "cos", "erf",
                      "rsqrt", "sqrt", "pow", "reciprocal", "silu")
SAVE_TRANSCENDENTALS_UUID = "fastmace-save-transcendentals-v1"
# PyTorch 2.14.1's partitioner and its op list, which save_transcendentals calls and filters.
PARTITIONER_SOURCES = {
    "torch._functorch.partitioners:get_default_op_list":
        "7f709132ee0250052e71f493fea11758a17701a09a02ef74aa85710bfda2487b",
    "torch._functorch.partitioners:min_cut_rematerialization_partition":
        "ad6fd8a246b3535b6c1048260539b87dd40eab42e146238b2b9b440ec24562ee",
}


def _partitioner_class():
    from torch._inductor.custom_graph_pass import CustomPartitionerFn

    class SaveTranscendentals(CustomPartitionerFn):
        """The min-cut partitioner without recomputation of ``TRANSCENDENTAL_OPS`` (``save_transcendentals``)."""

        def __call__(self, gm, joint_inputs, **kwargs):
            import torch
            from torch._functorch import partitioners

            original = partitioners.get_default_op_list
            banned = {getattr(torch.ops.aten, name) for name in TRANSCENDENTAL_OPS if hasattr(torch.ops.aten, name)}

            def op_list():
                op_types = original()
                for op in [op for op in op_types.recomputable_ops if op in banned]:
                    op_types.recomputable_ops.discard(op)
                return op_types

            partitioners.get_default_op_list = op_list  # for this call only, restored below
            try:
                return partitioners.min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)
            finally:
                partitioners.get_default_op_list = original

        def uuid(self):
            return SAVE_TRANSCENDENTALS_UUID

    return SaveTranscendentals


def save_transcendentals_partitioner():
    """The ``custom_partitioner_fn`` of ``save_transcendentals``."""
    return _partitioner_class()()


def _apply_save(model, context):
    if "custom_partitioner_fn" in inductor_options(model):
        from fastmace.rewrites.registry import RewriteError

        raise RewriteError("save_transcendentals: another record already set a custom partitioner")
    return sorted(set_options(model, {"custom_partitioner_fn": save_transcendentals_partitioner()}))


def _save_selfcheck(original, rewritten, context):
    """Only the compile option is set (the modules are unchanged); its exactness is measured on the GPU
    (``plans/ideas-20261027.md``, D3)."""
    option = inductor_options(rewritten).get("custom_partitioner_fn")
    passed = option is not None and option.uuid() == SAVE_TRANSCENDENTALS_UUID
    return CheckResult(passed, 0.0, 0.0, 0, "compile option only; transcendental ops saved, not recomputed"
                       if passed else "the custom partitioner is not save_transcendentals'")


RECORDS = {
    "combo_kernels": Rewrite(
        name="combo_kernels", applies_to=_always, requires=(), conflicts=(), mace_versions=MACE_VERSIONS,
        source_hashes={}, exactness="rounding", default="off", deterministic=True, workload_constraints=(),
        apply=_apply_combo, selfcheck=_combo_selfcheck),
    "save_transcendentals": Rewrite(
        name="save_transcendentals", applies_to=_always, requires=(), conflicts=(), mace_versions=MACE_VERSIONS,
        source_hashes=PARTITIONER_SOURCES, exactness="exact", default="off", deterministic=True,
        workload_constraints=(), apply=_apply_save, selfcheck=_save_selfcheck),
}

__all__ = ["COMBO_OPTIONS", "INDUCTOR_OPTIONS", "PARTITIONER_SOURCES", "RECORDS", "SAVE_TRANSCENDENTALS_UUID",
           "TRANSCENDENTAL_OPS", "inductor_options", "save_transcendentals_partitioner", "set_options"]
