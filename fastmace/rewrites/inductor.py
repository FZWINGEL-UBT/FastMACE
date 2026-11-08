"""Inductor options of the compiled force model, as records (optimisation round 3, OPT3-T04; ``plans/opt-3.md``).

``RECORDS`` holds ``combo_kernels``: Inductor's combo kernels for the compiled model, pointwise kernels only
(``torch._inductor.config.combo_kernels`` with ``combo_kernels_pointwise_only``). Inductor then launches
independent adjacent pointwise kernels of the forward and backward as one kernel. On NaCl 3×3×3 (MH-1,
hybrid float32, local RTX 5060 Ti) the force graph has 211 kernels instead of 264 and 128 µs of gaps
instead of 157 µs (``results/opt3-T04-graph-gaps/``).

- **Rounding, not exact** (measured, ``docs/validation/speed/opt3-T04.md``). Reductions are left out on purpose (a
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
    """The ``SaveTranscendentals`` class, built on first use (torch is imported late) and published as this
    module's attribute: Inductor's FX-graph cache pickles the compile options, so the class must be importable
    by name (``__getattr__`` below builds it in a fresh process)."""
    cls = globals().get("SaveTranscendentals")
    if cls is not None:
        return cls
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

    SaveTranscendentals.__module__ = __name__
    SaveTranscendentals.__qualname__ = "SaveTranscendentals"
    globals()["SaveTranscendentals"] = SaveTranscendentals
    return SaveTranscendentals


def __getattr__(name):
    if name == "SaveTranscendentals":
        return _partitioner_class()
    if name == "SmallMatmuls":
        return _small_matmuls_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


# A/B constant (off-polar-alex, 7 November 2026): a post-grad pass (``post_grad_custom_post_pass``) writes small
# matrix products as a broadcast multiply and sum, which Inductor fuses with their neighbours, instead of one GEMM
# launch each: an inner dimension up to ``SMALL_INNER`` (Inductor unrolls such sums into pointwise code), or an
# output with at most ``SMALL_OUTER`` rows or columns and an inner dimension up to ``SMALL_OUTER_MAX_INNER`` (one
# reduction kernel). POLAR-1 runs hundreds of launches under 5 us per MD step (results/off-polar-alex-20261106).
# ``compile_model`` adds the pass to every model it compiles while the constant is on. Rounding (another order of
# the sums). tests/cpu/test_small_matmuls.py.
SMALL_MATMULS = False
SMALL_INNER = 8
SMALL_OUTER = 4
SMALL_OUTER_MAX_INNER = 65536
SMALL_MATMULS_UUID = "fastmace-small-matmuls-v1"


def small_matmul_kind(a_shape, b_shape):
    """"inner", "outer" or None (a library GEMM) for an [M, K] x [K, N] product."""
    m, k = (int(d) for d in a_shape)
    n = int(b_shape[1])
    if k <= SMALL_INNER:
        return "inner"
    if min(m, n) <= SMALL_OUTER and k <= SMALL_OUTER_MAX_INNER:
        return "outer"
    return None


def small_matmul(a, b, kind=None):
    """``a @ b`` as a broadcast multiply and a sum over the inner dimension (what the pass writes)."""
    return (a[:, :, None] * b[None, :, :]).sum(dim=1)


def _shape(node):
    value = getattr(node, "meta", {}).get("val")
    if value is None or value.dim() != 2:
        return None
    try:
        return tuple(int(d) for d in value.shape)
    except TypeError:  # symbolic sizes: left to the library
        return None


_SMALL_MATMUL_PATTERNS = []  # the PatternMatcherPass, built on first use


def _small_matmul_patterns():
    if _SMALL_MATMUL_PATTERNS:
        return _SMALL_MATMUL_PATTERNS[0]
    import torch
    from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass, register_graph_pattern

    aten = torch.ops.aten
    patterns = PatternMatcherPass(pass_name="fastmace_small_matmuls")

    def kind_of(a, b):
        a_shape, b_shape = _shape(a), _shape(b)
        if a_shape is None or b_shape is None:
            return None
        return small_matmul_kind(a_shape, b_shape)

    @register_graph_pattern(CallFunction(aten.mm, Arg(), Arg()), pass_dict=patterns)
    def mm(match, a, b):
        if kind_of(a, b) is not None:
            match.replace_by_example(lambda x, y: small_matmul(x, y).to(x.dtype), [a, b])

    @register_graph_pattern(CallFunction(aten.addmm, Arg(), Arg(), Arg()), pass_dict=patterns)
    def addmm(match, bias, a, b):
        if kind_of(a, b) is not None:
            match.replace_by_example(lambda c, x, y: small_matmul(x, y).to(x.dtype) + c, [bias, a, b])

    _SMALL_MATMUL_PATTERNS.append(patterns)
    return patterns


def _small_matmuls_class():
    """The ``SmallMatmuls`` pass class, built on first use and published by name (the FX-graph cache pickles it)."""
    cls = globals().get("SmallMatmuls")
    if cls is not None:
        return cls
    from torch._inductor.custom_graph_pass import CustomGraphPass

    class SmallMatmuls(CustomGraphPass):
        """Small mm/addmm as multiply and sum (``SMALL_MATMULS``)."""

        def __call__(self, graph):
            _small_matmul_patterns().apply(graph)

        def uuid(self):
            return SMALL_MATMULS_UUID

    SmallMatmuls.__module__ = __name__
    SmallMatmuls.__qualname__ = "SmallMatmuls"
    globals()["SmallMatmuls"] = SmallMatmuls
    return SmallMatmuls


def small_matmul_pass():
    """The ``post_grad_custom_post_pass`` of ``SMALL_MATMULS``."""
    return _small_matmuls_class()()


def compile_options(options):
    """``options`` (a record's Inductor options) with the passes of the A/B constants that are on."""
    options = dict(options)
    if SMALL_MATMULS:
        if options.get("post_grad_custom_post_pass") is not None:
            raise ValueError("SMALL_MATMULS: another option already set a post-grad custom pass")
        options["post_grad_custom_post_pass"] = small_matmul_pass()
    return options


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
           "SMALL_MATMULS_UUID", "TRANSCENDENTAL_OPS", "compile_options", "inductor_options",
           "save_transcendentals_partitioner", "set_options", "small_matmul", "small_matmul_kind",
           "small_matmul_pass"]
