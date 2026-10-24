"""``polar_graph_sums``: PolarMACE's per-graph sums of one graph without scatter and index (opt-4 OPT4-T02).

MACE 0.3.16's ``PolarMACE.forward`` takes the Fukui normalisations and the predicted total charges as
``scatter_sum(x.double(), batch, dim_size=num_graphs)[batch]`` and the barycentre as ``scatter_mean`` and
``barycenter[batch]``. With one graph each sum adds every atom into one row (``scatter_add_``, atomics on CUDA)
and the gather back is an index whose backward is a serial, sort-based ``index_put``: on POLAR-1-S NbCl5
(1,944 atoms) 2.3 ms of the 25 ms force call, in 56 kernels (``docs/validation/opt4-T02.md``).

The record sets the instance's ``graph_sums`` flag; ``fastmace.longrange.polar`` then builds the forward with
``GRAPH_SUM_EDITS`` (a source rewrite, SPEC §6.3 "Source rewrites", composable with PR 1762 and the float64
energy reduction), whose helpers take the sum over atoms and broadcast it back when there is one graph, and
keep MACE's form otherwise. FastMACE's polar path has one graph per batch (``workload_constraints``).

- **Rounding, not exact:** a reduction sums in another order than the atomics and the serial ``index_put``.
- **Deterministic:** no atomics, no index_put (MACE's form is not, on CUDA, without the reproducible mode).
- **Default** (owner decision 58): as ``fx_e3nn_linear``, on under ``auto`` for POLAR-1-S and -M in hybrid
  float32 plans that are not reproducible. On top of ``fx_e3nn_linear`` it adds no speed; it keeps those sums
  deterministic.
"""

from __future__ import annotations

NAME = "polar_graph_sums"
MACE_VERSIONS = ">=0.3.16,<0.3.17"
# No pin of its own: once fastmace.longrange.polar is installed, PolarMACE.forward is its wrapper, which the
# registry would hash. applies_to checks polar's pins (SOURCE_HASHES there, MACE's unwrapped originals) and the
# edit's site counts instead.
SOURCE_HASHES: dict = {}


def applies_to(model):
    """(supported, reason): a PolarMACE model whose forward takes ``GRAPH_SUM_EDITS``."""
    from fastmace.longrange import polar
    from fastmace.rewrites.registry import Support

    if not polar.is_polar(model):
        return Support(False, f"{type(model).__name__} is not PolarMACE")
    import inspect

    problems = polar.source_problems()
    if problems:
        return Support(False, "unqualified MACE/graph_longrange source: " + "; ".join(problems))
    try:
        original = polar._original("mace.modules.extensions:PolarMACE.forward")
        polar.graph_sums_source(inspect.getsource(original))
    except (polar.PolarUnavailableError, ImportError, AttributeError, OSError) as error:
        return Support(False, str(error))
    return Support(True, "ok")


def apply(model, context=None):
    """Set the instance's ``graph_sums`` flag (the forward is built on its next call); returns [""]."""
    from fastmace.longrange import polar

    support = applies_to(model)
    if not support.supported:
        raise ValueError(f"{NAME} does not apply: {support.reason}")
    polar.set_graph_sums(model)
    return [""]


def selfcheck(original, rewritten, context=None, rows=37):
    """SPEC §6.4 check of the three one-graph helpers against MACE's forms, values and gradients, in float64.

    The models are not evaluated: the forward edit is checked by its site counts (``graph_sums_source``).
    """
    import torch
    from mace.tools.scatter import scatter_mean, scatter_sum

    from fastmace.longrange import polar
    from fastmace.rewrites.registry import CaseErrors

    generator = torch.Generator().manual_seed(getattr(context, "seed", 0))
    src = torch.randn(rows, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    row = torch.randn(1, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(rows, 3, generator=generator, dtype=torch.float64)
    batch = torch.zeros(rows, dtype=torch.long)
    cases = (
        ("graph_sum", src, lambda: polar.graph_sum(src, batch, 1),
         lambda: scatter_sum(src=src, index=batch, dim=0, dim_size=1)[batch]),
        ("graph_mean", src, lambda: polar.graph_mean(src, batch, 1),
         lambda: scatter_mean(src=src, index=batch, dim=0, dim_size=1)),
        ("graph_gather", row, lambda: polar.graph_gather(row, batch, 1), lambda: row[batch]),
    )
    errors = CaseErrors()
    for name, leaf, ours, mace in cases:
        got, expected = ours(), mace()
        errors.add(name, expected.detach(), got.detach())
        weight = weights[:expected.shape[0], :expected.shape[1]]
        grad_got, = torch.autograd.grad((got * weight).sum(), leaf)
        grad_expected, = torch.autograd.grad((expected * weight).sum(), leaf)
        errors.add(name, grad_expected, grad_got)
    return errors.result("float64")


RECORD = {
    "name": NAME,
    "applies_to": applies_to,
    "requires": (),
    "conflicts": (),
    "mace_versions": MACE_VERSIONS,
    "source_hashes": SOURCE_HASHES,
    "exactness": "rounding",
    "default": "off",
    "workload_constraints": ("single_graph",),
    "deterministic": True,  # a reduction and a broadcast, no atomics (2-T16)
    "apply": apply,
    "selfcheck": selfcheck,
    "config_field": "RewriteConfig.polar_graph_sums",
}
