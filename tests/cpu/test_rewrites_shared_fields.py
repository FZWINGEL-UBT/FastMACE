"""Config fields shared by several rewrite records (conductor change at the 1a-T07/T08 merge).

The MH-1 and GENERIC_* variants of reassociate_first, streaming_conv and fused_folded_backward
read the same RewriteConfig field. An explicit "on" must be honoured by whichever variant fits
the model, and refused only when none does.
"""

import pytest
from test_rewrites_registry import model, registry_of, toy

from fastmace.model.capabilities import RewriteSupport
from fastmace.rewrites import registry as reg
from fastmace.rewrites.registry import RewriteRefusedError


def _variants(mh1_fits, generic_fits):
    return registry_of(
        toy("mh1_variant", config_field="reassociate_first",
            applies_to=lambda m: RewriteSupport(mh1_fits, "ok" if mh1_fits else "not the MH-1 class")),
        toy("generic_variant", config_field="reassociate_first",
            applies_to=lambda m: RewriteSupport(generic_fits, "ok" if generic_fits else "not generic")),
    )


@pytest.mark.parametrize("mh1_fits, generic_fits", [(True, False), (False, True)])
def test_explicit_on_is_honoured_by_the_variant_that_fits(mh1_fits, generic_fits):
    resolution = _variants(mh1_fits, generic_fits).resolve(
        model(), {"mh1_variant": "on", "generic_variant": "on"}, mace_version="0.3.16")
    enabled = [d.name for d in resolution.decisions if d.enabled]
    assert enabled == (["mh1_variant"] if mh1_fits else ["generic_variant"])
    other = next(d for d in resolution.decisions if not d.enabled)
    assert not other.refused
    assert "honoured by" in other.reason


def test_explicit_on_is_refused_when_no_variant_fits():
    with pytest.raises(RewriteRefusedError):
        _variants(False, False).resolve(model(), {"mh1_variant": "on", "generic_variant": "on"},
                                        mace_version="0.3.16")


def test_default_registry_order_follows_the_research_engine():
    sources = reg.RECORD_SOURCES
    assert sources.index("fastmace.rewrites.mh1") < sources.index("fastmace.rewrites.generic") \
        < sources.index("fastmace.rewrites.linear")
    names = list(reg.default_registry().names())
    assert names.index("factor_species") < names.index("generic_species_skip") < names.index("block_linear")


def test_single_graph_strain_is_a_registry_candidate():
    """T15 resolves it like every other rewrite (review of 18 October 2026)."""
    registry = reg.default_registry()
    assert "single_graph_strain" in registry.names()
    assert registry.config_field("single_graph_strain") == "single_graph_strain"


def test_a_selfcheck_without_dtype_gets_the_strictest_threshold():
    from test_rewrites_registry import toy as make_toy

    from fastmace.rewrites.registry import CheckResult, Registry

    rewrite, field = make_toy("a", selfcheck=lambda o, r, c: CheckResult(True, 1e-6, 1e-4, 1))
    registry = Registry()
    registry.register(rewrite, config_field=field)
    result = registry.selfcheck("a", model(), model())
    assert result.threshold == min(reg.THRESHOLDS.values()) and not result.passed
