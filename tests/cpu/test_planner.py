"""The fixed-plan planner (plan task 1a-T15; SPEC §9.1, §9.5, §4.4).

The plan must equal what the research engine's ``main.py`` runs: ``workflow.parse_args`` (stage 1,
before the model is loaded) followed by ``workflow.adapt_options_to_model`` (stage 2, on
``gpu_langevin.rewrite_support`` of the loaded model), at ResearchModule tag
``research-engine-baseline-20261017``.

- Models: one stand-in per label of the 19 (SPEC §12.5), built from ``LABEL_FACTS``, which were read
  from the released checkpoints with ``fastmace.model.capabilities.model_traits`` (``~/models``,
  18 October 2026; ``tests/gpu/test_planner_gpu.py`` re-reads them when ``FASTMACE_MODELS`` is set).
  The traits only look at class names and attributes, so the stand-ins give the checkpoints' verdicts.
- Expected plans: written out below for the defaults (``--backend auto``, i.e. hybrid, and
  ``--backend cueq``), with the research-engine lines that give them; with ``FASTMACE_BASELINE`` set
  (an export of the tag), also the baseline's own ``parse_args`` and ``adapt_options_to_model`` for
  every label and a list of ``main.py`` argument sets, refusals included.
- Refusals carry reasons; every ``AUTO_RULES`` rule is implemented; the T15 follow-ups of the
  conductor decisions (CPU cuEq, constraint spellings, streaming needs receiver sorting,
  ``inplace_block_linear``, NPT keywords, process constraints) each have a test.
- ``build`` on the CPU: a tiny model, plain cuEq (naive kernels) float64, eager stages, equals an
  engine assembled by hand from the same parts.
"""

import dataclasses
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import units

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import (
    DEFAULT_SKIN_A,
    EngineConfig,
    IntegratorConfig,
    NeighbourConfig,
    PolarConfig,
    PrecisionConfig,
    RewriteConfig,
)
from fastmace.engine import constraints
from fastmace.kernels.triton import cache
from fastmace.model import capabilities, labels
from fastmace.planner import fixed
from fastmace.planner.fixed import FixedPlanner, PlanRefused
from fastmace.planner.plan import Hardware, Plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GPU = Hardware("NVIDIA GeForce RTX 5060 Ti", (12, 0), 16 * 2**30)
CPU = Hardware("cpu", (0, 0), 0)
TORCH_2_14 = "2.14.1"
UNSORTED = NeighbourConfig(sort_receivers="off")

# --------------------------------------------------------------------------- the 19 labels

MH1 = "RealAgnosticResidualNonLinearInteractionBlock"
RESIDUAL = "RealAgnosticResidualInteractionBlock"
DENSITY_FIRST, DENSITY = "RealAgnosticDensityInteractionBlock", "RealAgnosticDensityResidualInteractionBlock"
PLAIN = "RealAgnosticInteractionBlock"
PER_GRAPH = (("total_charge", "graph"), ("total_spin", "graph"))

# label -> (model class, interaction classes, apply_cutoff, joint embeddings), read from the released
# checkpoints (model_traits; every node embedding is MACE's LinearNodeEmbeddingBlock).
LABEL_FACTS = {
    "mp0-medium": ("ScaleShiftMACE", (RESIDUAL, RESIDUAL), True, ()),
    "mp0-large": ("ScaleShiftMACE", (RESIDUAL, RESIDUAL), True, ()),
    "mp0b3-medium": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "mpa0-medium": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "omat0-medium": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "matpes-pbe": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "matpes-r2scan": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "mh1-omat": ("ScaleShiftMACE", (MH1, MH1), False, ()),
    "mh1-mp": ("ScaleShiftMACE", (MH1, MH1), False, ()),
    "mh1-r2scan": ("ScaleShiftMACE", (MH1, MH1), False, ()),
    "polar1-s": ("PolarMACE", (MH1,), False, ()),
    "polar1-m": ("PolarMACE", (MH1, MH1), False, ()),
    "polar1-l": ("PolarMACE", (MH1, MH1, MH1), False, ()),
    "omol0-xl": ("ScaleShiftMACE", (MH1, MH1, MH1), False, PER_GRAPH),
    "off23-small": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off23-medium": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off23-large": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off23b-medium": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off24-medium": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
}
LABELS = tuple(LABEL_FACTS)
# MH-1's interaction class with the cutoff passed: every MH-1 rewrite applies (MH-1, POLAR-1, OMOL-0).
FACTORED_LABELS = tuple(label for label, facts in LABEL_FACTS.items() if facts[1][0] == MH1)


def stand_in(label):
    """An object with the attributes model_traits reads, as the released checkpoint of ``label`` has them."""
    model_class, interactions, apply_cutoff, joint = LABEL_FACTS[label]
    model = type(model_class, (), {})()
    model.interactions = [type(name, (), {})() for name in interactions]
    model.radial_embedding = types.SimpleNamespace(apply_cutoff=apply_cutoff)
    model.node_embedding = type("LinearNodeEmbeddingBlock", (), {})()
    if joint:
        model.joint_embedding = object()
        model.embedding_specs = {name: {"per": per} for name, per in joint}
    return model


def traits(label):
    return capabilities.model_traits(stand_in(label))


def test_the_19_labels():
    assert set(LABEL_FACTS) == set(labels.LABELS)
    assert len(LABELS) == 19


# --------------------------------------------------------------------------- planner helpers


def planner(**configs):
    """A FixedPlanner on a CUDA device with torch 2.14 and an empty environment (no side effects)."""
    configs.setdefault("environ", {})
    configs.setdefault("torch_version", TORCH_2_14)
    return FixedPlanner(**configs)


def plan_for(label, backend="auto", integrator=None, atoms=None, **configs):
    engine = configs.pop("engine", EngineConfig())
    engine = dataclasses.replace(engine, backend=backend)
    return planner(engine=engine, **configs).plan(traits(label), atoms if atoms is not None else nacl(),
                                                  integrator or IntegratorConfig(), GPU, model_label=label)


def on(value):
    """A resolved Toggle or bool as bool."""
    if isinstance(value, bool):
        return value
    assert value in ("on", "off"), value
    return value == "on"


# research-engine option (args attribute) -> (Plan attribute, field)
OPTIONS = {
    "factor_species": ("rewrites", "factor_species"),
    "fold_cutoff": ("rewrites", "fold_cutoff"),
    "density_first": ("rewrites", "density_first"),
    "reassociate_first": ("rewrites", "reassociate_first"),
    "joint_radial_input": ("rewrites", "joint_radial_input"),
    "fused_folded_backward": ("rewrites", "fused_folded_backward"),
    "segmented_first_sum": ("rewrites", "segmented_first_sum"),
    "streaming_conv": ("rewrites", "streaming_conv"),
    "species_skip": ("rewrites", "species_skip"),
    "fused_radial_hidden": ("rewrites", "fused_radial_hidden"),
    "block_linear": ("rewrites", "block_linear"),
    "grouped_linear": ("rewrites", "grouped_linear"),
    "merged_residual": ("rewrites", "merged_residual"),
    "single_graph_strain": ("rewrites", "single_graph_strain"),
    "sort_receivers": ("neighbours", "sort_receivers"),
    "gpu_neighbour_list": ("neighbours", "gpu_neighbour_list"),
    "pipelined": ("engine", "pipelined"),
    "async_output": ("engine", "async_output"),
    "speculative_graph": ("engine", "speculative_graph"),
}


def plan_options(plan):
    """{research-engine option: bool} of a plan, plus backend, dtype, cueq_conv_fusion and polar_uvu."""
    out = {name: on(getattr(getattr(plan, part), field)) for name, (part, field) in OPTIONS.items()}
    out["backend"] = plan.engine.backend
    out["dtype"] = plan.precision.dtype
    # The research engine records None for the hybrid (MACE's conversion untouched); the plan says off.
    out["cueq_conv_fusion"] = on(plan.engine.cueq_conv_fusion) if plan.engine.backend == "cueq" else None
    out["polar_uvu"] = "on" if plan.rewrites.polar_uvu else "off"
    return out


def decision(plan, item):
    matches = [d for d in plan.decisions if d[0] == item]
    assert len(matches) == 1, (item, matches)
    return matches[0]


# --------------------------------------------------------------------------- written-out expectations

# What workflow.parse_args + adapt_options_to_model give with main.py's defaults (graph engine):
# - stage 1, --backend auto = hybrid (parse_args: backend = "hybrid" if engine == "gpu"): factor_species,
#   sort_receivers and block_linear follow the backend; fold_cutoff follows factor_species; density_first,
#   reassociate_first, joint_radial_input and fused_folded_backward follow fold_cutoff; segmented_first_sum
#   needs reassociate_first and sort_receivers; streaming_conv the hybrid, fold_cutoff and sort_receivers;
#   grouped_linear and merged_residual follow block_linear; species_skip, gpu_neighbour_list, pipelined,
#   async_output and single_graph_strain are on for the graph engine; fused_radial_hidden only when "on";
#   cueq_conv_fusion is on for cueq and None for the hybrid;
# - stage 2, adapt_options_to_model: an auto option still on whose rewrite_support key is unsupported is
#   switched off with the support reason (density_first, joint_radial_input, fused_radial_hidden and
#   fold_cutoff: "fold_cutoff"; factor_species; reassociate_first and segmented_first_sum:
#   "reassociate_first"; fused_folded_backward, species_skip, streaming_conv: their own keys).
ENGINE_ON = {"gpu_neighbour_list", "pipelined", "async_output", "single_graph_strain"}
MH1_REWRITES = {"factor_species", "fold_cutoff", "density_first", "reassociate_first", "joint_radial_input",
                "fused_folded_backward", "segmented_first_sum", "streaming_conv"}
LINEAR = {"block_linear", "grouped_linear", "merged_residual"}
GENERIC = {"reassociate_first", "fused_folded_backward", "segmented_first_sum", "streaming_conv", "species_skip"}
NOT_FACTORED = "interactions {} are not ('RealAgnosticResidualNonLinearInteractionBlock',)"
NOT_GENERIC = "no MACE-MP-0 or density-family interaction"
# The intended differences from main.py: FastMACE-only rewrites that are on by default. Everything else must
# still equal the research engine's plan.
FASTMACE_ONLY_DEFAULTS = {
    # Owner decision, 20 October 2026 (plans/opt-1.md, OPT-T02 stage A): auto = on with merged_residual.
    "gated_rows": "on_with_merged_residual",
    # Owner decision, 20 October 2026 (OPT-T02 stage B): auto = on with gated_rows, float32, not reproducible.
    "segment_gemm": "on_with_gated_rows_float32_unless_reproducible",
    # Owner decision, 21 October 2026 (OPT-DEF-2): OPT2-T03's tiles wherever fused_folded_backward is (MH-1 class),
    # and the OPT-T08 records where the model takes them, float32, not reproducible.
    "folded_backward_tiles": "on_with_fused_folded_backward",
    "species_tables": "on_unless_reproducible_if_model_supports",
    "fold_readout_tail": "on_with_block_linear_unless_reproducible_if_model_supports",
    "fold_product_linear": "on_with_species_tables_and_block_linear_if_model_supports",
    # Owner decision 57, 24 October 2026 (OPT4-T04): the generic MACE family only, float32, not reproducible.
    "grouped_launch":
        "on_with_block_linear_float32_unless_reproducible_for_generic_mace_family_and_mh1_on_decision_66_gpus",
    # Owner decision 58, 24 October 2026 (OPT4-T02): the qualified POLAR-1 checkpoints (S and M), hybrid float32,
    # not reproducible.
    "fx_e3nn_linear": "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints",
    "polar_graph_sums": "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints",
    # Owner decision 60, 27 October 2026 (OPT6-T02): with streaming_conv and fused_folded_backward (MH-1 class).
    "chunked_conv_backward": "on_with_streaming_conv_and_fused_folded_backward",
    # Owner decision 62, 27 October 2026 (OPT6-T05): with fold_cutoff (MH-1 class), float32, not reproducible.
    "radial_tables": "on_with_fold_cutoff_unless_reproducible",
    # Owner decision 64, 27 October 2026 (OPT8-T02): with chunked_conv_backward, MH-1 only, float32, not
    # reproducible.
    "fused_folded_forward": "on_for_mh1_float32_unless_reproducible",
}
# Where each FastMACE-only default applies (labels): the factored interaction, or the OPT-T08 records' own scope
# (docs/design/opt-T08-linear-audit.md section 3: C1 and C3 the MH-1 labels, C2 also OMOL-0).
MH1_LABELS = ("mh1-omat", "mh1-mp", "mh1-r2scan")
# The generic MACE family (decision 57): every interaction MP-0's residual or the density family's (MACE-OFF's
# plain first interaction, MH-1, POLAR-1 and OMOL-0 are other families).
GENERIC_FAMILY_LABELS = tuple(label for label, facts in LABEL_FACTS.items()
                              if facts[0] != "PolarMACE" and set(facts[1]) <= {RESIDUAL, DENSITY_FIRST, DENSITY})
APPLIES = {"gated_rows": FACTORED_LABELS, "segment_gemm": FACTORED_LABELS, "folded_backward_tiles": FACTORED_LABELS,
           "chunked_conv_backward": FACTORED_LABELS, "radial_tables": FACTORED_LABELS,
           "species_tables": MH1_LABELS, "fold_product_linear": MH1_LABELS,
           "fold_readout_tail": (*MH1_LABELS, "omol0-xl"),
           "grouped_launch": (*GENERIC_FAMILY_LABELS, *MH1_LABELS),  # MH-1: decision 66 on the RTX 5060 Ti (GPU)
           "fx_e3nn_linear": fixed.POLAR_GRAPH_LABELS, "polar_graph_sums": fixed.POLAR_GRAPH_LABELS,
           "fused_folded_forward": MH1_LABELS}


def fastmace_only_expected(name, label, backend):
    """(on, switched off by stage 2) of a FastMACE-only default for main.py's defaults on ``backend``."""
    applies = label in APPLIES[name]
    hybrid = backend == "hybrid"
    # species_tables is noted in stage 1 for any backend, so stage 2 sees it on plain cuEq too, and since owner
    # decision 63 (27 October 2026) it stays on there in float64; the others follow hybrid-only rewrites.
    reaches_stage2 = hybrid or name == "species_tables"
    return reaches_stage2 and applies, reaches_stage2 and not applies


def expected(label, backend):
    """(options on, {option: reason} switched off by stage 2, dtype) for main.py's defaults."""
    factored = label in FACTORED_LABELS
    interactions = sorted(set(LABEL_FACTS[label][1]))
    if backend == "hybrid":
        if factored:
            return MH1_REWRITES | LINEAR | ENGINE_ON | {"sort_receivers"}, {"species_skip": NOT_GENERIC}, "float32"
        reason = NOT_FACTORED.format(interactions)
        off = dict.fromkeys(("density_first", "joint_radial_input", "fold_cutoff", "factor_species"), reason)
        return GENERIC | LINEAR | ENGINE_ON | {"sort_receivers"}, off, "float32"
    if factored:
        return ENGINE_ON, {"species_skip": NOT_GENERIC}, "float64"
    return ENGINE_ON | {"species_skip"}, {}, "float64"


@pytest.mark.parametrize("backend", ["hybrid", "cueq"])
@pytest.mark.parametrize("label", LABELS)
def test_default_plans_equal_main_py(label, backend):
    polar = LABEL_FACTS[label][0] == "PolarMACE"
    if polar and backend == "cueq":
        # workflow.GPUEngine.__init__ refuses compiled plain-cuEq PolarMACE (zero forces); the planner at plan
        # time. Uncompiled (--compile-mode none) it runs.
        with pytest.raises(PlanRefused, match="zero forces"):
            plan_for(label, backend="cueq")
        plan = plan_for(label, backend="cueq", engine=EngineConfig(compile_mode="none"))
    else:
        plan = plan_for(label, backend="auto" if backend == "hybrid" else "cueq")
    want_on, want_off, dtype = expected(label, backend)
    got = plan_options(plan)
    assert {name for name in OPTIONS if got[name]} == want_on
    assert {k: v for k, v in dict(plan.auto_options_off).items() if k not in FASTMACE_ONLY_DEFAULTS} == want_off
    # The FastMACE-only defaults are on where they apply on the hybrid (float32); stage 2 drops them where the
    # model does not take them; plain cuEq (float64) keeps species_tables where it applies (decision 63).
    for name in FASTMACE_ONLY_DEFAULTS:
        want_on, want_dropped = fastmace_only_expected(name, label, backend)
        assert on(getattr(plan.rewrites, name)) == want_on, name
        assert (name in dict(plan.auto_options_off)) == want_dropped, name
    assert (got["backend"], got["dtype"]) == (backend, dtype)
    assert got["cueq_conv_fusion"] == (True if backend == "cueq" else None)
    assert got["polar_uvu"] == "on" and plan.tier == "graph"
    # The research engine's MD defaults: no auto left anywhere.
    for part in (plan.rewrites, plan.engine, plan.precision, plan.neighbours, plan.integrator):
        assert "auto" not in dataclasses.astuple(part), part
    assert plan.integrator == IntegratorConfig(fixcm_projection="mass_weighted", dof="fastmace")  # decision 37; D2
    assert plan.polar.kspace_headroom == 0.0 if polar else plan.polar is None



def test_segment_gemm_default_keeps_the_stock_path_in_float64():
    """Owner decision, 20 October 2026: float64 plans keep the stock products and join (the Triton GEMM is
    float32); an explicit 'on' is kept (its op then runs the PyTorch path)."""
    plan = plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"))
    assert plan.precision.dtype == "float64" and on(plan.rewrites.gated_rows)
    item, choice, reason = decision(plan, "RewriteConfig.segment_gemm")
    assert choice == "off" and "float64 keeps the stock products" in reason
    explicit = plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"),
                        rewrites=RewriteConfig(segment_gemm="on"))
    assert on(explicit.rewrites.segment_gemm)
    assert on(plan_for("mh1-r2scan").rewrites.segment_gemm)

def test_grouped_launch_default_is_the_generic_mace_family():
    """Owner decision 57 (OPT4-T04): auto turns grouped_launch on for the generic MACE family, and (decision 66) for
    MH-1 on the measured GPUs, in float32 plans; explicit 'on' and 'off' override it."""
    assert set(GENERIC_FAMILY_LABELS) == {"mp0-medium", "mp0-large", "mp0b3-medium", "mpa0-medium", "omat0-medium",
                                          "matpes-pbe", "matpes-r2scan"}
    plan = plan_for("omat0-medium")
    item, choice, reason = decision(plan, "RewriteConfig.grouped_launch")
    assert choice == "on" and "generic_mace_family" in reason
    # Decision 66: MH-1 on the RTX 5060 Ti (this file's GPU) gets it; on the Ada (not measured faster) it does not.
    plan = plan_for("mh1-r2scan")
    item, choice, reason = decision(plan, "RewriteConfig.grouped_launch")
    assert choice == "on" and "decision 66 (NVIDIA GeForce RTX 5060 Ti)" in reason
    ada = planner(engine=EngineConfig()).plan(traits("mh1-r2scan"), nacl(), IntegratorConfig(), ADA,
                                              model_label="mh1-r2scan")
    item, choice, reason = decision(ada, "RewriteConfig.grouped_launch")
    assert choice == "off" and "MH-1 not on a measured GPU, decision 66 (NVIDIA RTX 4000 Ada Generation)" in reason
    for label in ("polar1-s", "omol0-xl", "off23-medium"):
        plan = plan_for(label)
        item, choice, reason = decision(plan, "RewriteConfig.grouped_launch")
        assert choice == "off" and "not the generic MACE family" in reason, label
        assert "grouped_launch" in dict(plan.auto_options_off), label
        assert on(plan_for(label, rewrites=RewriteConfig(grouped_launch="on")).rewrites.grouped_launch), label
    assert not on(plan_for("omat0-medium", rewrites=RewriteConfig(grouped_launch="off")).rewrites.grouped_launch)
    float64 = plan_for("omat0-medium", precision=PrecisionConfig(dtype="float64"))
    item, choice, reason = decision(float64, "RewriteConfig.grouped_launch")
    assert choice == "off" and "float64 keeps the stock products" in reason
    assert on(plan_for("omat0-medium", precision=PrecisionConfig(dtype="float64"),
                       rewrites=RewriteConfig(grouped_launch="on")).rewrites.grouped_launch)
    assert not on(plan_for("omat0-medium", backend="cueq").rewrites.grouped_launch)  # block_linear is hybrid-only


def test_reasons_name_the_stage_2_verdict():
    plan = plan_for("mp0-medium")
    item, choice, reason = decision(plan, "RewriteConfig.factor_species")
    assert choice == "off" and "on_for_hybrid_graph_engine_if_model_supports" in reason
    assert "stage 2" in reason and NOT_FACTORED.format([RESIDUAL]) in reason
    _, choice, reason = decision(plan, "RewriteConfig.streaming_conv")
    assert choice == "on" and "stage 2" in reason
    _, choice, reason = decision(plan_for("mh1-r2scan", "cueq"), "RewriteConfig.reassociate_first")
    assert choice == "off" and "stage 1" in reason


def test_describe_lists_every_decision():
    plan = plan_for("mh1-r2scan")
    text = plan.describe(full=True)  # str(plan) is the sectioned text (5-T05, tests/cpu/test_plan_text.py)
    assert str(plan) == plan.describe() != text
    assert text.splitlines()[0].startswith("Plan revision 0, tier graph")
    for item, choice, reason in plan.decisions:
        assert f"{item} = {choice}" in text and reason in text


def test_plan_is_frozen_and_planner_has_no_module_state():
    plan = plan_for("mh1-r2scan")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.tier = "eager"
    hash(plan.decisions)
    source = (REPOSITORY_ROOT / "fastmace/planner/fixed.py").read_text(encoding="utf-8")
    # Module-level names are constants: no dicts, lists or sets assigned at module level.
    import ast

    for node in ast.parse(source).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            assert not isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)), \
                ast.unparse(node)


def test_traits_record_or_model_or_loaded_model():
    """plan() takes ModelTraits, a model or anything with ``.model`` (a LoadedModel). A bare model has no
    label-table label unless one is passed: the label-gated defaults (decisions 58 and 64) stay off without it."""
    reference = plan_for("mh1-r2scan")
    p = planner()
    for given in (stand_in("mh1-r2scan"), types.SimpleNamespace(model=stand_in("mh1-r2scan"), calculator=None)):
        assert p.plan(given, nacl(), IntegratorConfig(), GPU, model_label="mh1-r2scan").decisions == \
            reference.decisions
        unlabelled = p.plan(given, nacl(), IntegratorConfig(), GPU)
        assert unlabelled.rewrites.fused_folded_forward == "off"
        assert "not MH-1, decision 64 (label None)" in dict(unlabelled.auto_options_off)["fused_folded_forward"]


# --------------------------------------------------------------------------- AUTO_RULES


def _auto_rules():
    spec = importlib.util.spec_from_file_location("_planner_test_config", Path(__file__).parent / "test_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The plan's config parts; FallbackConfig and GateConfig (2-T01) are not resolved into a plan.
    parts = ("RewriteConfig", "EngineConfig", "PrecisionConfig", "NeighbourConfig", "PolarConfig", "IntegratorConfig")
    return {field: rule for field, rule in module.AUTO_RULES.items() if field.split(".")[0] in parts}


def test_every_auto_rule_is_implemented():
    rules = _auto_rules()
    assert dict(fixed.AUTO_RULES) == rules
    polar_npt = plan_for("polar1-s", integrator=IntegratorConfig(ensemble="npt"))
    plans = [plan_for("mh1-r2scan"), plan_for("mp0-medium", "cueq"), polar_npt]
    for field, rule in rules.items():
        for plan in plans:
            items = [d for d in plan.decisions if d[0] == field]
            assert len(items) == 1, (field, plan.decisions)
            requested = dict(plan.requested).get(field, "auto")
            want = f"auto ({rule})" if requested == "auto" else f"requested {requested!r}"
            assert want in items[0][2], (field, items[0])
        assert any(f"auto ({rule})" in d[2] for plan in plans for d in plan.decisions if d[0] == field), field


def test_backend_and_dtype_rules():
    assert planner().model_settings() == fixed.ModelSettings("hybrid", "float32", "cuda:0", None)
    cueq = planner(engine=EngineConfig(backend="cueq")).model_settings()
    assert cueq == fixed.ModelSettings("cueq", "float64", "cuda:0", True)
    off = planner(engine=EngineConfig(backend="cueq", cueq_conv_fusion="off")).model_settings()
    assert off.cueq_conv_fusion is False
    explicit = planner(engine=EngineConfig(backend="hybrid"), precision=PrecisionConfig(dtype="float64"))
    assert explicit.model_settings().dtype == "float64"


@pytest.mark.parametrize(("environ", "version", "choice"), [
    ({}, TORCH_2_14, "on"),
    ({}, "2.13.0", "off"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}, TORCH_2_14, "off"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, TORCH_2_14, "on"),
])
def test_expandable_segments_rule_and_allocator(environ, version, choice):
    environ = dict(environ)
    p = planner(environ=environ, torch_version=version)
    plan = p.plan(traits("mh1-r2scan"), nacl(), IntegratorConfig(), GPU)
    assert plan.engine.expandable_segments == choice
    before = dict(environ)
    value = p.configure_process(plan.engine)
    if choice == "on":
        assert environ["PYTORCH_CUDA_ALLOC_CONF"] == value and "expandable_segments:True" in value
    else:  # the allocator is left alone (the compiler cache variables are set: OPT4-T05b, SPEC §14)
        assert environ.get("PYTORCH_CUDA_ALLOC_CONF") == before.get("PYTORCH_CUDA_ALLOC_CONF")
        assert set(environ) - set(before) <= set(cache.VARIABLES) <= set(environ)
        assert all(environ[key] == value for key, value in before.items())


def test_expandable_segments_is_off_on_the_cpu():
    plan = planner(engine=EngineConfig(device="cpu", backend="cueq")).plan(traits("mh1-r2scan"), nacl(),
                                                                         IntegratorConfig(), CPU)
    assert plan.engine.expandable_segments == "off"
    assert "CPU" in decision(plan, "EngineConfig.expandable_segments")[2]


def test_pipelined_resolves_to_on_or_off():
    assert plan_for("mh1-r2scan").engine.pipelined == "on"
    speculative = plan_for("mh1-r2scan", engine=EngineConfig(speculative_graph=True))
    assert speculative.engine.pipelined == "off"
    assert "speculative" in decision(speculative, "EngineConfig.pipelined")[2]
    assert plan_for("mh1-r2scan", engine=EngineConfig(pipelined="off")).engine.pipelined == "off"


def test_polar_kspace_headroom_follows_the_cell():
    assert plan_for("polar1-m").polar.kspace_headroom == 0.0
    assert plan_for("polar1-m", integrator=IntegratorConfig(ensemble="npt")).polar.kspace_headroom == 0.1
    given = plan_for("polar1-m", polar=PolarConfig(kspace_headroom=0.25), integrator=IntegratorConfig(ensemble="npt"))
    assert given.polar.kspace_headroom == 0.25
    assert plan_for("mh1-r2scan").polar is None


# --------------------------------------------------------------------------- follow-ups of the decisions


def test_follow_up_plain_cueq_never_reassociates():
    """Conductor decision after T11: the reassociation probe refuses plain cuEq, so the plan leaves it off."""
    plan = plan_for("mh1-r2scan", "cueq", rewrites=RewriteConfig(factor_species="on"),
                    neighbours=NeighbourConfig(sort_receivers="on"))
    assert plan.rewrites.fold_cutoff == "on" and plan.rewrites.density_first == "on"
    assert plan.rewrites.reassociate_first == "off" and plan.rewrites.segmented_first_sum == "off"
    assert "plain cuEq" in decision(plan, "RewriteConfig.reassociate_first")[2]
    with pytest.raises(PlanRefused, match="reassociate_first.*plain cuEq"):
        plan_for("mh1-r2scan", "cueq", rewrites=RewriteConfig(factor_species="on", reassociate_first="on"))


def test_follow_up_streaming_requires_receiver_sorting():
    plan = plan_for("mh1-r2scan", neighbours=NeighbourConfig(sort_receivers="off"))
    assert plan.rewrites.streaming_conv == "off" and plan.rewrites.segmented_first_sum == "off"
    assert "sort_receivers" in decision(plan, "RewriteConfig.streaming_conv")[2]
    with pytest.raises(PlanRefused, match="streaming_conv.*sort_receivers"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(streaming_conv="on"), neighbours=UNSORTED)
    with pytest.raises(PlanRefused, match="streaming_conv.*sort_receivers"):
        plan_for("mp0-medium", rewrites=RewriteConfig(streaming_conv="on"), neighbours=UNSORTED)


def test_follow_up_records_with_sorting_constraints_are_checked_at_build():
    """build() checks the enabled records' workload constraints against the plan (both streaming variants)."""
    from fastmace.rewrites import registry as reg

    registry = reg.default_registry()
    plan = plan_for("mh1-r2scan")
    unsorted = dataclasses.replace(plan, neighbours=dataclasses.replace(plan.neighbours, sort_receivers="off"))
    for name in ("streaming_conv", "generic_streaming", "segmented_first_sum"):
        reasons = fixed.workload_problems((name,), unsorted, registry)
        assert any("sort_receivers" in reason for reason in reasons), (name, reasons)
        assert fixed.workload_problems((name,), plan, registry) == []
    cueq = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, backend="cueq"))
    assert any("hybrid" in reason for reason in fixed.workload_problems(("streaming_conv",), cueq, registry))


def test_follow_up_constraint_spellings_are_one_vocabulary():
    """generic_streaming uses MH-1's spelling (sort_receivers, per_engine_edge_counter)."""
    from fastmace.rewrites import registry as reg

    registry = reg.default_registry()
    assert registry["generic_streaming"].workload_constraints == ("sort_receivers", "per_engine_edge_counter")
    for record in registry:
        assert set(record.workload_constraints) <= set(fixed.WORKLOAD_CONSTRAINTS), record.name


def test_follow_up_inplace_block_linear_is_refused():
    with pytest.raises(PlanRefused, match="inplace_block_linear.*not carried over in 1a"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(inplace_block_linear=True))


def test_follow_up_npt_keywords_follow_baoab_settings():
    integrator = IntegratorConfig(ensemble="npt", pressure_GPa=1.5, cell_mode="full")
    plan = plan_for("mh1-r2scan", integrator=integrator)
    assert plan.integrator == dataclasses.replace(integrator, fixcm_projection="ase", dof="fastmace")  # 3-T22b, D2
    keywords = fixed.integrator_keywords(plan.integrator)
    assert keywords == {"externalstress": -1.5 * units.GPa, "disable_cell_langevin": False}
    assert fixed.integrator_keywords(IntegratorConfig()) == {}
    item = decision(plan, "integrator")
    assert item[1] == "LangevinBAOAB" and "baoab_settings" in item[2]
    assert decision(plan, "compute_stress")[1] == "on"


# --------------------------------------------------------------------------- refusals


def test_refusals_carry_reasons_and_come_together():
    atoms = nacl()
    atoms.pbc = (True, True, False)
    from ase.constraints import FixAtoms

    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(PlanRefused) as caught:
        plan_for("mh1-r2scan", atoms=atoms, rewrites=RewriteConfig(inplace_block_linear=True))
    reasons = caught.value.reasons
    assert any("3D" in r and "PBC" in r for r in reasons)
    assert any("constraint" in r and "FixAtoms" in r for r in reasons)
    assert any("inplace_block_linear" in r for r in reasons)
    assert all(isinstance(r, str) and r for r in reasons)
    assert str(caught.value).count(";") >= 2


def test_inexact_padding_is_refused_before_compilation():
    model = stand_in("mh1-r2scan")
    model.radial_embedding.apply_cutoff = True  # a NonLinear model applying its cutoff: padding is inexact
    model.interactions[0].source_embedding = object()
    record = capabilities.model_traits(model)
    assert not record.padding_is_exact
    with pytest.raises(PlanRefused, match="padding"):
        planner().plan(record, nacl(), IntegratorConfig(), GPU)


@pytest.mark.parametrize(("label", "configs", "match"), [
    # parse_args
    ("mh1-r2scan", dict(backend="cueq", rewrites=RewriteConfig(fold_cutoff="on")), "fold_cutoff.*factor_species"),
    ("mh1-r2scan", dict(rewrites=RewriteConfig(factor_species="off", density_first="on")),
     "density_first.*fold_cutoff"),
    ("mh1-r2scan", dict(neighbours=UNSORTED, rewrites=RewriteConfig(segmented_first_sum="on")),
     "segmented_first_sum.*sort_receivers"),
    ("mh1-r2scan", dict(rewrites=RewriteConfig(block_linear="off", grouped_linear="on")),
     "grouped_linear.*block_linear"),
    ("mh1-r2scan", dict(backend="cueq", rewrites=RewriteConfig(streaming_conv="on")), "streaming_conv.*hybrid"),
    ("mh1-r2scan", dict(engine=EngineConfig(cueq_conv_fusion="on")), "cueq_conv_fusion.*cueq"),
    ("mh1-r2scan", dict(engine=EngineConfig(speculative_graph=True, pipelined="on")), "pipelined.*speculative"),
    ("mh1-r2scan", dict(engine=EngineConfig(speculative_graph=True), integrator=IntegratorConfig(ensemble="npt")),
     "speculative.*NPT"),
    # adapt_options_to_model: an explicit "on" the model cannot take
    ("mp0-medium", dict(rewrites=RewriteConfig(factor_species="on")), "factor_species.*not supported by the model"),
    ("mh1-r2scan", dict(rewrites=RewriteConfig(species_skip="on")), "species_skip.*" + NOT_GENERIC),
    ("off23-small", dict(rewrites=RewriteConfig(fused_radial_hidden="on")), "fused_radial_hidden.*not supported"),
    # FastMACE's own hard rules
    ("polar1-s", dict(backend="cueq"), "PolarMACE.*zero forces"),
    ("mh1-r2scan", dict(engine=EngineConfig(device="cpu")), "hybrid.*CUDA"),
])
def test_explicit_requests_that_cannot_be_honoured_are_refused(label, configs, match):
    configs = dict(configs)
    backend = configs.pop("backend", "auto")
    if "engine" in configs:
        backend = configs["engine"].backend
    with pytest.raises(PlanRefused, match=match):
        plan_for(label, backend=backend, **configs)


def test_polar_cueq_is_allowed_uncompiled():
    plan = plan_for("polar1-s", "cueq", engine=EngineConfig(backend="cueq", compile_mode="none"))
    assert plan.engine.backend == "cueq" and plan.engine.compile_mode == "none"


# --------------------------------------------------------------------------- the baseline's own resolution


def baseline(name):
    """langevin.<name> from the export named by FASTMACE_BASELINE (research-engine-baseline-20261017)."""
    root = os.environ.get("FASTMACE_BASELINE")
    if not root:
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    directory = Path(root).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin"
    assert directory.is_dir(), f"FASTMACE_BASELINE={root} has no Experiments/Langevin_Thermostat/langevin"
    if "langevin" not in sys.modules:
        package = types.ModuleType("langevin")
        package.__path__ = [str(directory)]
        sys.modules["langevin"] = package
    return importlib.import_module(f"langevin.{name}")


# main.py flag -> (planner config, field); "on"/"off" flags of bool fields become bools.
FLAGS = {
    "--backend": ("engine", "backend"), "--dtype": ("precision", "dtype"),
    "--compile-mode": ("engine", "compile_mode"), "--ensemble": ("integrator", "ensemble"),
    "--speculative-graph": ("engine", "speculative_graph"), "--pipelined": ("engine", "pipelined"),
    "--async-output": ("engine", "async_output"), "--cueq-conv-fusion": ("engine", "cueq_conv_fusion"),
    "--sort-receivers": ("neighbours", "sort_receivers"), "--gpu-neighbour-list": ("neighbours", "gpu_neighbour_list"),
    "--polar-uvu": ("rewrites", "polar_uvu"),
    **{f"--{name.replace('_', '-')}": ("rewrites", name)
       for name in ("factor_species", "fold_cutoff", "density_first", "reassociate_first", "joint_radial_input",
                    "fused_folded_backward", "segmented_first_sum", "streaming_conv", "species_skip",
                    "fused_radial_hidden", "block_linear", "grouped_linear", "merged_residual",
                    "single_graph_strain")},
}
CLASSES = {"engine": EngineConfig, "precision": PrecisionConfig, "neighbours": NeighbourConfig,
           "rewrites": RewriteConfig, "integrator": IntegratorConfig}
MODEL_DEPENDENT = ("factor_species", "fold_cutoff", "density_first", "reassociate_first", "joint_radial_input",
                   "fused_folded_backward", "segmented_first_sum", "species_skip", "streaming_conv",
                   "fused_radial_hidden")
ARGV_CASES = [
    (), ("--backend", "cueq"), ("--ensemble", "npt"), ("--backend", "cueq", "--ensemble", "npt"),
    ("--fold-cutoff", "off"), ("--factor-species", "off"), ("--sort-receivers", "off"), ("--streaming-conv", "off"),
    ("--speculative-graph", "on"), ("--pipelined", "off", "--async-output", "off"), ("--block-linear", "off"),
    ("--grouped-linear", "off", "--merged-residual", "off"), ("--species-skip", "off"),
    ("--single-graph-strain", "off"), ("--backend", "cueq", "--cueq-conv-fusion", "off"),
    ("--backend", "hybrid", "--dtype", "float64"), ("--backend", "cueq", "--dtype", "float32"),
    ("--fused-radial-hidden", "on"), ("--reassociate-first", "off"), ("--segmented-first-sum", "off"),
    ("--density-first", "off", "--joint-radial-input", "off"), ("--fused-folded-backward", "off"),
    ("--backend", "cueq", "--block-linear", "on"), ("--backend", "cueq", "--sort-receivers", "on"),
    ("--factor-species", "on"), ("--streaming-conv", "on"), ("--species-skip", "on"), ("--reassociate-first", "on"),
    ("--polar-uvu", "off"), ("--compile-mode", "none"),
    tuple(token for name in MODEL_DEPENDENT for token in (f"--{name.replace('_', '-')}", "off")),  # generic_only_argv
]


def configs_from_argv(argv):
    values = {part: {} for part in CLASSES}
    for flag, value in zip(argv[::2], argv[1::2]):
        part, field = FLAGS[flag]
        if field in ("speculative_graph", "polar_uvu"):
            value = value == "on"
        values[part][field] = value
    return {part: CLASSES[part](**fields) for part, fields in values.items()}


def research_engine(argv, model, label):
    """parse_args + adapt_options_to_model of the baseline: (options, auto_options_off), or the refusal."""
    workflow = baseline("workflow")
    try:
        args = workflow.parse_args(["--model", label, *argv])
        off = workflow.adapt_options_to_model(args, model)
    except SystemExit as error:
        return None, str(error)
    out = {name: bool(getattr(args, name)) for name in OPTIONS}
    out.update(backend=args.backend, dtype=args.dtype, cueq_conv_fusion=args.cueq_conv_fusion,
               polar_uvu=args.polar_uvu)
    return out, off


# Cases where FastMACE deliberately differs: the research engine would switch reassociation on for plain
# cuEq, whose probe then refuses it at engine construction (conductor decision after T11); the plan leaves
# it off (auto) or refuses it (on) at plan time.
def _plain_cueq_reassociation(argv, research):
    return research is not None and research["backend"] == "cueq" and research["reassociate_first"]


# workflow.GPUEngine.__init__ refuses compiled plain-cuEq PolarMACE (zero forces); the planner at plan time.
def _refused_at_engine_construction(label, argv, research):
    compiled = "none" not in argv
    return research is not None and LABEL_FACTS[label][0] == "PolarMACE" and research["backend"] == "cueq" and compiled


@pytest.mark.parametrize("label", LABELS)
def test_plans_equal_the_research_engine(label, capsys):
    pytest.importorskip("matscipy")  # gpu_langevin imports it
    gpu_langevin = baseline("gpu_langevin")
    model = stand_in(label)
    # The stand-ins give the baseline's own verdicts too.
    assert {k: tuple(v) for k, v in gpu_langevin.rewrite_support(model).items()} == \
           {k: tuple(v) for k, v in capabilities.rewrite_support(model).items() if k not in FASTMACE_ONLY_DEFAULTS}
    for name in FASTMACE_ONLY_DEFAULTS:  # unknown to the research engine; each has its own scope
        if name in capabilities.rewrite_support(model):
            supported, _ = capabilities.rewrite_support(model)[name]
            assert supported == (label in APPLIES[name]), name
    compared = 0
    for argv in ARGV_CASES:
        research, off = research_engine(argv, model, label)
        configs = configs_from_argv(argv)
        integrator = configs.pop("integrator")
        try:
            plan = planner(**configs).plan(traits(label), nacl(), integrator, GPU, model_label=label)
        except PlanRefused as error:
            if research is not None and not (_plain_cueq_reassociation(argv, research)
                                             or _refused_at_engine_construction(label, argv, research)):
                pytest.fail(f"{label} {argv}: planner refused ({error}), the research engine ran")
            continue
        assert research is not None, f"{label} {argv}: the research engine refused ({off}), the planner did not"
        if _plain_cueq_reassociation(argv, research):
            research = dict(research, reassociate_first=False, segmented_first_sum=False)
        assert plan_options(plan) == research, (label, argv)
        got_off = dict(plan.auto_options_off)
        assert {k: v for k, v in got_off.items() if k not in FASTMACE_ONLY_DEFAULTS} == off, (label, argv)
        # The FastMACE-only rewrites, unknown to the research engine, are switched off by stage 2 only where the
        # model does not take them, with the model's own reason; grouped_launch by its family rule (decision 57).
        stage2 = dict(fixed.STAGE2_SUPPORT)
        for name in FASTMACE_ONLY_DEFAULTS:
            if name == "grouped_launch" and name in got_off:
                assert label not in APPLIES[name], (label, argv)
                assert got_off[name].startswith("not the generic MACE family, decision 57"), (label, argv)
            elif name in fixed.POLAR_GRAPH_RECORDS and name in got_off:
                assert label not in APPLIES[name], (label, argv)
                assert got_off[name].startswith("not a qualified POLAR-1 checkpoint, decision 58"), (label, argv)
            elif name in got_off:
                supported, reason = capabilities.rewrite_support(model)[stage2[name]]
                assert label not in APPLIES[name] and not supported and got_off[name] == reason, (label, argv, name)
        compared += 1
    assert compared >= len(ARGV_CASES) - 8
    capsys.readouterr()  # adapt_options_to_model prints what it switched off


def test_npt_keywords_equal_baoab_settings():
    workflow = baseline("workflow")
    for pressure, mode in ((0.0, "hydrostatic"), (1.5, "full"), (-2.0, "hydrostatic")):
        config = IntegratorConfig(ensemble="npt", pressure_GPa=pressure, cell_mode=mode)
        settings = workflow.baoab_settings(config.timestep_fs, config.friction, pressure, mode)
        keywords = fixed.integrator_keywords(config)
        assert keywords["externalstress"] == settings["externalstress"]
        assert keywords["disable_cell_langevin"] is False  # ASE's default; baoab_settings does not pass it
        assert settings["hydrostatic"] == (config.cell_mode == "hydrostatic")


# --------------------------------------------------------------------------- CPU cuEq and build


@pytest.fixture
def tiny_checkpoint(tmp_path):
    def make(family):
        model = tiny_mace_model([11, 17], **FAMILIES[family])
        path = tmp_path / f"tiny-{family}.model"
        torch.save(model, path)
        return path

    return make


@pytest.fixture
def quiet_licences(monkeypatch):
    from fastmace.model import loading

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())


def cpu_planner(**configs):
    engine = configs.pop("engine", EngineConfig(device="cpu", backend="cueq", compile_mode="none"))
    return FixedPlanner(engine=engine, environ={}, **configs)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_follow_up_cpu_cueq_uses_the_naive_kernels(tiny_checkpoint, quiet_licences):
    """loading.load(device="cpu", backend="cueq") converts inside compile.naive_cueq()."""
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import compile as compiling
    from fastmace.model import loading

    loaded = loading.load(str(tiny_checkpoint("mh1")), backend="cueq", device="cpu", trusted=True)
    assert any(type(m).__module__.startswith("cuequivariance_torch") for m in loaded.model.modules())
    assert compiling.cpu_cueq_problems(loaded.model) == []
    atoms = nacl(rattle=0.05, seed=1)
    atoms.calc = loaded.calculator
    assert np.isfinite(atoms.get_potential_energy())
    # The switch is undone afterwards.
    from cuequivariance_torch.primitives import segmented_polynomial

    assert segmented_polynomial.HAS_CUE_OPS == (importlib.util.find_spec("cuequivariance_ops_torch") is not None)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("family", "npt"), [("mh1", False), ("density", False), ("mh1", True)])
def test_build_on_the_cpu_equals_a_hand_built_engine(tiny_checkpoint, quiet_licences, family, npt):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.engine.graph import GraphEngine
    from fastmace.integrators.baoab import LangevinBAOAB
    from fastmace.integrators.langevin import Langevin
    from fastmace.model.forces import MACEForceField
    from fastmace.state import SimulationState

    p = cpu_planner()
    loaded = p.load(str(tiny_checkpoint(family)), trusted=True)
    assert (loaded.backend, loaded.dtype, loaded.device) == ("cueq", "float64", "cpu")
    atoms = nacl(rattle=0.05, seed=1)
    integrator = IntegratorConfig(ensemble="npt", pressure_GPa=0.5) if npt else IntegratorConfig()
    plan = p.plan(loaded, atoms, integrator, CPU)
    record = constraints.ProcessRecord()
    engine = p.build(plan, loaded, atoms, temperature_K=300.0, constraints=record)
    assert isinstance(engine, GraphEngine) and engine.mode == "pipelined" and record.engines == 1
    # main.py --backend cueq: no MH-1 or block-linear rewrites; the strain patch is process-wide (both ensembles);
    # species_tables applies to MH-1 in float64 since owner decision 63.
    applied = set(engine.force_field.info()["rewrites"])
    family_rewrites = {"generic_species_skip"} if family == "density" else {"species_tables"}
    assert applied == family_rewrites | {"single_graph_strain"}

    # The same parts by hand (T13's test helper), from the plan's resolved configs.
    names = engine.force_field.resolution.enabled
    from fastmace.rewrites import registry as reg

    rewrites = {name: "on" if name in names else "off" for name in reg.default_registry().names()}
    field = MACEForceField(loaded, atoms, rewrites=rewrites, engine=plan.engine, neighbours=plan.neighbours,
                           compute_stress=npt)
    state = SimulationState.from_atoms(atoms, "cpu", plan.integrator.seed, compute_stress=npt)
    dyn = (LangevinBAOAB(state, plan.integrator, temperature_K=300.0) if npt
           else Langevin(state, plan.integrator, temperature_K=300.0))
    manual = GraphEngine(plan.engine, field, dyn, state, neighbours=plan.neighbours,
                         constraints=constraints.ProcessRecord())
    for target in (engine, manual):
        target.run(6)
        target.flush()
    np.testing.assert_array_equal(engine.state.positions.numpy(), manual.state.positions.numpy())
    assert engine.potential_energy() == manual.potential_energy()
    if npt:
        np.testing.assert_array_equal(engine.state.cell.numpy(), manual.state.cell.numpy())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_build_refuses_a_model_loaded_with_other_settings(tiny_checkpoint, quiet_licences):
    p = cpu_planner()
    loaded = p.load(str(tiny_checkpoint("mh1")), trusted=True)
    plan = p.plan(loaded, nacl(), IntegratorConfig(), CPU)
    other = dataclasses.replace(plan, precision=dataclasses.replace(plan.precision, dtype="float32"))
    with pytest.raises(ValueError, match="float32"):
        p.build(other, loaded, nacl(), constraints=constraints.ProcessRecord())
    unresolved = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, pipelined="auto"))
    with pytest.raises(ValueError, match="auto"):
        p.build(unresolved, loaded, nacl(), constraints=constraints.ProcessRecord())
    with pytest.raises(ValueError, match="tier"):
        p.build(Plan(tier="eager"), loaded, nacl())


def test_phase_3_auto_fields_resolve_to_todays_values():  # 3-T01: until 3-T04, 3-T05 and 3-T11 give the rules
    plan = plan_for("mh1-r2scan")
    assert plan.neighbours.skin == DEFAULT_SKIN_A and plan.neighbours.headroom == "fixed"
    assert plan.engine.recapture_budget == plan.engine.graph_variant_cap == 64
    assert plan.precision.mc_energy == plan.precision.dtype
    decided = {item: why for item, _, why in plan.decisions}
    for item in ("NeighbourConfig.skin", "NeighbourConfig.headroom", "EngineConfig.recapture_budget",
                 "PrecisionConfig.mc_energy"):
        assert f"auto ({fixed.AUTO_RULES[item]})" in decided[item]


CSVR = IntegratorConfig(ensemble="nvt", scheme="csvr", parameters={"tau_fs": 100.0})


@pytest.mark.parametrize("integrator, name", [(IntegratorConfig(), "Langevin"),
                                              (IntegratorConfig(ensemble="nve"), "NVE"), (CSVR, "CSVR")])
def test_plan_record_names_the_scheme_that_runs(integrator, name):
    """6-T11 (phase 4 open finding 2): the record named Langevin for every fixed-cell scheme."""
    item, choice, reason = decision(plan_for("mh1-r2scan", integrator=integrator), "integrator")
    assert choice == name, (choice, reason)
    if name == "Langevin":
        assert str(reason) == f"NVT, fixcm={integrator.fixcm} (workflow.GPUEngine)"


@pytest.mark.parametrize(("ratio", "want"), [(1 / 64, "on"), (1 / 2, "off")])
def test_fp64_emulation_follows_the_measured_float64_rate(monkeypatch, ratio, want):
    """Owner decision 63 (f64-T01): float64 plans emulate float64 products where the GPU's float64 rate is below
    1/8 of its float32 rate; float32 and reproducible plans never do."""
    from fastmace.kernels.triton import fp64_rate

    monkeypatch.setattr(fixed, "_cuda_available", lambda: True)
    monkeypatch.setattr(fp64_rate, "fp64_rate_ratio", lambda device: ratio)
    float64 = PrecisionConfig(dtype="float64")
    item, choice, reason = decision(plan_for("mh1-r2scan", precision=float64), "EngineConfig.fp64_emulation")
    assert choice == want and "measured" in str(reason), reason
    assert plan_for("mh1-r2scan", precision=float64).engine.fp64_emulation == want
    assert plan_for("mh1-r2scan").engine.fp64_emulation == "off"  # float32
    off = plan_for("mh1-r2scan", precision=float64, engine=EngineConfig(fp64_emulation="off"))
    assert off.engine.fp64_emulation == "off"


def test_fp64_emulation_without_cuda_is_off():
    plan = plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"))
    if not torch.cuda.is_available():
        assert plan.engine.fp64_emulation == "off"
        assert "not available" in str(decision(plan, "EngineConfig.fp64_emulation")[2])


def test_configure_fp64_emulation_sets_a_fixed_mantissa_only_when_asked():
    from fastmace.model import compile as compiling

    environ = {}
    assert compiling.FP64_EMULATION_MANTISSA_VARIABLE not in compiling.configure_fp64_emulation(environ)
    assert compiling.FP64_EMULATION_MANTISSA_VARIABLE not in environ
    environ = {}
    values = compiling.configure_fp64_emulation(environ, mantissa_bits=56)
    variable = compiling.FP64_EMULATION_MANTISSA_VARIABLE
    assert values[variable] == "56" == environ["CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"]
    environ = {variable: "64"}
    assert compiling.configure_fp64_emulation(environ, mantissa_bits=56)[variable] == "64"
    assert compiling.FP64_EMULATION_MANTISSA_VARIABLE not in compiling.configure_fp64_emulation({}, "dynamic")
    for bad in (True, 4, 200, 56.0, "56", None):
        with pytest.raises(ValueError, match="fp64_emulation_mantissa_bits"):
            EngineConfig(fp64_emulation_mantissa_bits=bad)
    assert EngineConfig(fp64_emulation_mantissa_bits=56).fp64_emulation_mantissa_bits == 56
    assert EngineConfig().fp64_emulation_mantissa_bits == "auto"


def test_configure_fp64_emulation_keeps_the_callers_settings():
    from fastmace.model import compile as compiling

    environ = {"CUBLAS_EMULATION_STRATEGY": "performant"}
    assert compiling.configure_fp64_emulation(environ) == {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                                           "CUBLAS_EMULATION_STRATEGY": "performant"}


ADA = Hardware("NVIDIA RTX 4000 Ada Generation", (8, 9), 20 * 2**30)
A100 = Hardware("NVIDIA A100-SXM4-40GB", (8, 0), 40 * 2**30)
L4 = Hardware("NVIDIA L4", (8, 9), 24 * 2**30)  # sm_89 like the Ada card, never measured
RTX_5090 = Hardware("NVIDIA GeForce RTX 5090", (12, 0), 32 * 2**30)  # sm_120 like the RTX 5060 Ti, never measured
DECISION_65 = ("RewriteConfig.w_tables", "RewriteConfig.save_transcendentals",
               "EngineConfig.fp64_emulation_mantissa_bits")


def float64_plan(monkeypatch, hardware, ratio=1 / 64, label="mh1-r2scan", atoms=None, dtype="float64", **configs):
    """A hybrid plan on ``hardware`` whose float64 rate is ``ratio`` of float32 (decision 63's probe)."""
    from fastmace.kernels.triton import fp64_rate

    monkeypatch.setattr(fixed, "_cuda_available", lambda: True)
    monkeypatch.setattr(fp64_rate, "fp64_rate_ratio", lambda device: ratio)
    engine = dataclasses.replace(configs.pop("engine", EngineConfig()), backend="auto")
    return planner(engine=engine, precision=PrecisionConfig(dtype=dtype), **configs).plan(
        traits(label), atoms if atoms is not None else nacl(), IntegratorConfig(), hardware, model_label=label)


@pytest.mark.parametrize("hardware", [GPU, ADA], ids=lambda h: h.name)
def test_decision_65_turns_d1_d2_d3_on_on_the_measured_gpus(monkeypatch, hardware):
    """Owner decision 65 (28 October 2026): w_tables and save_transcendentals are on in float64 MH-1 plans on the GPU
    models where the A/Bs measured a speed-up. D1, the 56 fixed mantissa bits, is suspended (owner, 28 October 2026):
    with them, a graph captured during a run computed wrong energies and forces (fp64-D1 fix), so "auto" is
    cuBLAS's dynamic choice; an explicit bit count still applies."""
    assert fixed.DECISION_65_GPUS == ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")
    plan = float64_plan(monkeypatch, hardware)
    assert plan.engine.fp64_emulation == "on" and plan.engine.fp64_emulation_mantissa_bits == "dynamic"
    assert "suspended" in str(decision(plan, "EngineConfig.fp64_emulation_mantissa_bits")[2])
    assert plan.rewrites.w_tables == "on" and plan.rewrites.save_transcendentals == "on"
    assert plan.rewrites.radial_tables == "on" and plan.rewrites.chunked_conv_backward == "on"
    for item in DECISION_65:
        assert "decision 65" in str(decision(plan, item)[2]), item


@pytest.mark.parametrize(("hardware", "ratio"), [(A100, 1 / 2), (L4, 1 / 64), (RTX_5090, 1 / 64)],
                         ids=lambda value: getattr(value, "name", value))
def test_decision_65_keeps_other_gpus_off(monkeypatch, hardware, ratio):
    """The A100 measured slower; unmeasured GPUs stay off even where their compute capability matches a measured
    one. The reason names the GPU; it is not listed among the auto options switched off."""
    plan = float64_plan(monkeypatch, hardware, ratio)
    assert plan.engine.fp64_emulation == ("off" if ratio == 1 / 2 else "on")  # decision 63, unchanged
    assert plan.engine.fp64_emulation_mantissa_bits == "dynamic"
    assert plan.rewrites.w_tables == "off" and plan.rewrites.save_transcendentals == "off"
    for item in DECISION_65[:2]:
        assert f"not a measured GPU, decision 65 ({hardware.name})" in str(decision(plan, item)[2]), item
    if ratio != 1 / 2:
        assert f"not a measured GPU, decision 65 ({hardware.name})" in str(decision(plan, DECISION_65[2])[2])
    assert not {"w_tables", "save_transcendentals"} & set(dict(plan.auto_options_off))


def test_decision_65_keeps_float32_the_cpu_other_models_and_reproducible_off(monkeypatch):
    float32 = float64_plan(monkeypatch, GPU, dtype="float32")
    assert float32.rewrites.w_tables == "off" and float32.rewrites.save_transcendentals == "off"
    assert float32.engine.fp64_emulation == "off" and float32.engine.fp64_emulation_mantissa_bits == "dynamic"
    assert "float32 plan, decision 65" in str(decision(float32, "RewriteConfig.save_transcendentals")[2])
    cpu = float64_plan(monkeypatch, CPU)
    assert cpu.rewrites.w_tables == "off" and cpu.rewrites.save_transcendentals == "off"
    assert "not a measured GPU, decision 65 (cpu)" in str(decision(cpu, "RewriteConfig.save_transcendentals")[2])
    other = float64_plan(monkeypatch, GPU, label="mp0-medium")
    assert other.rewrites.w_tables == "off" and other.rewrites.save_transcendentals == "off"
    assert other.engine.fp64_emulation_mantissa_bits == "dynamic"
    assert "not MH-1, decision 65" in str(decision(other, "RewriteConfig.save_transcendentals")[2])
    # Reproducible plans (minimal planner, rule 5) keep the native route the speed-ups were not measured on.
    assert fixed._decision_65("float64", True, GPU, "mh1-r2scan") == "reproducible=True, decision 65"
    assert fixed._decision_65("float64", False, GPU, "mh1-r2scan") is None


def test_decision_65_w_tables_stay_off_above_three_species(monkeypatch):
    """Measured on two and three species; four would pass W_TABLE_MAX_BYTES (the tables grow with the square)."""
    from ase import Atoms

    four = Atoms("NaNbOCl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0.5)],
                 cell=[4.0, 4.0, 4.0], pbc=True).repeat((2, 2, 2))
    plan = float64_plan(monkeypatch, ADA, atoms=four)
    assert plan.rewrites.w_tables == "off" and plan.rewrites.save_transcendentals == "on"
    assert plan.engine.fp64_emulation_mantissa_bits == "dynamic"  # D1 suspended
    assert dict(plan.auto_options_off)["w_tables"].startswith("4 species, decision 65")


def test_decision_65_explicit_requests_override(monkeypatch):
    on = EngineConfig(fp64_emulation="on", fp64_emulation_mantissa_bits=48)
    plan = float64_plan(monkeypatch, A100, 1 / 2, engine=on,
                        rewrites=RewriteConfig(w_tables="on", save_transcendentals="on"))
    assert plan.rewrites.w_tables == "on" and plan.rewrites.save_transcendentals == "on"
    assert plan.engine.fp64_emulation_mantissa_bits == 48
    off = EngineConfig(fp64_emulation_mantissa_bits="dynamic")
    plan = float64_plan(monkeypatch, ADA, engine=off, rewrites=RewriteConfig(w_tables="off",
                                                                            save_transcendentals="off"))
    assert plan.rewrites.w_tables == "off" and plan.rewrites.save_transcendentals == "off"
    assert plan.engine.fp64_emulation == "on" and plan.engine.fp64_emulation_mantissa_bits == "dynamic"
    with pytest.raises(PlanRefused, match="w-tables|needs float64"):
        float64_plan(monkeypatch, ADA, dtype="float32", rewrites=RewriteConfig(w_tables="on"))
