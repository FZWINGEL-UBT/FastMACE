"""Config objects (plan task 1a-T01; SPEC §3.2, §5, §6).

The defaults are checked against ``docs/design/switch-map.md``: every map row that names
a ``fastmace.config`` field gives that field's default, and every field has a row. The
``auto`` fields are checked against ``AUTO_RULES``, the resolution rules that the fixed
planner (plan task 1a-T15) implements.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import typing
from pathlib import Path

import pytest

from fastmace import config

REPO_ROOT = Path(__file__).resolve().parents[2]
SWITCH_MAP = REPO_ROOT / "docs" / "design" / "switch-map.md"

CLASSES = (
    config.RewriteConfig,
    config.EngineConfig,
    config.PrecisionConfig,
    config.NeighbourConfig,
    config.PolarConfig,
    config.IntegratorConfig,
    config.FallbackConfig,
    config.GateConfig,
    config.RunnerConfig,
    config.MCConfig,
    config.AdaptationConfig,
    config.CalibrationConfig,
)

# Every field that can be "auto", with the named rule that resolves it. 1a-T15 implements
# these rules (fastmace/planner/fixed.py); switch-map.md states each in full. Stage 1 is
# workflow.parse_args (model-independent), stage 2 workflow.adapt_options_to_model
# (gpu_langevin.rewrite_support of the loaded model).
AUTO_RULES = {
    "EngineConfig.backend": "backend_hybrid_for_graph_engine",
    "EngineConfig.pipelined": "on_with_graph_engine_unless_speculative",
    "EngineConfig.async_output": "on_with_graph_engine",
    "EngineConfig.cueq_conv_fusion": "on_for_plain_cueq_unless_reproducible",
    "EngineConfig.expandable_segments": "on_for_torch_2_14_unless_alloc_conf_set",
    "EngineConfig.fp64_emulation": "on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible",
    "EngineConfig.fp64_emulation_mantissa_bits": "dynamic_decision_65_d1_suspended_after_recapture_defect",
    "PrecisionConfig.dtype": "float32_for_hybrid_float64_for_cueq",
    "NeighbourConfig.sort_receivers": "on_for_hybrid_graph_engine",
    "NeighbourConfig.gpu_neighbour_list": "on_with_graph_engine",
    "NeighbourConfig.skin": "rule_11_skin_from_workload",
    "NeighbourConfig.headroom": "rule_11_headroom_from_workload",
    "NeighbourConfig.d3_skin": "rule_11_d3_skin_from_workload",
    "IntegratorConfig.fixcm_projection": "mass_weighted_under_nvt_fixcm_else_ase",
    "IntegratorConfig.dof": "owner_decision_d2_default",
    "EngineConfig.recapture_budget": "recapture_budget_from_workload",
    "PrecisionConfig.mc_energy": "force_precision_if_move_energy_gate_passes_else_float64",
    "RewriteConfig.factor_species": "on_for_hybrid_graph_engine_if_model_supports",
    "RewriteConfig.fold_cutoff": "on_with_factor_species_if_model_supports",
    "RewriteConfig.density_first": "on_with_fold_cutoff_if_model_supports",
    "RewriteConfig.reassociate_first": "on_with_stage1_fold_cutoff_if_model_supports",
    "RewriteConfig.joint_radial_input": "on_with_fold_cutoff_if_model_supports",
    "RewriteConfig.fused_folded_backward": "on_with_stage1_fold_cutoff_if_model_supports",
    "RewriteConfig.folded_backward_tiles": "on_with_fused_folded_backward",
    "RewriteConfig.segmented_first_sum": "on_with_reassociate_first_and_sort_receivers",
    "RewriteConfig.streaming_conv": "on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports",
    "RewriteConfig.chunked_conv_backward": "on_with_streaming_conv_and_fused_folded_backward",
    "RewriteConfig.fused_folded_forward": "on_for_mh1_float32_unless_reproducible",
    "RewriteConfig.radial_tables": "on_with_fold_cutoff_unless_reproducible",
    "RewriteConfig.w_tables": "on_for_mh1_float64_on_measured_gpus_decision_65",
    "RewriteConfig.wprime_backward": "never_automatic",
    "RewriteConfig.species_skip": "on_with_graph_engine_if_model_supports",
    "RewriteConfig.fused_radial_hidden": "never_automatic",
    "RewriteConfig.direct_node_feats": "never_automatic",
    "RewriteConfig.folded_constants": "never_automatic",
    "RewriteConfig.block_linear": "on_for_hybrid_graph_engine",
    "RewriteConfig.grouped_linear": "on_with_block_linear",
    "RewriteConfig.merged_residual": "on_with_block_linear",
    "RewriteConfig.gated_rows": "on_with_merged_residual",
    "RewriteConfig.segment_gemm": "on_with_gated_rows_float32_unless_reproducible",
    "RewriteConfig.grouped_launch":
        "on_with_block_linear_float32_unless_reproducible_for_generic_mace_family_and_mh1_on_decision_66_gpus",
    "RewriteConfig.combo_kernels": "never_automatic",
    "RewriteConfig.save_transcendentals": "on_for_mh1_float64_on_measured_gpus_decision_65",
    "RewriteConfig.tf32x3_gemms": "never_automatic",
    "RewriteConfig.fx_e3nn_linear": "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints",
    "RewriteConfig.polar_graph_sums": "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints",
    "RewriteConfig.species_tables": "on_unless_reproducible_if_model_supports",
    "RewriteConfig.fold_readout_tail": "on_with_block_linear_unless_reproducible_if_model_supports",
    "RewriteConfig.fold_product_linear": "on_with_species_tables_and_block_linear_if_model_supports",
    "RewriteConfig.species_residual": "never_automatic",
    "RewriteConfig.single_graph_strain": "on",
    "PolarConfig.kspace_headroom": "variable_cell_0_1_else_0",
    "GateConfig.reduced_cell": "largest_smaller_supercell_at_least_min_height",
    "FallbackConfig.fallback": "fallback_search_on_failure",
    "AdaptationConfig.adapt": "adapt_between_segments",
}

# Research-engine switches that SPEC §6.2 or the research engine's own records reject;
# they must not come across as fields under any spelling.
NOT_CARRIED_OVER = (
    "inductor_pdl",
    "inductor_combo",
    "decompose_small_mm",
    "decompose_small_mm_limit",
    "contiguous_folded_weight",
    "streaming_mul_ir",
    "streaming_receiver_backward",
    "joint_fused_backward",
    "activation_memory_budget",
    "preferred_blas",
    "spin_restricted",
    "generic_density_first",
)

CODE_SPAN = re.compile(r"`([^`]*)`")


def _fields():
    return {f"{cls.__name__}.{field.name}": (cls, field) for cls in CLASSES for field in dataclasses.fields(cls)}


def _default(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    assert field.default_factory is not dataclasses.MISSING, f"{field.name} has no default"
    return field.default_factory()


def _map_rows():
    """{"Class.field": [default, ...]} from the switch map's tables (column 2 field, column 3 default)."""
    rows = {}
    for line in SWITCH_MAP.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        target = CODE_SPAN.findall(cells[1])
        if len(target) != 1 or not re.fullmatch(r"[A-Za-z]+Config\.[A-Za-z0-9_]+", target[0]):
            continue
        default = CODE_SPAN.findall(cells[2])
        assert default, f"switch map row for {target[0]} has no default code span: {line}"
        rows.setdefault(target[0], []).append(ast.literal_eval(default[0]))
    return rows


def _admits_auto(cls, name):
    hint = typing.get_type_hints(cls)[name]
    stack = [hint]
    while stack:
        current = stack.pop()
        if typing.get_origin(current) is typing.Literal and "auto" in typing.get_args(current):
            return True
        stack.extend(typing.get_args(current))
    return False


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_frozen(cls):
    instance = cls()
    name = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, getattr(instance, name))
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_hashable_and_equal_by_value(cls):
    assert cls() == cls() and hash(cls()) == hash(cls())


def test_switch_map_exists():
    assert SWITCH_MAP.is_file()


def test_every_field_is_in_the_map():
    rows = _map_rows()
    missing = sorted(set(_fields()) - set(rows))
    assert not missing, f"fields without a switch-map row: {missing}"


def test_map_names_only_existing_fields():
    unknown = sorted(set(_map_rows()) - set(_fields()))
    assert not unknown, f"switch-map rows name fields that config.py lacks: {unknown}"


def test_defaults_equal_the_map():
    fields = _fields()
    for name, defaults in _map_rows().items():
        _, field = fields[name]
        for value in defaults:
            assert _default(field) == value, f"{name}: config default {_default(field)!r}, map {value!r}"


def test_every_auto_field_has_a_rule():
    auto_fields = {name for name, (cls, field) in _fields().items() if _admits_auto(cls, field.name)}
    assert auto_fields == set(AUTO_RULES), (
        f"auto fields without a rule: {sorted(auto_fields - set(AUTO_RULES))}; "
        f"rules for fields that cannot be auto: {sorted(set(AUTO_RULES) - auto_fields)}")


def test_auto_rules_are_named_in_the_map():
    text = SWITCH_MAP.read_text(encoding="utf-8")
    for field, rule in AUTO_RULES.items():
        assert f"`{rule}`" in text, f"{field}: rule {rule} is not stated in the switch map"


def test_rejected_switches_have_no_field():
    names = {field.name for cls in CLASSES for field in dataclasses.fields(cls)}
    assert not names & set(NOT_CARRIED_OVER)
    text = SWITCH_MAP.read_text(encoding="utf-8")
    for name in NOT_CARRIED_OVER:
        pattern = re.compile(rf"`(?:\w+\.)?{name.upper()}`")
        assert pattern.search(text), f"{name.upper()} is not recorded in the switch map"


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (config.RewriteConfig, {"factor_species": "yes"}),
        (config.EngineConfig, {"backend": "e3nn"}),
        (config.PrecisionConfig, {"dtype": "float16"}),
        (config.NeighbourConfig, {"skin": 0.0}),
        (config.NeighbourConfig, {"edge_multiple": 0}),
        (config.PolarConfig, {"fukui_regularisation": 1.0}),
        (config.PolarConfig, {"dipole_cap": (3.0, 2.0)}),
        (config.PolarConfig, {"kspace_headroom": -0.1}),
        (config.IntegratorConfig, {"timestep_fs": 0.0}),
        (config.IntegratorConfig, {"ensemble": "nph"}),  # "nve" is valid since 4-T01
        (config.IntegratorConfig, {"dof": "3n"}),
        (config.IntegratorConfig, {"barostat_tau_fs": -1.0}),
        (config.AdaptationConfig, {"adapt": "on"}),  # 5-T01
        (config.AdaptationConfig, {"trigger": "rebuild-rate"}),
        (config.AdaptationConfig, {"recapture_time_fraction": 1.0}),
        (config.AdaptationConfig, {"max_switches": -1}),
        (config.AdaptationConfig, {"max_switches": True}),
        (config.AdaptationConfig, {"switch_back_after": 0}),
        (config.AdaptationConfig, {"reproducible": "yes"}),
        (config.CalibrationConfig, {"calibrate": 1}),
        (config.CalibrationConfig, {"steps": 0}),
        (config.CalibrationConfig, {"margin": 0.0}),
        (config.CalibrationConfig, {"choose": "auto"}),
        (config.CalibrationConfig, {"choose": config.PENDING}),
    ],
)
def test_invalid_values_are_refused(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_replace_keeps_validation():
    rewrites = dataclasses.replace(config.RewriteConfig(), factor_species="off")
    assert rewrites.factor_species == "off"
    with pytest.raises(ValueError):
        dataclasses.replace(rewrites, block_linear=True)


def test_dipole_cap_is_a_tuple():
    assert config.PolarConfig(dipole_cap=[2.0, 3.0]).dipole_cap == (2.0, 3.0)
