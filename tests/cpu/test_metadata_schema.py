"""Schema-versioned provenance (plan task 1b-T10; SPEC §13, §10.6, §12.1): ``metadata.json`` version 1.

- Every §13 item maps to a key, listed here bullet by bullet; items a later phase fills are an
  explicit ``null``.
- The schema version ``fastmace.metadata/1`` is the first key.
- The merged 1b records go in as they are: T12's ``Plan`` with its revision chain, T05's
  ``ForceFieldIdentity``, T06's chunk indices at restart points, T11's setup stages, T08's trust.
- Generators are recorded as hashes; a full generator state or a noise batch anywhere is refused.
- The record round-trips through ``json``; ``write_metadata`` writes it atomically.
- The 1a blocks (``model_block``, ``fastmace_block``, ``run_record``) keep their output; the 1a model
  block nests under ``"model"`` unchanged, and gains only ``trust``.
"""

import fastmace  # noqa: F401  (before torch: it switches PyTorch's file locks, SPEC §14)

# isort: split
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_minimal_planner import plan_label

from fastmace import provenance
from fastmace.state import ForceFieldIdentity

HEX64 = re.compile(r"[0-9a-f]{64}")

# SPEC §13, bullet by bullet: (item, dotted key path in metadata.json).
SPEC_13 = (
    # model path, SHA-256, head, licence
    ("model path", "checkpoint.path"),
    ("model SHA-256", "checkpoint.sha256"),
    ("model head", "checkpoint.head"),
    ("model licence", "checkpoint.licence"),
    # the force-field identity (§10.6) and the reference record (§12.1)
    ("force-field identity", "force_field_identity"),
    ("reference record", "reference_record"),
    # the plan (tier, rewrites on/off with reasons), force and integration precision, TF32 flag,
    # backend (plain backend: convolution fused), fallbacks tried
    ("plan tier", "plan.tier"),
    ("rewrites on/off", "plan.rewrites"),
    ("rewrite reasons", "plan.decisions"),
    ("force precision", "precision.force"),
    ("integration precision", "precision.integration"),
    ("TF32 flag", "precision.tf32"),
    ("backend", "plan.backend"),
    ("convolution fused (plain backend)", "plan.conv_fusion"),
    ("fallbacks tried", "plan.fallbacks_tried"),
    # file-lock mode and whether the CUDA driver cache was disabled (§14)
    ("file-lock mode", "runtime.file_locks"),
    ("CUDA driver cache", "runtime.cuda_driver_cache"),
    # versions of all packages and kernels (OEq extension build id, Triton version), GPU name,
    # driver, compute capability
    ("package versions", "software.versions"),
    ("kernels", "software.kernels"),
    ("OEq extension build id", "software.oeq_extension"),
    ("Triton version", "software.triton"),
    ("GPU name", "hardware.gpu"),
    ("driver", "hardware.driver"),
    ("compute capability", "hardware.compute_capability"),
    # seeds, generator-state hashes and stream positions (cursors), integrator parameters, schedule
    ("seeds", "random.seed"),
    ("generator-state hashes", "random.generators"),
    ("cursors", "random.cursors"),
    ("integrator parameters", "integrator"),
    ("schedule", "schedule"),
    # gate status (§12.3) and self-check results
    ("gate status", "gate.status"),
    ("self-check results", "gate.selfchecks"),
    # re-capture counts; rebuild counts and build times per neighbour list with skins; overflows
    # redone; tier switches; MC counters; restart points with chunk indices; timing summary
    ("re-capture counts", "counters.recaptures"),
    ("neighbour-list rebuilds, build times, skins", "counters.neighbour_lists"),
    ("overflows redone", "counters.overflows_redone"),
    ("tier switches", "counters.tier_switches"),
    ("MC counters", "counters.mc"),
    ("restart points with chunk indices", "restart_points"),
    ("timing summary", "timing"),
    # code version (package version + git hash when installed from source)
    ("package version", "code.version"),
    ("git hash", "code.git_commit"),
)
# What plan task 1b-T10 adds beyond the bullets.
PLAN_1B = (
    ("schema version", "schema"),
    ("plan revision chain", "plan_revisions"),
    ("setup-time breakdown", "timing.setup_s"),
    ("run report (1b-T15)", "report"),
    ("1a model block", "model"),
    ("checkpoint trust (1b-T08)", "checkpoint.trust"),
    ("float64 energy accumulation (1b-T03)", "precision.float64_energy"),
)
# Filled by a later phase: an explicit null in 1b.
LATER_PHASES = ("counters.tier_switches", "counters.mc")

IDENTITY = ForceFieldIdentity(checkpoint_sha256="b" * 64, head="matpes_r2scan", species=(11, 17),
                              force_precision="float64", physics={"charge": None, "spin": None},
                              reference_record_id="mace-0.3.16/mh1/1", mace_version="0.3.16",
                              fastmace_version="0.1.0")
CHECKPOINT = {"path": "/models/mace-mh-1.model", "sha256": "b" * 64, "head": "matpes_r2scan", "licence": "ASL",
              "label": "mh1-r2scan", "trust": "label"}


def _get(record, path, null_parent=False):
    """The value at ``path``; with ``null_parent``, an explicit null on the way counts (no plan given)."""
    value = record
    for part in path.split("."):
        if value is None and null_parent:
            return None
        assert isinstance(value, dict) and part in value, f"{path}: no key {part!r}"
        value = value[part]
    return value


class _Report:
    def to_metadata(self):
        return {"setup_text": "FastMACE run\n", "speed": {"mean_step_ms": 4.2}}


def _full_record(**changes):
    plan = plan_label("mh1-r2scan", precision="float64")
    revised = plan.revise("edge bucket overflow", 1200)
    arguments = dict(
        model={"model": "mh1-r2scan", "fastmace": {"version": "0.1.0"}},
        checkpoint=CHECKPOINT,
        identity=IDENTITY,
        reference_record={"id": "mace-0.3.16/mh1/1", "entries": ["stock"]},
        plan=revised,
        seed=1234,
        generators={"integrator": torch.Generator().manual_seed(7)},
        cursors={"integrator.noise": 17},
        schedule={"segments": [[0, 1000, 300.0, 300.0]], "step": 0},
        gate={"status": "passed", "selfchecks": {"factor_species": 1e-7}},
        counters={"recaptures": 2, "overflows_redone": 1,
                  "neighbour_lists": {"mace": {"rebuilds": 12, "build_time_s": 0.03, "skin_A": 1.0}}},
        engine_info={"engine": "graph", "edge_bucket": 4096, "graphs": ["B", "C"]},
        restart_points=[{"step": 10000, "output_chunk": 0, "output_rows": 500, "output_frames": 500}],
        setup={"load": 24.1, "plan": 0.01, "rewrites": 3.2, "self-checks": 1.1, "model copy": 0.4,
               "first compiled call": 40.5, "captures": 2.0},
        timing={"steps": 20000, "wall_s": 95.0},
        report=_Report(),
        created_utc="2026-10-19T12:00:00+00:00",
    )
    arguments.update(changes)
    return provenance.metadata_record(**arguments)


# ------------------------------------------------------------------------ keys


def test_every_spec_13_item_maps_to_a_key():
    assert dict(provenance.SPEC_13_KEYS) == dict(SPEC_13 + PLAN_1B)
    full, empty = _full_record(), provenance.metadata_record()
    for item, path in SPEC_13 + PLAN_1B:
        _get(full, path)  # present, null or not
        _get(empty, path, null_parent=True)  # without a plan, "plan" is null


def test_schema_version_is_the_first_key():
    record = provenance.metadata_record()
    assert provenance.SCHEMA == "fastmace.metadata/1"
    assert next(iter(record)) == "schema" and record["schema"] == provenance.SCHEMA


def test_later_phase_items_are_an_explicit_null():
    for record in (provenance.metadata_record(), _full_record()):
        for path in LATER_PHASES:
            assert _get(record, path) is None, path


def test_an_empty_record_records_what_the_process_knows():
    record = provenance.metadata_record()
    assert record["precision"]["integration"] == "float64"  # SPEC §7.3: always float64
    assert record["runtime"]["file_locks"] == fastmace.runtime_info().file_locks
    assert record["code"]["version"] == fastmace.__version__
    assert record["software"]["versions"]["python"]
    # D1: the gate is opt-in; without a gate record the run says why it did not run.
    assert record["gate"] == {"status": "skipped", "reason": 'opt-in: check="gate"', "selfchecks": None}
    assert record["plan"] is None and record["plan_revisions"] == []
    assert record["restart_points"] == []


# ------------------------------------------------------------------------ the merged records


def test_plan_and_its_revision_chain():
    record = _full_record()
    plan = record["plan"]
    assert plan["revision"] == 1 and plan["tier"] == "graph" and plan["step"] == 1200
    assert plan["reason"] == "edge bucket overflow"
    assert plan["fallbacks_tried"] == []
    assert plan["backend"] == plan["engine"]["backend"]
    assert plan["conv_fusion"] == plan["engine"]["cueq_conv_fusion"]
    assert all(set(entry) == {"key", "choice", "reason", "code", "rule"} for entry in plan["decisions"])
    assert {entry["key"] for entry in plan["decisions"]} >= {"mode", "EngineConfig.backend"}
    assert "Plan revision 1" in plan["description"]
    revisions = record["plan_revisions"]
    assert [entry["revision"] for entry in revisions] == [0, 1]
    assert revisions[1]["step"] == 1200 and revisions[1]["reason"] == "edge bucket overflow"
    assert revisions[0]["description"].startswith("Plan revision 0")


def test_precision_records_float64_energy_from_the_plan():
    record = _full_record()
    assert record["precision"] == {"force": "float64", "integration": "float64", "tf32": False,
                                   "float64_energy": True}


def test_identity_and_reference_record():
    record = _full_record()
    identity = record["force_field_identity"]
    assert identity["digest"] == IDENTITY.digest
    assert {key: value for key, value in identity.items() if key != "digest"} == IDENTITY.to_dict()
    assert record["reference_record"]["id"] == "mace-0.3.16/mh1/1"


def test_reference_record_object_with_an_id():
    record = _full_record(identity=None, reference_record=SimpleNamespace(
        id="mace-0.3.16/mh1/2", to_dict=lambda: {"id": "mace-0.3.16/mh1/2"}))
    assert record["reference_record"] == {"id": "mace-0.3.16/mh1/2"}
    assert _full_record(identity=None, reference_record=SimpleNamespace(id="x/1"))["reference_record"] == {"id": "x/1"}


def test_reference_record_must_be_the_identitys():
    with pytest.raises(ValueError, match="force-field identity"):
        _full_record(reference_record={"id": "mace-0.3.16/mh1/2"})


def test_checkpoint_from_a_loaded_model_carries_the_trust():
    loaded = SimpleNamespace(provenance=lambda: {"path": "/m.model", "sha256": "c" * 64, "trust": "sha256",
                                                 "label": None, "head": None, "licence": None, "backend": "e3nn"})
    record = provenance.metadata_record(checkpoint=loaded)
    assert record["checkpoint"] == {"path": "/m.model", "sha256": "c" * 64, "head": None, "licence": None,
                                    "label": None, "trust": "sha256"}


def test_restart_points_carry_the_chunk_indices():
    record = _full_record()
    assert record["restart_points"] == [{"step": 10000, "output_chunk": 0, "output_rows": 500,
                                         "output_frames": 500}]
    with pytest.raises(ValueError, match="output_frames"):
        _full_record(restart_points=[{"step": 10, "output_chunk": 0, "output_rows": 1}])


def test_setup_stages_and_timing():
    record = _full_record()
    setup = record["timing"]["setup_s"]
    assert setup["load"] == 24.1 and setup["captures"] == 2.0
    assert record["timing"]["setup_total_s"] == pytest.approx(sum(setup.values()))
    assert record["timing"]["steps"] == 20000
    with pytest.raises(ValueError, match="load"):
        _full_record(setup={"load": -1.0})


def test_gate_report_counters_and_engine_info():
    record = _full_record()
    assert record["gate"] == {"status": "passed", "selfchecks": {"factor_species": 1e-7}}
    assert record["report"] == _Report().to_metadata()
    counters = record["counters"]
    assert counters["recaptures"] == 2 and counters["overflows_redone"] == 1
    assert counters["neighbour_lists"]["mace"]["rebuilds"] == 12
    assert counters["engine"]["edge_bucket"] == 4096
    gate = _full_record(gate={"status": "unavailable", "reason": "no reference"})["gate"]
    assert gate["selfchecks"] is None  # until 1b-T16, the self-checks are part of the gate record


def test_unknown_counter_and_gate_without_status_are_refused():
    with pytest.raises(ValueError, match="tier_switches"):
        _full_record(counters={"tier_switches": 1})  # phase 2: not settable in 1b
    with pytest.raises(ValueError, match="status"):
        _full_record(gate={"selfchecks": {}})


# ------------------------------------------------------------------------ random state


def test_generators_are_hashed_never_stored():
    cpu = torch.Generator().manual_seed(7)
    philox = torch.zeros(16, dtype=torch.uint8)  # a CUDA generator's state is 16 bytes
    host = np.random.default_rng(3)
    record = _full_record(generators={"integrator": cpu, "device": philox, "host": host, "mc": "d" * 64})
    hashes = record["random"]["generators"]
    assert set(hashes) == {"integrator", "device", "host", "mc"}
    assert all(HEX64.fullmatch(value) for value in hashes.values())
    assert hashes["integrator"] == provenance.generator_digest(cpu.get_state())
    assert hashes["integrator"] != provenance.generator_digest(torch.Generator().manual_seed(8))
    assert hashes["host"] == provenance.generator_digest(host)
    assert hashes["mc"] == "d" * 64
    assert len(json.dumps(record)) < 64_000  # a CPU generator state alone is 5,056 bytes, a batch more


@pytest.mark.parametrize("where", ["integrator", "schedule", "counters", "report", "cursors"])
def test_a_full_state_or_noise_batch_anywhere_is_refused(where):
    batch = torch.randn(64, 3, dtype=torch.float64)
    values = {"integrator": {"noise": batch}, "schedule": {"state": torch.Generator().get_state()},
              "counters": {"neighbour_lists": {"mace": {"buffer": np.zeros(100)}}},
              "report": {"blob": b"\x00" * 32}, "cursors": {"noise": batch}}
    with pytest.raises(ValueError, match="restart file"):
        _full_record(**{where: values[where]})


def test_small_arrays_become_lists():
    record = _full_record(integrator={"pressure_GPa": np.eye(3), "friction": np.float64(0.02),
                                      "steps": torch.tensor(5)})
    assert record["integrator"]["pressure_GPa"] == np.eye(3).tolist()
    assert type(record["integrator"]["friction"]) is float and record["integrator"]["steps"] == 5


def test_integrator_parameters_default_to_the_plan_integrator():
    record = _full_record()
    assert record["integrator"] == record["plan"]["integrator"]
    assert _full_record(integrator={"name": "langevin"})["integrator"]["name"] == "langevin"


# ------------------------------------------------------------------------ json and the file


def test_round_trip_through_json():
    record = _full_record()
    text = json.dumps(record, allow_nan=False)
    assert json.loads(text) == record


def test_write_metadata_is_atomic(tmp_path):
    path = tmp_path / "metadata.json"
    record = _full_record()
    assert provenance.write_metadata(path, record) == path
    assert json.loads(path.read_text()) == record
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json"]
    bad = dict(record, timing={"wall_s": float("nan")})
    with pytest.raises(ValueError):
        provenance.write_metadata(path, bad)
    assert json.loads(path.read_text()) == record  # the old file stands
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json"]


def test_write_metadata_refuses_another_schema(tmp_path):
    with pytest.raises(ValueError, match="fastmace.metadata/1"):
        provenance.write_metadata(tmp_path / "metadata.json", {"model": {}})


# ------------------------------------------------------------------------ the 1a blocks


def test_the_1a_model_block_nests_unchanged():
    block = {"model": "mh1-r2scan", "checkpoint": "/x", "fastmace": {"version": "0.1.0"}}
    assert _full_record(model=block)["model"] == block
    assert provenance.metadata_record()["model"] is None


def test_model_block_records_the_trust(loaded_model):
    from fastmace.config import EngineConfig, NeighbourConfig, PolarConfig, RewriteConfig

    off = {name: "off" for name in ("factor_species", "fold_cutoff", "density_first", "reassociate_first",
                                     "joint_radial_input", "fused_folded_backward", "segmented_first_sum",
                                     "streaming_conv", "species_skip", "fused_radial_hidden", "block_linear",
                                     "grouped_linear", "merged_residual")}
    block = provenance.model_block(loaded_model, rewrites=RewriteConfig(**off, single_graph_strain="on"),
                                   engine=EngineConfig(backend="cueq", cueq_conv_fusion="on"),
                                   neighbours=NeighbourConfig(gpu_neighbour_list="on"), polar=PolarConfig(),
                                   compute_stress=True)
    assert block["trust"] == loaded_model.provenance()["trust"] == "label"
    keys = list(block)
    assert keys.index("trust") == keys.index("checkpoint_sha256") + 1


@pytest.fixture(scope="module")
def loaded_model():
    """A LoadedModel around a tiny random MACE model (as in test_provenance.py)."""
    pytest.importorskip("mace")
    from pathlib import Path

    from _support.tiny_models import tiny_mace_calculator
    from fastmace.model.loading import LoadedModel

    return LoadedModel(
        calculator=tiny_mace_calculator([11, 17]), path=Path("/models/mace-mh-1.model"), sha256="a" * 64,
        trust="label", checkpoint_source="explicit", label="mh1-r2scan", head="matpes_r2scan",
        heads=("matpes_r2scan", "omat_pbe"), model_type="MACE", family="mh1", functional="r2scan",
        licence="ASL", backend="cueq", dtype="float64", device="cpu", cueq_conv_fusion=True,
        fused_cueq={"device": "cpu", "fused_cueq_modules": 0, "checked": False,
                    "cuequivariance_ops_torch_imports": True},
        embedding_specs=None, charge=None, spin_multiplicity=None)
