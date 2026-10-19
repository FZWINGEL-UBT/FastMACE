"""Model loading (fastmace/model/loading.py, labels.py, fastmace/_patches.py; plan task 1a-T05), on the CPU.

Tiny random models stand in for checkpoints; nothing is downloaded and no GPU is used.
The naive-cuEq guard tests are ported from the research engine's
tests/core/test_upstream_mace.py.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model
from fastmace import _patches
from fastmace.model import labels as table
from fastmace.model import loading

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# SPEC §12.5: MH-1 (three heads), MP-0 medium/large, MP-0b3, MPA-0, OMAT-0, MatPES-PBE/r2SCAN,
# POLAR-1-S/M/L, OMOL-0 XL, MACE-OFF23 S/M/L, OFF23b M, OFF24 M.
ACCEPTANCE_LABELS = {
    "mh1-r2scan", "mh1-omat", "mh1-mp", "mp0-medium", "mp0-large", "mp0b3-medium", "mpa0-medium", "omat0-medium",
    "matpes-pbe", "matpes-r2scan", "polar1-s", "polar1-m", "polar1-l", "omol0-xl",
    "off23-small", "off23-medium", "off23-large", "off23b-medium", "off24-medium",
}
MIT_LABELS = {"mp0-medium", "mp0-large", "mp0b3-medium", "mpa0-medium"}  # SPEC §4.3; the rest is ASL


# ------------------------------------------------------------------------ label table


def test_label_table_has_the_19_acceptance_labels():
    assert len(ACCEPTANCE_LABELS) == 19
    assert set(table.LABELS) == ACCEPTANCE_LABELS
    assert set(table.BENCHMARK_LABELS) <= ACCEPTANCE_LABELS and len(table.BENCHMARK_LABELS) == 10
    assert set(table.MACE_OFF_LABELS) == {label for label in ACCEPTANCE_LABELS if label.startswith("off")}


@pytest.mark.parametrize("label", sorted(ACCEPTANCE_LABELS))
def test_every_label_is_complete(label):
    entry = table.LABELS[label]
    checkpoint = entry.checkpoint
    assert entry.label == label
    assert checkpoint.url.startswith("https://") and "ACEsuit" in checkpoint.url
    assert checkpoint.file_name.endswith(".model")
    assert re.fullmatch(r"[0-9a-f]{64}", checkpoint.sha256)
    assert entry.licence == (table.MIT if label in MIT_LABELS else table.ASL)
    assert entry.licence in table.LICENCE_TEXT
    assert entry.model_type == ("PolarMACE" if label.startswith("polar") else "MACE")
    assert entry.family in table.FAMILIES and entry.family in FAMILIES
    if label.startswith("polar"):
        assert entry.functional is None  # not stated by the release; the research engine records None
    else:
        assert entry.functional in ("pbe", "r2scan", "wb97m-v", "wb97m-d3bj")
    if entry.head is not None:
        assert entry.head in checkpoint.heads
    assert (checkpoint.embedding_specs is not None) == (label == "omol0-xl")
    assert "/mnt/homes" not in checkpoint.url


def test_labels_match_the_research_engine_heads_and_functionals():
    # common.MODELS at research-engine-baseline-20261017: label -> (name, head, functional).
    expected = {"mh1-omat": ("mh-1", "omat_pbe", "pbe"), "mh1-mp": ("mh-1", "mp_pbe_refit_add", "pbe"),
                "mh1-r2scan": ("mh-1", "matpes_r2scan", "r2scan"), "omol0-xl": ("omol-0-xl", "omol", "wb97m-v"),
                "matpes-r2scan": ("mace-matpes-r2scan-0", None, "r2scan"), "mp0-large": ("large", None, "pbe"),
                "polar1-m": ("polar-1-m", None, None), "off24-medium": ("off24-medium", None, "wb97m-d3bj")}
    for label, (name, head, functional) in expected.items():
        entry = table.LABELS[label]
        assert (entry.checkpoint.name, entry.head, entry.functional) == (name, head, functional)
    files = [c.file_name for c in table.CHECKPOINTS.values()]
    assert len(files) == len(set(files)) == 17


def test_find_labels_aliases_and_heads():
    assert table.find("mh1", head="matpes_r2scan") is table.LABELS["mh1-r2scan"]
    assert table.find("mh-1", head="omat_pbe") is table.LABELS["mh1-omat"]
    assert table.find("mh1-r2scan") is table.LABELS["mh1-r2scan"]
    assert table.find("mh1-r2scan", head="matpes_r2scan") is table.LABELS["mh1-r2scan"]
    assert table.find("omol0") is table.LABELS["omol0-xl"]
    assert table.find("medium") is table.LABELS["mp0-medium"]
    spice = table.find("mh1", head="spice_wB97M")  # a head without its own label
    assert (spice.label, spice.head, spice.functional, spice.licence) == ("mh-1:spice_wB97M", "spice_wB97M",
                                                                         "wb97m-d3bj", table.ASL)
    assert table.find("/some/file.model") is None
    with pytest.raises(ValueError, match="matpes_r2scan"):
        table.find("mh1")  # several heads: the error lists them
    with pytest.raises(ValueError, match="no head"):
        table.find("mh1", head="nonsense")
    with pytest.raises(ValueError, match="is head"):
        table.find("mh1-r2scan", head="omat_pbe")


# ------------------------------------------------------------------------ resolution


def write_fake(path, content=b"model"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_resolution_order(tmp_path):
    name = table.LABELS["mh1-r2scan"].checkpoint.file_name
    models, cache = tmp_path / "models", tmp_path / "cache"
    environ = {"FASTMACE_CACHE": str(cache), "FASTMACE_MODELS": str(models), "HOME": str(tmp_path)}
    cached = write_fake(cache / "models" / name)
    # Only the cache directory has it.
    resolved = loading.resolve_checkpoint("mh1-r2scan", environ=environ)
    assert (resolved.path, resolved.source) == (cached.resolve(), "cache")
    assert resolved.searched == (str(models / name), str(cache / "models" / name))
    # FASTMACE_MODELS wins over the cache.
    stored = write_fake(models / name)
    resolved = loading.resolve_checkpoint("mh1", head="matpes_r2scan", environ=environ)
    assert (resolved.path, resolved.source, resolved.label.label) == (stored.resolve(), "FASTMACE_MODELS",
                                                                      "mh1-r2scan")
    # An explicit path wins over both, and keeps the label.
    explicit = write_fake(tmp_path / "elsewhere" / "copy.model")
    resolved = loading.resolve_checkpoint("mh1-r2scan", checkpoint=explicit, environ=environ)
    assert (resolved.path, resolved.source, resolved.label.label) == (explicit.resolve(), "explicit", "mh1-r2scan")
    # A missing explicit path is an error, not a fallback.
    with pytest.raises(FileNotFoundError, match="missing.model"):
        loading.resolve_checkpoint("mh1-r2scan", checkpoint=tmp_path / "missing.model", environ=environ)


def test_resolution_without_a_file_names_the_places(tmp_path):
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache"), "FASTMACE_MODELS": str(tmp_path / "models")}
    with pytest.raises(FileNotFoundError) as error:
        loading.resolve_checkpoint("off24-medium", environ=environ)
    assert str(tmp_path / "models") in str(error.value) and str(tmp_path / "cache") in str(error.value)
    assert "download=True" in str(error.value)
    # FASTMACE_MODELS without the file falls through to the cache directory.
    cached = write_fake(tmp_path / "cache" / "models" / "MACE-OFF24_medium.model")
    assert loading.resolve_checkpoint("off24-medium", environ=environ).path == cached.resolve()


def test_default_cache_is_the_user_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    environ = {"HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "xdg")}
    assert loading.models_cache_directory(environ) == (tmp_path / "xdg" / "fastmace" / "models").resolve()


def test_a_path_resolves_as_it_is(tmp_path):
    path = write_fake(tmp_path / "x.model")
    resolved = loading.resolve_checkpoint(str(path), environ={})
    assert (resolved.path, resolved.source, resolved.label) == (path.resolve(), "explicit", None)
    with pytest.raises(FileNotFoundError, match="neither a label"):
        loading.resolve_checkpoint(str(tmp_path / "nothing.model"), environ={})


class FakeResponse(io.BytesIO):
    def __init__(self, data, content_type="application/octet-stream"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(data))}

    def __exit__(self, *args):
        self.close()


def test_download_is_verified_before_it_is_kept(tmp_path, monkeypatch):
    data = b"checkpoint bytes"
    entry = table.LABELS["off23-small"]
    fake_checkpoint = table.Checkpoint(**{**entry.checkpoint.__dict__, "sha256": hashlib.sha256(data).hexdigest()})
    fake = table.ModelLabel("off23-small", fake_checkpoint, None, "wb97m-d3bj")
    monkeypatch.setattr(table, "LABELS", {**table.LABELS, "off23-small": fake})
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache")}
    urls = []

    def urlopen(url, timeout):
        urls.append(url)
        return FakeResponse(data)

    monkeypatch.setattr(loading.urllib.request, "urlopen", urlopen)
    resolved = loading.resolve_checkpoint("off23-small", environ=environ, download=True)
    assert urls == [entry.checkpoint.url]
    assert resolved.source == "download" and resolved.path.read_bytes() == data
    assert resolved.path == (tmp_path / "cache" / "models" / "MACE-OFF23_small.model").resolve()
    # A file whose hash differs from the table is not kept.
    monkeypatch.setattr(loading.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"tampered"))
    with pytest.raises(loading.CheckpointTrustError, match="SHA-256"):
        loading.download_checkpoint(entry.checkpoint.url, tmp_path / "d" / "f.model", fake_checkpoint.sha256)
    assert not (tmp_path / "d" / "f.model").exists() and list((tmp_path / "d").iterdir()) == []


def test_no_nas_path_in_loading_sources():
    for path in ("fastmace/model/loading.py", "fastmace/model/labels.py", "fastmace/_patches.py",
                 "fastmace/kernels/_oeq_runtime.py"):
        assert "/mnt/homes" not in (REPOSITORY_ROOT / path).read_text(encoding="utf-8")


# ------------------------------------------------------------------------ hash and trust


@pytest.fixture
def tiny_checkpoint(tmp_path):
    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return path, model


@pytest.fixture
def fresh_licences(monkeypatch):
    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())


def test_trust_rules(tmp_path):
    path = write_fake(tmp_path / "a.model", b"abc")
    digest = loading.sha256_of(path)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    bare = loading.ResolvedCheckpoint(path, "explicit", None)
    with pytest.raises(loading.CheckpointTrustError, match=digest):
        loading.check_trust(bare, digest)
    assert loading.check_trust(bare, digest, sha256=digest.upper()) == "sha256"
    assert loading.check_trust(bare, digest, trusted=True) == "trusted"
    with pytest.raises(loading.CheckpointTrustError, match="not the expected"):
        loading.check_trust(bare, digest, sha256="0" * 64, trusted=True)
    labelled = loading.ResolvedCheckpoint(path, "explicit", table.LABELS["mh1-r2scan"])
    with pytest.raises(loading.CheckpointTrustError, match="a522eb7f"):
        loading.check_trust(labelled, digest)
    assert loading.check_trust(labelled, digest, trusted=True) == "trusted"
    assert loading.check_trust(labelled, table.LABELS["mh1-r2scan"].checkpoint.sha256) == "label"


def test_load_records_the_hash_and_matches_mace(tiny_checkpoint, fresh_licences):
    path, model = tiny_checkpoint
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(loading.CheckpointTrustError):
        loading.load(str(path), backend="e3nn", device="cpu")
    with pytest.warns(loading.ModelLicenceWarning, match="licence is unknown"):
        loaded = loading.load(str(path), backend="e3nn", device="cpu", sha256=digest)
    assert loaded.sha256 == digest and loaded.trust == "sha256" and loaded.path == path.resolve()
    assert (loaded.backend, loaded.dtype, loaded.device, loaded.label, loaded.cueq_conv_fusion) == (
        "e3nn", "float64", "cpu", None, None)
    record = json.loads(json.dumps(loaded.provenance()))
    assert record["sha256"] == digest and record["path"] == str(path.resolve())
    # The same numbers as MACE's calculator built from the same file.
    from mace.calculators import MACECalculator
    reference = MACECalculator(model_paths=str(path), device="cpu", default_dtype="float64")
    results = []
    for calc in (reference, loaded.calculator, reference):  # a throwaway first evaluation (docs/development.md)
        atoms = nacl(rattle=0.05, seed=1)
        atoms.calc = calc
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
    np.testing.assert_allclose(results[1][0], results[2][0], rtol=1e-12)
    np.testing.assert_allclose(results[1][1], results[2][1], rtol=1e-12, atol=1e-14)


def test_licence_is_shown_once_per_checkpoint(tiny_checkpoint, fresh_licences):
    path, _ = tiny_checkpoint
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loading.load(str(path), backend="e3nn", device="cpu", trusted=True)
        loading.load(str(path), backend="e3nn", device="cpu", trusted=True)
    assert [type(w.message) for w in caught].count(loading.ModelLicenceWarning) == 1
    entry = table.LABELS["mh1-r2scan"]
    message = loading.licence_message(entry, Path(entry.checkpoint.file_name), entry.checkpoint.sha256)
    assert "Academic Software License" in message and "non-commercial" in message
    assert "MIT" in loading.licence_message(table.LABELS["mp0-medium"], Path("x"), "0" * 64)


def test_load_refuses_bad_arguments(tiny_checkpoint):
    path, _ = tiny_checkpoint
    with pytest.raises(ValueError, match="backend"):
        loading.load(str(path), backend="oeq", trusted=True)
    with pytest.raises(ValueError, match="dtype"):
        loading.load(str(path), dtype="float16", trusted=True)


# ------------------------------------------------------------------------ naive-cuEq guard (ported)


def fake_cueq(name, method=None, children=()):
    """A module whose class looks like a cuequivariance_torch one (name, module, .method)."""
    cls = type(name, (torch.nn.Module,), {"__module__": "cuequivariance_torch.fake"})
    module = cls()
    if method is not None:
        module.method = method
    for index, child in enumerate(children):
        module.add_module(f"child{index}", child)
    return module


def fake_model(**modules):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(1))
    for name, module in modules.items():
        model.add_module(name, module)
    return model


@pytest.fixture
def ops_import():
    with mock.patch.object(loading, "_ops_import_error", return_value=None) as patched:
        yield patched


def fused_model():
    return fake_model(
        symmetric=fake_cueq("SymmetricContraction", "uniform_1d",
                            [fake_cueq("SegmentedPolynomial", "uniform_1d",
                                       [fake_cueq("SegmentedPolynomialFromUniform1dJit")])]),
        conv_tp=fake_cueq("ChannelWiseTensorProduct", "uniform_1d"),
        linear=fake_cueq("Linear", "naive", [fake_cueq("SegmentedPolynomial", "naive",
                                                       [fake_cueq("SegmentedPolynomialNaive")])]))


def test_fused_kernels_pass_and_linear_layers_are_ignored(ops_import):
    model = fused_model()
    assert loading.fused_cueq_problems(model, "cuda") == []
    summary = loading.require_fused_cueq(model, "cuda")
    assert (summary["fused_cueq_modules"], summary["checked"], summary["naive"]) == (2, True, [])


def test_naive_symmetric_contraction_is_refused(ops_import):
    model = fake_model(symmetric=fake_cueq("SymmetricContraction", "uniform_1d",
                                           [fake_cueq("SegmentedPolynomial", "naive",
                                                      [fake_cueq("SegmentedPolynomialNaive")])]))
    problems = loading.fused_cueq_problems(model, "cuda")
    assert len(problems) == 2  # the polynomial's method and its naive implementation
    assert all(problem.startswith("symmetric.child0") for problem in problems)
    with pytest.raises(loading.NaiveCueqError, match="naive kernels"):
        loading.require_fused_cueq(model, "cuda")
    summary = loading.require_fused_cueq(model, "cuda", allow_naive=True)  # recorded, not refused
    assert summary["allowed_naive"] and summary["naive"] == problems


def test_naive_fused_convolution_is_refused(ops_import):
    model = fake_model(interaction=fake_model(conv_tp=fake_cueq("SegmentedPolynomial", "naive")))
    assert len(loading.fused_cueq_problems(model, "cuda")) == 1


def test_missing_ops_extension_is_refused(ops_import):
    ops_import.return_value = "ImportError: libcublas"
    problems = loading.fused_cueq_problems(fused_model(), "cuda")
    assert len(problems) == 1 and "cuequivariance_ops_torch cannot be imported" in problems[0]


def test_cpu_and_models_without_cueq_pass(ops_import):
    naive = fake_model(symmetric=fake_cueq("SymmetricContraction", "naive"))
    assert loading.fused_cueq_problems(naive) == []  # parameters on the CPU
    assert not loading.require_fused_cueq(naive)["checked"]
    plain = tiny_mace_model([11, 17])  # e3nn only
    assert loading.fused_cueq_problems(plain, "cuda") == []


def test_load_refuses_a_model_that_fell_back(tiny_checkpoint, fresh_licences, monkeypatch):
    path, _ = tiny_checkpoint
    monkeypatch.setattr(loading, "fused_cueq_problems", lambda model, device_type=None: ["x runs naive"])
    with pytest.raises(loading.NaiveCueqError, match="x runs naive"):
        loading.load(str(path), backend="e3nn", device="cpu", trusted=True)
    loaded = loading.load(str(path), backend="e3nn", device="cpu", trusted=True, allow_naive_cueq=True)
    assert loaded.fused_cueq["allowed_naive"]


# ------------------------------------------------------------------------ hybrid


def test_hybrid_converter_keeps_the_last_layer_irreps(monkeypatch):
    """MACE 0.3.16's hybrid run passes keep_last_layer_irreps=False; the fix passes the config's value."""
    import mace.cli.convert_e3nn_hybrid as hybrid
    received = []
    monkeypatch.setattr(hybrid, "transfer_symmetric_contractions",
                        lambda *args, keep_last_layer_irreps=False, **kwargs: received.append(keep_last_layer_irreps))
    original_extract = hybrid.extract_config_mace_model
    model = tiny_mace_model([11, 17], keep_last_layer_irreps=True)

    def extract_like_polar(source):  # PolarMACE's config carries keep_last_layer_irreps=True
        return {**original_extract(source), "keep_last_layer_irreps": source is model}

    monkeypatch.setattr(hybrid, "extract_config_mace_model", extract_like_polar)
    patched_transfer, patched_extract = hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model
    for source, expected in ((model, True), (tiny_mace_model([11, 17]), False)):
        with loading.hybrid_keeps_last_layer_irreps():
            config = hybrid.extract_config_mace_model(source)  # as hybrid.run does, then the transfer
            hybrid.transfer_symmetric_contractions({}, {}, 1, None, 2, 2, True, keep_last_layer_irreps=False)
        assert config["keep_last_layer_irreps"] is expected
    assert received == [True, False]
    assert hybrid.transfer_symmetric_contractions is patched_transfer  # restored on exit
    assert hybrid.extract_config_mace_model is patched_extract
    hybrid.transfer_symmetric_contractions({}, {}, 1, None, 2, 2, True, keep_last_layer_irreps=False)
    assert received[-1] is False  # outside the context MACE's own value is passed through


@pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange (the polar extra)")
def test_polar_config_asks_to_keep_the_last_layer():
    import mace.cli.convert_e3nn_hybrid as hybrid
    model = tiny_mace_model([11, 17], **FAMILIES["polar"])
    assert hybrid.extract_config_mace_model(model).get("keep_last_layer_irreps") is True


def test_hybrid_is_refused_after_mace_was_imported(tiny_checkpoint, monkeypatch, tmp_path):
    path, _ = tiny_checkpoint
    monkeypatch.setattr(_patches, "_RECORDS", {})
    monkeypatch.setitem(sys.modules, "mace", sys.modules.get("mace") or object())
    assert "mace" in loading.hybrid_blockers()
    with pytest.raises(loading.HybridUnavailableError, match="mace was imported before") as error:
        loading.load(str(path), backend="hybrid", trusted=True, oeq_cache=tmp_path / "oeq")
    assert "mace" in error.value.reason
    assert not (tmp_path / "oeq").exists()  # nothing prepared


def test_hybrid_reuses_the_runtime_prepared_earlier(monkeypatch):
    record = _patches.PatchRecord("oeq_runtime", "/cache/cuda-runtime", ("env:CUDA_HOME",))
    monkeypatch.setattr(_patches, "_RECORDS", {"oeq_runtime": record})
    monkeypatch.setitem(sys.modules, "openequivariance", object())
    assert loading.prepare_hybrid() is record


def test_hybrid_failure_of_the_runtime_is_a_reason(monkeypatch, tmp_path):
    from fastmace.kernels import _oeq_runtime
    monkeypatch.setattr(_patches, "_RECORDS", {})
    monkeypatch.setattr(loading, "hybrid_blockers", lambda: [])

    def fail(cache, preload=True, environ=None):
        raise _oeq_runtime.OEqRuntimeError("no CUDA 13 runtime wheel")

    monkeypatch.setattr(_oeq_runtime, "prepare_in_process", fail)
    with pytest.raises(loading.HybridUnavailableError, match="no CUDA 13 runtime wheel"):
        loading.prepare_hybrid(tmp_path)


# ------------------------------------------------------------------------ OMOL-0 embedding


def test_omol_embedding_is_checked_against_the_table():
    entry = table.LABELS["omol0-xl"]
    model = torch.nn.Module()
    model.embedding_specs = {k: dict(v) for k, v in entry.checkpoint.embedding_specs.items()}
    assert loading.check_embedding(model, entry, charge=0, spin_multiplicity=1) == model.embedding_specs
    assert loading.check_embedding(model, entry, charge=-100, spin_multiplicity=100)
    with pytest.raises(ValueError, match="total_charge 101"):
        loading.check_embedding(model, entry, charge=101, spin_multiplicity=None)
    with pytest.raises(ValueError, match="total_spin -1"):
        loading.check_embedding(model, entry, charge=None, spin_multiplicity=-1)
    model.embedding_specs["total_charge"]["offset"] = 0
    with pytest.raises(ValueError, match="differ from the label table"):
        loading.check_embedding(model, entry, None, None)


def test_charge_needs_a_joint_embedding():
    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    assert loading.check_embedding(model, table.LABELS["mh1-r2scan"], None, None) is None
    with pytest.raises(ValueError, match="no total_charge embedding"):
        loading.check_embedding(model, None, charge=0, spin_multiplicity=None)
    tiny_omol = tiny_mace_model([11, 17], **FAMILIES["omol"])
    specs = loading.check_embedding(tiny_omol, None, charge=1, spin_multiplicity=2)
    assert specs["total_charge"]["offset"] == 100 and specs["total_charge"]["num_classes"] == 201


# ------------------------------------------------------------------------ patches


@pytest.fixture
def clean_patches(monkeypatch):
    """MACE's own converter in place and an empty registry; the process's patches come back afterwards.

    Patches kept by their own module (``polar_longrange``) cannot be undone, so an earlier PolarMACE
    test in the same process leaves one in ``applied()``; the registry here hides them (2-T14).
    """
    import mace.calculators.mace  # noqa: F401
    from mace.cli import convert_e3nn_cueq
    calculator_module = sys.modules["mace.calculators.mace"]
    previous = _patches.record_of("cueq_conv_fusion")  # e.g. left by a GPU test that loaded a model
    _patches.restore("cueq_conv_fusion")
    monkeypatch.setattr(_patches, "_RECORDED_ELSEWHERE", ())
    saved = dict(_patches._RECORDS)
    _patches._RECORDS.clear()
    originals = convert_e3nn_cueq.run, calculator_module.run_e3nn_to_cueq
    try:
        yield convert_e3nn_cueq, calculator_module, originals
    finally:
        _patches.restore("cueq_conv_fusion")
        convert_e3nn_cueq.run, calculator_module.run_e3nn_to_cueq = originals
        _patches._RECORDS.clear()
        _patches._RECORDS.update(saved)
        if previous is not None:
            _patches.set_cueq_conv_fusion(previous.value)


def test_conv_fusion_device_rule():
    cases = {(True, "cuda:0"): True, (True, "cuda"): True, (True, "cuda:1"): True, (True, "cpu"): False,
             (False, "cuda"): False, (False, "cuda:0"): False,
             (None, "cuda"): True, (None, "cuda:0"): False, (None, "cpu"): False}
    for (fused, device), expected in cases.items():
        assert _patches.conv_fusion_for(fused, device) is expected, (fused, device)


def test_conv_fusion_patch_is_idempotent_recorded_and_restored(clean_patches):
    convert, calculator_module, (run, alias) = clean_patches
    record = _patches.set_cueq_conv_fusion(True)
    assert record.value is True and record.name == "cueq_conv_fusion"
    assert record.targets == ("mace.cli.convert_e3nn_cueq:run", "mace.calculators.mace:run_e3nn_to_cueq")
    replacement = convert.run
    assert replacement is not run and replacement.conv_fusion is True
    assert calculator_module.run_e3nn_to_cueq is replacement
    assert _patches.set_cueq_conv_fusion(True) is record and convert.run is replacement  # idempotent
    assert _patches.applied() == (record,)
    assert _patches.provenance() == [{"name": "cueq_conv_fusion", "value": True, "targets": list(record.targets),
                                      "detail": record.detail}]
    unfused = _patches.set_cueq_conv_fusion(False)
    assert unfused.value is False and convert.run.conv_fusion is False
    assert _patches.set_cueq_conv_fusion(None) is None  # MACE's own rule again
    assert convert.run is run and calculator_module.run_e3nn_to_cueq is alias
    assert _patches.applied() == ()
    assert not _patches.restore("cueq_conv_fusion")


def test_conv_fusion_scope_puts_back_the_rule_in_force(clean_patches):
    """loading.load converts inside _patches.cueq_conv_fusion: the fusion rule must not outlive the block."""
    convert, calculator_module, (run, alias) = clean_patches
    with _patches.cueq_conv_fusion(True) as record:
        assert record.value is True and convert.run.conv_fusion is True
    assert convert.run is run and calculator_module.run_e3nn_to_cueq is alias and _patches.applied() == ()
    outer = _patches.set_cueq_conv_fusion(False)
    with pytest.raises(KeyError), _patches.cueq_conv_fusion(True):
        assert convert.run.conv_fusion is True
        raise KeyError("restored on errors too")
    assert _patches.record_of("cueq_conv_fusion") == outer and convert.run.conv_fusion is False
    with _patches.cueq_conv_fusion(None) as record:  # MACE's own rule inside, the outer one after
        assert record is None and convert.run is run
    assert _patches.record_of("cueq_conv_fusion") == outer


def test_conv_fusion_patch_refuses_another_signature(clean_patches, monkeypatch):
    convert, _, _ = clean_patches

    def other(input_model, device="cpu"):
        raise AssertionError("not called")

    monkeypatch.setattr(convert, "run", other)
    monkeypatch.setattr(_patches, "_ORIGINALS", {})
    with pytest.raises(RuntimeError, match="0.3.16 parameters"):
        _patches.set_cueq_conv_fusion(True)
    assert _patches.applied() == ()


def test_conv_fusion_marker_mismatch_is_refused():
    """A MACE that ignores the replaced config class (fusion asked for, none built) is refused."""
    stand_in = type("Model", (), {"interactions": [torch.nn.Module(), torch.nn.Module()]})()
    with pytest.raises(RuntimeError, match="conv_fusion"):
        _patches.cueq_conv_fusion_run(True, lambda *args, **kwargs: stand_in)(None, device="cuda:0")
    assert _patches.cueq_conv_fusion_run(False, lambda *args, **kwargs: stand_in)(None, device="cuda:0") is stand_in


def test_oeq_runtime_cannot_be_undone(monkeypatch):
    monkeypatch.setattr(_patches, "_RECORDS", {"oeq_runtime": _patches.PatchRecord("oeq_runtime", "/r", ())})
    with pytest.raises(RuntimeError, match="cannot be undone"):
        _patches.restore("oeq_runtime")


# ------------------------------------------------------------------------ imports


def test_import_fastmace_does_not_import_mace():
    code = textwrap.dedent("""
        import sys
        import fastmace
        import fastmace.model.labels, fastmace.model.loading, fastmace._patches, fastmace.kernels._oeq_runtime
        print(sorted(m for m in ("mace", "openequivariance", "torch.utils.cpp_extension", "cuequivariance_torch")
                     if m in sys.modules))
    """)
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
