"""Public ``fm.load``, the models cache and checkpoint trust (SPEC §7.1, §7.6, §7.7 "Heads", §14; plan task 1b-T08).

Tiny random models stand in for checkpoints. Downloads come from a local HTTP
server; nothing leaves the machine and no GPU is used.
"""

import functools
import hashlib
import http.server
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import _cache
from fastmace.errors import CheckpointTrustError, FastMACEError, HybridUnavailableError
from fastmace.model import labels as table
from fastmace.model import loading

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KNOWN_FUNCTIONALS = ("pbe", "r2scan", "wb97m-v", "wb97m-d3bj")


# ------------------------------------------------------------------------ fixtures


class Payload:
    """Unpickling this object writes ``marker``: a stand-in for a malicious checkpoint."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return exec, (f"open({self.marker!r}, 'w').write('unpickled')",)


@pytest.fixture
def evil_checkpoint(tmp_path):
    marker = tmp_path / "unpickled.txt"
    path = tmp_path / "evil.model"
    torch.save(Payload(marker), path)
    assert not marker.exists()
    return path, marker


@pytest.fixture
def fresh_licences(monkeypatch):
    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())


def save(tmp_path, name, model):
    path = tmp_path / name
    torch.save(model, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def single_head(tmp_path):
    return save(tmp_path, "tiny.model", tiny_mace_model([11, 17], **FAMILIES["mh1"]))


def multi_head_model(heads):
    return tiny_mace_model([11, 17], heads=list(heads), atomic_energies=np.zeros((len(heads), 2)),
                           **FAMILIES["mh1"])


@pytest.fixture
def http_server(tmp_path):
    """A local HTTP server for ``tmp_path / "served"``; yields (directory, base URL)."""
    served = tmp_path / "served"
    served.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(served)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield served, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------------ the models cache kind


def test_models_is_a_cache_kind(tmp_path):
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache")}
    assert "models" in _cache.CACHE_KINDS
    path = _cache.cache_dir("models", create=False, environ=environ)
    assert path == (tmp_path / "cache").resolve() / "models" and not path.exists()
    assert loading.models_cache_directory(environ) == path


# ------------------------------------------------------------------------ public fm.load


def test_public_load_is_pristine(single_head, fresh_licences):
    import fastmace as fm

    path, digest = single_head
    assert fm.load is loading.load_pristine
    with pytest.warns(loading.ModelLicenceWarning):
        loaded = fm.load(str(path), sha256=digest)
    assert (loaded.backend, loaded.dtype, loaded.device) == ("e3nn", "float64", "cpu")
    assert (loaded.sha256, loaded.trust, loaded.cueq_conv_fusion) == (digest, "sha256", None)


def test_public_load_keeps_the_float64_e0_table(tmp_path, fresh_licences):
    # 1b-T03: the float64 energy needs the checkpoint's E0 table; plans build from the pristine model.
    e0 = np.array([-1.2345678901234567, -12522.123456789012])
    path, digest = save(tmp_path, "e0.model", tiny_mace_model([11, 17], atomic_energies=e0, **FAMILIES["mh1"]))
    with pytest.warns(loading.ModelLicenceWarning):
        loaded = loading.load_pristine(str(path), sha256=digest)
    assert loaded.atomic_energies.dtype == torch.float64
    assert loaded.atomic_energies.flatten().tolist() == e0.tolist()


def test_internal_load_keeps_its_defaults():
    import inspect

    defaults = {name: p.default for name, p in inspect.signature(loading.load).parameters.items()}
    assert (defaults["backend"], defaults["dtype"], defaults["device"]) == ("cueq", "float64", "cuda:0")


# ------------------------------------------------------------------------ trust


def test_a_path_without_sha256_or_trusted_is_refused(single_head):
    path, digest = single_head
    with pytest.raises(CheckpointTrustError) as error:
        loading.load_pristine(str(path))
    assert isinstance(error.value, FastMACEError)
    assert digest in str(error.value) and "trusted=True" in str(error.value)
    assert error.value.why and error.value.next_step  # SPEC §7.7 message shape


def test_a_wrong_hash_is_refused_before_unpickling(evil_checkpoint):
    path, marker = evil_checkpoint
    with pytest.raises(CheckpointTrustError, match="not the expected"):
        loading.load_pristine(str(path), sha256="0" * 64)
    with pytest.raises(CheckpointTrustError, match="not the expected"):
        loading.load_pristine(str(path), sha256="0" * 64, trusted=True)  # a stated hash always counts
    with pytest.raises(CheckpointTrustError):
        loading.load_pristine(str(path))
    # A label's file must match the release: a local copy that does not is refused too.
    with pytest.raises(CheckpointTrustError, match="a522eb7f"):
        loading.load_pristine("mh1-r2scan", checkpoint=path)
    assert not marker.exists()
    # The control: the file does run code when it is unpickled.
    torch.load(path, weights_only=False)
    assert marker.read_text() == "unpickled"


def test_the_provenance_carries_the_trust_decision(single_head, fresh_licences):
    path, digest = single_head
    with pytest.warns(loading.ModelLicenceWarning):
        by_hash = loading.load_pristine(str(path), sha256=digest.upper())
    trusted = loading.load_pristine(str(path), trusted=True)
    assert (by_hash.provenance()["trust"], by_hash.provenance()["sha256"]) == ("sha256", digest)
    assert (trusted.provenance()["trust"], trusted.provenance()["sha256"]) == ("trusted", digest)


# ------------------------------------------------------------------------ downloads


def test_a_url_download_is_verified_into_the_models_cache(tmp_path, http_server, single_head, fresh_licences):
    served, base = http_server
    path, digest = single_head
    (served / "tiny.model").write_bytes(path.read_bytes())
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache")}
    url = f"{base}/tiny.model"
    # Not opted in: nothing is downloaded.
    with pytest.raises(FileNotFoundError, match="download=True"):
        loading.load_pristine(url, sha256=digest, environ=environ)
    # Neither a hash nor trust: refused before anything is fetched.
    with pytest.raises(CheckpointTrustError):
        loading.load_pristine(url, download=True, environ=environ)
    assert not (tmp_path / "cache" / "models").exists()
    with pytest.warns(loading.ModelLicenceWarning):
        loaded = loading.load_pristine(url, sha256=digest, download=True, environ=environ)
    cached = _cache.cache_dir("models", create=False, environ=environ) / "tiny.model"
    assert loaded.path == cached and cached.read_bytes() == path.read_bytes()
    assert (loaded.checkpoint_source, loaded.trust) == ("download", "sha256")


def test_a_tampered_url_download_is_not_kept_or_unpickled(tmp_path, http_server, evil_checkpoint):
    served, base = http_server
    path, marker = evil_checkpoint
    (served / "evil.model").write_bytes(path.read_bytes())
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache")}
    with pytest.raises(CheckpointTrustError, match="SHA-256"):
        loading.load_pristine(f"{base}/evil.model", sha256="0" * 64, download=True, environ=environ)
    assert list((tmp_path / "cache" / "models").iterdir()) == []
    assert not marker.exists()


def test_a_label_download_is_verified_against_the_table(tmp_path, http_server, monkeypatch, evil_checkpoint):
    served, base = http_server
    path, marker = evil_checkpoint
    entry = table.LABELS["off23-small"]
    (served / entry.checkpoint.file_name).write_bytes(path.read_bytes())
    local = table.Checkpoint(**{**entry.checkpoint.__dict__, "url": f"{base}/{entry.checkpoint.file_name}"})
    monkeypatch.setattr(table, "LABELS", {**table.LABELS, "off23-small": table.ModelLabel(
        "off23-small", local, None, entry.functional)})
    environ = {"FASTMACE_CACHE": str(tmp_path / "cache")}
    with pytest.raises(CheckpointTrustError, match=entry.checkpoint.sha256):
        loading.load_pristine("off23-small", download=True, environ=environ)
    assert list((tmp_path / "cache" / "models").iterdir()) == []
    assert not marker.exists()


# ------------------------------------------------------------------------ heads


@pytest.mark.parametrize("heads", [("alpha", "beta"), ("default", "beta")])
def test_a_multi_head_checkpoint_needs_a_head(tmp_path, heads, fresh_licences):
    # ("default", "beta"): MACE would pick "default" silently; SPEC §7.7 has no default head.
    path, digest = save(tmp_path, "multi.model", multi_head_model(heads))
    with pytest.raises(ValueError) as error:
        loading.load_pristine(str(path), sha256=digest)
    assert all(head in str(error.value) for head in heads)
    with pytest.warns(loading.ModelLicenceWarning):
        loaded = loading.load_pristine(str(path), "beta", sha256=digest)
    assert (loaded.head, loaded.heads) == ("beta", heads)
    with pytest.raises(ValueError, match="no head"):
        loading.load_pristine(str(path), "gamma", sha256=digest)


def test_a_label_names_its_head():
    with pytest.raises(ValueError, match="matpes_r2scan"):
        loading.load_pristine("mh1")
    assert table.find("mh1-r2scan").head == "matpes_r2scan"


# ------------------------------------------------------------------------ head functionals


@pytest.mark.parametrize("label", sorted(table.LABELS))
def test_every_label_has_a_functional_or_an_explicit_unknown(label):
    entry = table.LABELS[label]
    source = entry.functional_source
    assert source, f"{label}: no source for the functional"
    if entry.functional is None:
        assert source.startswith("unknown:"), source
    else:
        assert entry.functional in KNOWN_FUNCTIONALS
        assert not source.startswith("unknown")


def test_every_head_of_every_checkpoint_has_a_functional_source():
    for checkpoint in table.CHECKPOINTS.values():
        keys = checkpoint.heads or ("",)
        assert set(checkpoint.head_functionals) == set(keys), checkpoint.name
        assert set(checkpoint.functional_sources) == set(keys), checkpoint.name


def test_mh1_head_functionals_against_the_release():
    mh1 = table.CHECKPOINTS["mh-1"]
    assert dict(mh1.head_functionals) == {
        "matpes_r2scan": "r2scan", "mp_pbe_refit_add": "pbe", "spice_wB97M": "wb97m-d3bj",
        "oc20_usemppbe": "pbe", "omol": "wb97m-v", "omat_pbe": "pbe"}
    for head, source in mh1.functional_sources.items():
        assert "huggingface.co/mace-foundations/mace-mh-1" in source or "arXiv:2510.25380" in source, head


def test_unknown_functional_is_not_a_string():
    # T20 tells "unknown" from a functional by None; a path or URL has no label and no functional.
    assert table.find("polar1-s").functional is None
    assert table.find("polar1-s").functional_source.startswith("unknown:")


# ------------------------------------------------------------------------ OEq runtime before MACE (1b-T08b)


@pytest.fixture
def no_runtime_record(monkeypatch):
    """Neither an OEq runtime record nor a blocking import: preparation is still possible."""
    monkeypatch.setattr(loading._patches, "record_of", lambda name: None)
    monkeypatch.setattr(loading, "hybrid_blockers", lambda modules=None: [])


def test_oeq_runtime_is_prepared_when_possible(no_runtime_record, monkeypatch):
    calls = []
    monkeypatch.setattr(loading, "_oeq_installed", lambda: True)
    monkeypatch.setattr(loading, "_cuda_visible", lambda: True)
    monkeypatch.setattr(loading, "prepare_hybrid", lambda cache=None: calls.append(cache) or "record")
    assert loading.prepare_hybrid_if_possible() == "prepared"
    assert calls == [None]


@pytest.mark.parametrize("oeq, cuda, expected", [
    (False, True, "skipped: OpenEquivariance is not installed"),
    (True, False, "skipped: no CUDA device is visible"),
])
def test_oeq_runtime_is_skipped_without_oeq_or_cuda(no_runtime_record, monkeypatch, oeq, cuda, expected):
    monkeypatch.setattr(loading, "_oeq_installed", lambda: oeq)
    monkeypatch.setattr(loading, "_cuda_visible", lambda: cuda)
    monkeypatch.setattr(loading, "prepare_hybrid", lambda cache=None: pytest.fail("prepared"))
    assert loading.prepare_hybrid_if_possible() == expected


def test_oeq_runtime_is_not_prepared_too_late(monkeypatch):
    monkeypatch.setattr(loading._patches, "record_of", lambda name: None)
    monkeypatch.setattr(loading, "hybrid_blockers", lambda modules=None: ["mace"])
    monkeypatch.setattr(loading, "_oeq_installed", lambda: True)
    monkeypatch.setattr(loading, "_cuda_visible", lambda: True)
    monkeypatch.setattr(loading, "prepare_hybrid", lambda cache=None: pytest.fail("prepared"))
    assert loading.prepare_hybrid_if_possible() == "too late: mace was imported first"


def test_an_earlier_oeq_runtime_is_reused(monkeypatch):
    monkeypatch.setattr(loading._patches, "record_of", lambda name: "record" if name == "oeq_runtime" else None)
    monkeypatch.setattr(loading, "prepare_hybrid", lambda cache=None: pytest.fail("prepared again"))
    assert loading.prepare_hybrid_if_possible() == "prepared"


def test_a_failed_oeq_runtime_is_tolerated_and_recorded(no_runtime_record, monkeypatch, single_head,
                                                         fresh_licences):
    path, digest = single_head

    def fail(cache=None):
        raise HybridUnavailableError("the OEq runtime is missing")

    monkeypatch.setattr(loading, "_oeq_installed", lambda: True)
    monkeypatch.setattr(loading, "_cuda_visible", lambda: True)
    monkeypatch.setattr(loading, "prepare_hybrid", fail)
    with pytest.warns(loading.ModelLicenceWarning):
        loaded = loading.load_pristine(str(path), sha256=digest)
    assert loaded.oeq_runtime == ("failed: the cuEquivariance/OpenEquivariance hybrid is unavailable: "
                                  "the OEq runtime is missing")
    assert loaded.provenance()["oeq_runtime"] == loaded.oeq_runtime
    assert loaded.backend == "e3nn"


def test_the_cpu_only_path_records_why_and_stays_quiet(single_head, monkeypatch, fresh_licences):
    import warnings

    path, digest = single_head
    monkeypatch.setattr(loading, "_cuda_visible", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = loading.prepare_hybrid_if_possible()
    assert caught == [] and status.startswith("skipped:")
    with pytest.warns(loading.ModelLicenceWarning):
        loaded = loading.load_pristine(str(path), sha256=digest)
    assert loaded.oeq_runtime == status


def test_fm_load_prepares_the_oeq_runtime_before_mace_is_imported(single_head):
    # A fresh interpreter: MACE is not imported yet. OEq and CUDA are mocked present; the
    # preparation records whether MACE was imported when it ran.
    path, digest = single_head
    code = textwrap.dedent(f"""
        import json, sys
        import fastmace as fm
        from fastmace.model import loading
        seen = []
        loading._oeq_installed = lambda: True
        loading._cuda_visible = lambda: True
        loading.prepare_hybrid = lambda cache=None: seen.append(sorted(m for m in sys.modules
                                                                       if m.split(".")[0] == "mace")) or "record"
        loaded = fm.load({str(path)!r}, sha256={digest!r})
        print(json.dumps({{"seen": seen, "status": loaded.oeq_runtime, "mace_after": "mace" in sys.modules}}))
    """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {"seen": [[]], "status": "prepared", "mace_after": True}
