"""Model loading (SPEC §7.1 ``fm.load``, §7.6 checkpoint trust, §9.2 naive-cuEq guard, §13 hashes).

Copied from the research engine's ``common.load_calculator`` (benchmark
scripts), ``model_store.checkpoint``, ``workflow.load_md_calculator`` and
``langevin/upstream_mace.py`` and refactored: no NAS paths, no environment
variables as switches, no module globals.

``fm.load`` is ``load_pristine``: ``load`` as e3nn in float64 on the CPU, the
pristine model that each plan converts (1b-T12). ``load`` itself keeps its explicit
backend, dtype and device for ``FixedPlanner`` and ``main.py``.

What ``load`` does, in order:

1. **Resolve** the checkpoint (``resolve_checkpoint``): an explicit path, else
   ``$FASTMACE_MODELS/<file>``, else ``<cache root>/models/<file>`` (SPEC §14),
   else, with ``download=True``, the label's official URL into the cache
   directory. Never a NAS path; workstation runs set ``FASTMACE_MODELS`` or
   pass the path.
2. **Hash and trust** (``sha256_of``, ``check_trust``): the SHA-256 is computed
   before anything is unpickled. A label's file must match the table's hash; a
   bare path needs ``sha256=`` or ``trusted=True`` (MACE checkpoints are pickles
   that run code when loaded).
3. **OEq runtime** for the hybrid backend: ``fastmace._patches.prepare_oeq_runtime``
   before the first MACE import. If MACE, OpenEquivariance or
   ``torch.utils.cpp_extension`` was imported earlier, the hybrid is refused
   with the reason (``HybridUnavailableError``); the planner then falls back to
   plain cuEq.
4. **Build** MACE's calculator (MACE's own loader and cuEq/OEq converters), with
   the fused cuEq convolution of MACE PR 1641 for plain cuEq
   (``_patches.cueq_conv_fusion``, in force for this conversion only) and, for the hybrid, the converter fix
   that keeps the last layer's irreps (``hybrid_keeps_last_layer_irreps``),
   between fresh cuEquivariance group-theory caches (``_fresh_cueq_basis_cache``).
5. **Check** that the backend asked for was built (no silent e3nn fallback), that
   cuEquivariance's fused kernels did not fall back to its naive ones
   (``require_fused_cueq``), the head, and OMOL-0's joint embedding.
6. Show the model's **licence** (a ``ModelLicenceWarning``, once per checkpoint
   and process) and return a ``LoadedModel`` with its provenance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import tempfile
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastmace import _cache, _patches
from fastmace.errors import (  # noqa: F401 (re-exported: the 1a names)
    BackendError,
    CheckpointTrustError,
    HybridUnavailableError,
    ModelLicenceWarning,
    NaiveCueqError,
)
from fastmace.model import labels as label_table

MODELS_VARIABLE = "FASTMACE_MODELS"
BACKENDS = ("hybrid", "cueq", "e3nn")
DTYPES = ("float32", "float64")
# cuEquivariance modules that MACE builds with a fused kernel (method "uniform_1d"). cuEq's Linear
# layers are naive by design (plain matrix products) and are not checked.
FUSED_CUEQ_TYPES = ("SymmetricContraction", "ChannelWiseTensorProduct")
# Modules whose import fixes the OEq build environment: after any of them, the hybrid is too late.
HYBRID_BLOCKING_MODULES = ("mace", "openequivariance", "torch.utils.cpp_extension")
_CHUNK = 1 << 20


# Checkpoints whose licence has been shown in this process (by SHA-256): a record of
# messages already printed, not a switch.
_LICENCES_SHOWN: set = set()


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Where a checkpoint was found."""

    path: Path
    source: str  # "explicit", "FASTMACE_MODELS", "cache" or "download"
    label: label_table.ModelLabel | None
    searched: tuple = ()  # the places looked at, in order


@dataclass(frozen=True)
class LoadedModel:
    """A loaded MACE model and everything provenance needs about it (SPEC §13)."""

    calculator: object  # mace.calculators.MACECalculator
    path: Path
    sha256: str
    trust: str  # "label", "sha256" or "trusted"
    checkpoint_source: str  # ResolvedCheckpoint.source
    label: str | None
    head: str | None
    heads: tuple
    model_type: str
    family: str | None
    functional: str | None
    licence: str | None
    backend: str
    dtype: str
    device: str
    cueq_conv_fusion: bool | None  # plain cuEq: whether the convolution is fused; None for the others
    fused_cueq: dict  # require_fused_cueq's summary
    embedding_specs: dict | None
    charge: int | None
    spin_multiplicity: int | None
    patches: tuple = field(default=())  # _patches.applied() while the model was built
    # The checkpoint's E0 table as float64, copied before MACE's dtype conversion (SPEC §7.3).
    atomic_energies: object = None
    # fm.load only: what became of the OEq runtime before MACE was imported (prepare_hybrid_if_possible).
    oeq_runtime: str | None = None

    @property
    def model(self):
        """The (converted) torch model."""
        return self.calculator.models[0]

    def provenance(self):
        """The model block of metadata.json."""
        return {
            "path": str(self.path), "sha256": self.sha256, "trust": self.trust,
            "checkpoint_source": self.checkpoint_source, "label": self.label, "head": self.head,
            "heads": list(self.heads), "model_type": self.model_type, "family": self.family,
            "functional": self.functional, "licence": self.licence, "backend": self.backend, "dtype": self.dtype,
            "device": self.device, "cueq_conv_fusion": self.cueq_conv_fusion, "fused_cueq": dict(self.fused_cueq),
            "embedding_specs": self.embedding_specs, "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity, "oeq_runtime": self.oeq_runtime,
            "patches": [{"name": r.name, "value": r.value, "targets": list(r.targets), "detail": r.detail}
                        for r in self.patches],
        }


# ------------------------------------------------------------------------ resolution


def _is_url(source):
    return isinstance(source, str) and source.startswith(("https://", "http://"))


def models_cache_directory(environ=None):
    """``<cache root>/models``: where downloads go and the last place a label's file is looked for."""
    return _cache.cache_dir("models", create=False, environ=environ)


def resolve_checkpoint(source, head=None, checkpoint=None, environ=None, download=False, sha256=None):
    """Find the checkpoint of ``source`` (a label, alias, checkpoint name, path or URL).

    Order for a label: ``checkpoint`` (explicit path), else ``$FASTMACE_MODELS/<file>``,
    else ``<cache root>/models/<file>``, else (``download=True``) the official URL into
    the cache directory. A path is used as it is; a URL's file is taken from the cache
    directory if it is there, else downloaded into it with ``download=True`` only. A download is checked against the
    table's SHA-256, or for a URL against ``sha256``, and not kept if it differs.
    """
    environ = os.environ if environ is None else environ
    entry = None if isinstance(source, os.PathLike) or _is_url(source) else label_table.find(str(source), head)
    if checkpoint is not None:
        path = Path(os.path.expanduser(checkpoint))
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {path} not found")
        return ResolvedCheckpoint(path.resolve(), "explicit", entry, (str(path),))
    if entry is None and not _is_url(source):
        path = Path(os.path.expanduser(source))
        if not path.is_file():
            raise FileNotFoundError(f"{source!r} is neither a label of the table ({', '.join(label_table.LABELS)}) "
                                    "nor an existing checkpoint file")
        return ResolvedCheckpoint(path.resolve(), "explicit", None, (str(path),))
    file_name = entry.checkpoint.file_name if entry is not None else source.rsplit("/", 1)[1]
    searched = []
    models = environ.get(MODELS_VARIABLE)
    if models and entry is not None:
        path = Path(os.path.expanduser(models)) / file_name
        searched.append(str(path))
        if path.is_file() and path.stat().st_size:
            return ResolvedCheckpoint(path.resolve(), MODELS_VARIABLE, entry, tuple(searched))
    path = models_cache_directory(environ) / file_name
    searched.append(str(path))
    if path.is_file() and path.stat().st_size:
        return ResolvedCheckpoint(path.resolve(), "cache", entry, tuple(searched))
    if not download:
        raise FileNotFoundError(f"checkpoint {file_name} not found (looked in {', '.join(searched)}); pass "
                                f"checkpoint=, set {MODELS_VARIABLE}, or load with download=True")
    url = entry.checkpoint.url if entry is not None else source
    expected = entry.checkpoint.sha256 if entry is not None else (sha256.lower() if sha256 else None)
    return ResolvedCheckpoint(download_checkpoint(url, path, expected), "download", entry, tuple(searched))


def download_checkpoint(url, target, sha256=None):
    """Download ``url`` to ``target`` atomically; with ``sha256``, refuse a file whose hash differs."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
    temporary = Path(temporary)
    try:
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise OSError(f"checkpoint URL returned HTML: {url}")
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                output.write(chunk)
            expected_length = response.headers.get("Content-Length")
            if expected_length is not None and output.tell() != int(expected_length):
                raise OSError(f"incomplete checkpoint download: {url}")
            if not output.tell():
                raise OSError(f"empty checkpoint: {url}")
        if sha256 is not None and digest.hexdigest() != sha256:
            raise CheckpointTrustError(
                f"the download {url} has SHA-256 {digest.hexdigest()}, not the expected {sha256}",
                why="the file differs from the one whose hash was given, so it was deleted without being unpickled",
                next_step="Check the URL and the hash; a checkpoint is a pickle and runs code when loaded.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


def sha256_of(path):
    """The SHA-256 of a file, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check_trust(resolved, digest, sha256=None, trusted=False):
    """Decide whether the file may be unpickled (SPEC §7.6); returns "label", "sha256" or "trusted".

    ``sha256=`` must match the file. A label's file must match the table. A file
    without either needs ``trusted=True``.
    """
    if sha256 is not None:
        if digest != sha256.lower():
            raise CheckpointTrustError(
                f"{resolved.path} has SHA-256 {digest}, not the expected {sha256}",
                why="the file is not the one whose hash was given, so it was not unpickled")
        return "sha256"
    if resolved.label is not None:
        if digest == resolved.label.checkpoint.sha256:
            return "label"
        if trusted:
            return "trusted"
        raise CheckpointTrustError(
            f"{resolved.path} has SHA-256 {digest}, but label {resolved.label.label!r} is the release with "
            f"SHA-256 {resolved.label.checkpoint.sha256}",
            why="the file is not the released checkpoint of the label, so it was not unpickled",
            next_step="Download the release again (download=True), or pass trusted=True to load this file anyway.")
    if trusted:
        return "trusted"
    raise CheckpointTrustError(
        f"{resolved.path} (SHA-256 {digest}) is not a label of the table and was not unpickled",
        why="MACE checkpoints are pickles that run code when loaded; a file from a path or URL needs a stated hash",
        next_step=f"Pass sha256='{digest}' after checking the file against its publisher, or trusted=True.")


# ------------------------------------------------------------------------ backend checks


def _cueq_enabled(calculator):
    """Whether MACE's calculator converted its model to cuEquivariance.

    MACE 0.3.16 records it as ``_enable_cueq``; an older MACE does not, so its model is searched for
    cuEquivariance modules (2-T04, unknown MACE versions).
    """
    flag = getattr(calculator, "_enable_cueq", None)
    if flag is not None:
        return bool(flag)
    return any(type(module).__module__.startswith("cuequivariance")
               for model in getattr(calculator, "models", ()) for module in model.modules())


def _device_type(device):
    import torch

    return torch.device(device).type


def _model_device_type(model):
    for tensor in model.parameters():
        return tensor.device.type
    for tensor in model.buffers():
        return tensor.device.type
    return "cpu"


def _is_cueq(module):
    return type(module).__module__.startswith("cuequivariance_torch")


def _ops_import_error():
    """None if cuequivariance_ops_torch imports, else the error text."""
    try:
        importlib.import_module("cuequivariance_ops_torch")
    except Exception as error:  # an ImportError or a failing shared library behind it
        return f"{type(error).__name__}: {error}"
    return None


def fused_cueq_problems(model, device_type=None):
    """Why the model's cuEquivariance kernels are not the fused ones MACE built them for ([] = none).

    cuEquivariance replaces a requested fused kernel by its naive implementation
    with only a UserWarning when cuequivariance_ops_torch cannot be imported, for
    example when the ops wheel resolved against a cuBLAS it was not built for
    (MACE PR 1614). A run would then be slow without an error. Checks, on CUDA
    only (cuEq's fused kernels are CUDA-only; CPU runs use the naive path by
    design), that cuequivariance_ops_torch imports and that no symmetric
    contraction or cuEq convolution (a cuEq module named conv_tp, the fused
    convolution) runs cuEq's naive implementation. Models without cuEq modules pass.
    """
    modules = [(name, module) for name, module in model.named_modules() if _is_cueq(module)]
    if not modules or (device_type or _model_device_type(model)) != "cuda":
        return []
    problems = []
    error = _ops_import_error()
    if error is not None:
        problems.append(f"cuequivariance_ops_torch cannot be imported ({error})")
    for name, module in modules:
        if type(module).__name__ not in FUSED_CUEQ_TYPES and name.rsplit(".", 1)[-1] != "conv_tp":
            continue
        for inner_name, inner in module.named_modules():
            where = f"{name}.{inner_name}" if inner_name else name
            if getattr(inner, "method", None) == "naive" or type(inner).__name__ == "SegmentedPolynomialNaive":
                problems.append(f"{where} ({type(inner).__name__}) runs cuEquivariance's naive implementation")
    return problems


def require_fused_cueq(model, device_type=None, allow_naive=False):
    """Raise NaiveCueqError if fused_cueq_problems finds any (unless allow_naive); else return a summary."""
    device_type = device_type or _model_device_type(model)
    problems = fused_cueq_problems(model, device_type)
    if problems and not allow_naive:
        raise NaiveCueqError(
            "cuEquivariance fell back to its naive kernels, which would make the run slow without an "
            "error: " + "; ".join(problems) + ". cuequivariance_ops_torch must import: check that the "
            "cuequivariance-ops-torch wheel matches torch.version.cuda and that the cuBLAS it links is "
            "the one installed (MACE PR 1614). Pass allow_naive_cueq=True to run anyway.")
    checked = sum(1 for name, module in model.named_modules() if _is_cueq(module)
                  and (type(module).__name__ in FUSED_CUEQ_TYPES or name.rsplit(".", 1)[-1] == "conv_tp"))
    return {"device": device_type, "fused_cueq_modules": checked, "checked": device_type == "cuda" and checked > 0,
            "cuequivariance_ops_torch_imports": _ops_import_error() is None, "naive": problems,
            "allowed_naive": bool(problems) and allow_naive}


def check_cueq():
    """Raise BackendError unless cuEquivariance and MACE's enable_cueq are usable."""
    try:
        for name in ("cuequivariance", "cuequivariance_torch"):
            importlib.import_module(name)
        from mace.calculators import MACECalculator
        if "enable_cueq" not in inspect.signature(MACECalculator).parameters:
            raise ImportError("installed MACE does not expose enable_cueq")
    except (ImportError, OSError) as error:
        raise BackendError(f"cuEquivariance is unavailable or incompatible: {error}") from error


def check_oeq():
    """Raise HybridUnavailableError unless OpenEquivariance and MACE's hybrid conversion are usable.

    OpenEquivariance 0.7 can import while its C++ extension failed to build, and
    MACE exposes its availability flag even then. Check the import, the built
    extension, MACE's wrapper flag and the calculator API independently, so a
    requested hybrid never silently becomes e3nn.
    """
    try:
        oeq = importlib.import_module("openequivariance")
        if getattr(oeq, "BUILT_EXTENSION", None) is not True:
            detail = getattr(oeq, "BUILT_EXTENSION_ERROR", None)
            raise ImportError("openequivariance.BUILT_EXTENSION is not True" + (f": {detail}" if detail else ""))
        extension_path = getattr(oeq, "torch_ext_so_path", None)
        if not callable(extension_path):
            raise ImportError("openequivariance.torch_ext_so_path is unavailable")
        extension_path = extension_path()
        if not extension_path or not Path(extension_path).is_file():
            raise ImportError(f"openequivariance extension artifact is unusable: {extension_path!r}")
        if not getattr(importlib.import_module("mace.modules.wrapper_ops"), "OEQ_AVAILABLE", False):
            raise ImportError("mace.modules.wrapper_ops.OEQ_AVAILABLE is false")
        mace_module = importlib.import_module("mace.calculators.mace")
        if "enable_oeq" not in inspect.signature(mace_module.MACECalculator).parameters:
            raise ImportError("installed MACE does not expose enable_oeq")
        if not getattr(mace_module, "HYBRID_AVAILABLE", False):
            raise ImportError("installed MACE does not expose hybrid cueq+oeq conversion")
    except (ImportError, OSError, TypeError, AttributeError) as error:
        raise HybridUnavailableError(f"OpenEquivariance is unavailable or incompatible ({error})") from error


def hybrid_blockers(modules=None):
    """The already imported modules that make the in-process OEq runtime too late (empty: not too late)."""
    modules = sys.modules if modules is None else modules
    return [name for name in HYBRID_BLOCKING_MODULES if name in modules]


def prepare_hybrid(cache=None):
    """Prepare the OEq runtime before MACE is imported; raises HybridUnavailableError with the reason."""
    record = _patches.record_of("oeq_runtime")
    if record is not None:
        return record
    blockers = hybrid_blockers()
    if blockers:
        raise HybridUnavailableError(
            f"{blockers[0]} was imported before FastMACE could prepare the OpenEquivariance build (in-process "
            "OEq runtime, docs/design/oeq-runtime.md); load the hybrid before importing MACE or OpenEquivariance")
    from fastmace.kernels import _oeq_runtime
    try:
        return _patches.prepare_oeq_runtime(_cache.cache_dir("oeq") if cache is None else cache)
    except _oeq_runtime.OEqRuntimeError as error:
        raise HybridUnavailableError(str(error)) from error


def _oeq_installed():
    """Whether OpenEquivariance is installed, found without importing it (an import would be too early)."""
    return importlib.util.find_spec("openequivariance") is not None


def _cuda_visible():
    import torch

    return torch.cuda.is_available()


def prepare_hybrid_if_possible(cache=None):
    """Prepare the OEq runtime when the hybrid could be planned later; never raises. Returns what happened.

    ``fm.load`` imports MACE, after which the runtime can no longer be prepared and
    every plan in the process gets plain cuEq (SPEC §9.4). So it prepares the
    runtime first, but only when OpenEquivariance is installed, a CUDA device is
    visible and nothing blocks it yet: a CPU-only process does no extra work. A
    failure is recorded, not raised; the planner then gives plain cuEq with its reason.
    """
    if _patches.record_of("oeq_runtime") is not None:
        return "prepared"
    if not _oeq_installed():
        return "skipped: OpenEquivariance is not installed"
    if not _cuda_visible():
        return "skipped: no CUDA device is visible"
    blockers = hybrid_blockers()
    if blockers:
        return f"too late: {blockers[0]} was imported first"
    try:
        prepare_hybrid(cache)
    except HybridUnavailableError as error:
        return f"failed: {error.what}"
    return "prepared"


@contextmanager
def hybrid_keeps_last_layer_irreps():
    """While a hybrid calculator is built, pass the model's keep_last_layer_irreps to the transfer.

    MACE 0.3.16's hybrid converter passes keep_last_layer_irreps=False to the
    symmetric-contraction transfer for every model; the cuEquivariance converter
    reads it from the model config. PolarMACE always keeps the last layer's
    irreps, so its hybrid conversion fails (state-dict size mismatch). Use the
    config value while a calculator is built; process-local and undone on exit,
    the package is not modified.
    """
    import mace.cli.convert_e3nn_hybrid as hybrid
    transfer, extract = hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model
    keep = [False]

    def extract_config(model):
        config = extract(model)
        keep[0] = bool(config.get("keep_last_layer_irreps", False))
        return config

    def transfer_from_config(*args, keep_last_layer_irreps=False, **kwargs):
        return transfer(*args, keep_last_layer_irreps=keep[0], **kwargs)

    hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model = transfer_from_config, extract_config
    try:
        yield
    finally:
        hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model = transfer, extract


@contextmanager
def checkpoint_warning_filter():
    """Quiet only the legacy torch.load notices from MACE's and e3nn's loaders; other warnings stay visible."""
    with warnings.catch_warnings():
        for module in (r"e3nn\.o3\._wigner", r"mace\.calculators\.mace"):
            warnings.filterwarnings(
                "ignore", message=r"You are using `torch\.load` with `weights_only=False`",
                category=FutureWarning, module=module + r"$",
            )
        yield


LOGGER = logging.getLogger("fastmace.model.loading")

# cuequivariance's reduced-tensor-product module and its dict of basis products (0.12.0).
_BASIS_MODULE = "cuequivariance.group_theory.irreps_array.reduced_tensor_product"
_BASIS_DICT = "_cache_reduce_basis_product"


@functools.cache
def _note_basis_cache_missing(reason):
    """Log once per reason that the clearing was skipped (another cuequivariance version)."""
    LOGGER.debug("cuEquivariance basis caches not cleared: %s", reason)


def clear_cueq_basis_caches():
    """Empty cuequivariance's group-theory caches: the basis-product dict and the memoised bases and descriptors.

    MACE 0.3.16's ``O3_e3nn.clebsch_gordan`` computes Clebsch-Gordan coefficients in torch's default
    dtype, and cuequivariance 0.12.0 keeps what it builds from them (``_cache_reduce_basis_product``,
    a module dict, and ``functools`` caches such as the reduced bases and the symmetric-contraction
    descriptor), keyed without that dtype. After a float32 conversion, every later float64 conversion
    in the process took float32-rounded coefficients (up to 1.3e-8 in the basis; 2-T14). Only modules
    already imported are touched; nothing is imported here.
    """
    module = sys.modules.get(_BASIS_MODULE)
    if module is None:
        if any(name.startswith("cuequivariance.group_theory") for name in list(sys.modules)):
            _note_basis_cache_missing(f"{_BASIS_MODULE} is not part of this cuequivariance")
    else:
        store = getattr(module, _BASIS_DICT, None)
        if isinstance(store, dict):
            store.clear()
        else:
            _note_basis_cache_missing(f"{_BASIS_MODULE} has no dict {_BASIS_DICT}")
    for name, loaded in list(sys.modules.items()):
        if loaded is not None and name.startswith("cuequivariance.group_theory"):
            for value in list(vars(loaded).values()):
                if callable(getattr(value, "cache_clear", None)) and hasattr(value, "cache_info"):
                    value.cache_clear()


@contextmanager
def _fresh_cueq_basis_cache():
    """A conversion that depends only on its own dtype, as the first one of a process does (2-T14).

    The caches are cleared before (so the conversion does not take what another dtype left) and after
    (so it leaves nothing of its own dtype for a later one, including MACE's converter called directly).
    """
    clear_cueq_basis_caches()
    try:
        yield
    finally:
        clear_cueq_basis_caches()


@contextmanager
def capture_atomic_energies():
    """Keep an exact float64 copy of the E0 table of every model ``torch.load`` returns in the block.

    MACE's calculator converts the loaded model to its ``default_dtype``, which rounds a float64 E0
    table to float32 (Cl −12,522 eV by up to 0.5 meV). The float64 energy (SPEC §7.3) needs the
    checkpoint's table, so it is copied as the checkpoint is unpickled, before the conversion.
    Yields the list the copies go to. ``torch.load`` is wrapped for the block only.
    """
    import torch

    original = torch.load
    found = []

    def load(*args, **kwargs):
        value = original(*args, **kwargs)
        block = getattr(value, "atomic_energies_fn", None)
        if block is not None and torch.is_tensor(getattr(block, "atomic_energies", None)):
            found.append(block.atomic_energies.detach().to("cpu", torch.float64).clone())
        return value

    torch.load = load
    try:
        yield found
    finally:
        torch.load = original


# ------------------------------------------------------------------------ heads, embeddings, licence


def _plain(value):
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    return value


def check_embedding(model, entry, charge, spin_multiplicity):
    """The model's joint embedding (dict or None); checks it against the table and the charge/spin values."""
    specs = _plain(getattr(model, "embedding_specs", None)) or None
    expected = _plain(entry.checkpoint.embedding_specs) if entry is not None else None
    if entry is not None and specs != expected:
        raise ValueError(f"{entry.label}: the checkpoint's embedding_specs {specs} differ from the label table's "
                         f"{expected}")
    for name, value in (("total_charge", charge), ("total_spin", spin_multiplicity)):
        if value is None:
            continue
        if not specs or name not in specs:
            raise ValueError(f"the model has no {name} embedding; do not pass "
                             f"{'charge' if name == 'total_charge' else 'spin_multiplicity'}")
        spec = specs[name]
        index = int(value) + int(spec.get("offset", 0))
        if int(value) != value or not 0 <= index < int(spec["num_classes"]):
            low = -int(spec.get("offset", 0))
            raise ValueError(f"{name} {value!r} outside the embedding's range {low}..{low + spec['num_classes'] - 1}")
    return specs


def licence_message(entry, path, digest):
    if entry is None:
        return (f"{path.name} (SHA-256 {digest[:12]}…) is not a label of the FastMACE table; its licence is "
                "unknown, check it with the model's release")
    text = label_table.LICENCE_TEXT.get(entry.licence, entry.licence)
    return f"{entry.label} ({entry.checkpoint.file_name}) is distributed under the {text}"


def show_licence(entry, path, digest):
    """Warn once per checkpoint and process with the model's licence; True if it was shown now."""
    if digest in _LICENCES_SHOWN:
        return False
    _LICENCES_SHOWN.add(digest)
    warnings.warn(licence_message(entry, path, digest), ModelLicenceWarning, stacklevel=3)
    return True


# ------------------------------------------------------------------------ load


def load(source, head=None, *, checkpoint=None, backend="cueq", dtype="float64", device="cuda:0",
         cueq_conv_fusion=True, sha256=None, trusted=False, download=False, allow_naive_cueq=False,
         charge=None, spin_multiplicity=None, oeq_cache=None, environ=None):
    """Load a MACE model: a label (``"mh1-r2scan"``), an alias with a head (``"mh1"``, ``head=``), a path or a URL.

    ``backend``: ``"hybrid"`` (cuEq with OEq convolutions), ``"cueq"`` (plain
    cuEquivariance) or ``"e3nn"`` (MACE's reference modules). ``cueq_conv_fusion``
    (plain cuEq only): True fuses the convolution on any CUDA device (MACE PR
    1641, the research engine's default since 16 October 2026), False never, None
    MACE 0.3.16's rule. ``charge`` and ``spin_multiplicity`` are checked against
    the model's joint embedding (OMOL-0) and recorded; MACE's calculator reads them
    from ``atoms.info`` at evaluation time.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} is not one of {BACKENDS}")
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype!r} is not one of {DTYPES}")
    environ = os.environ if environ is None else environ
    if _is_url(source) and checkpoint is None and sha256 is None and not trusted:
        # Refused before anything is fetched: the file could only be refused after the download.
        raise CheckpointTrustError(
            f"{source} is not a label of the table and was not downloaded",
            why="MACE checkpoints are pickles that run code when loaded; a file from a URL needs a stated hash",
            next_step="Pass sha256= with the hash its publisher gives, or trusted=True.")
    resolved = resolve_checkpoint(source, head, checkpoint, environ, download, sha256)
    entry = resolved.label
    digest = sha256_of(resolved.path)
    trust = check_trust(resolved, digest, sha256, trusted)
    if backend == "hybrid":
        prepare_hybrid(oeq_cache)  # before the first MACE import
    if backend == "hybrid":
        check_cueq()
        check_oeq()
    elif backend == "cueq":
        check_cueq()
    fusion = None
    if backend == "cueq":
        fusion = _patches.conv_fusion_for(cueq_conv_fusion, device)

    model_type = entry.model_type if entry is not None else "MACE"
    kwargs = {"model_paths": str(resolved.path), "device": device, "default_dtype": dtype,
              "model_type": model_type}
    if entry is not None and entry.head is not None:
        kwargs["head"] = entry.head
    elif head is not None:
        kwargs["head"] = head
    if backend in ("hybrid", "cueq"):
        kwargs["enable_cueq"] = True
    if backend == "hybrid":
        kwargs["enable_oeq"] = True
    # The fusion rule acts on this conversion only; the rule in force before is back afterwards.
    scope = _patches.cueq_conv_fusion(cueq_conv_fusion, device) if backend == "cueq" else contextlib.nullcontext()
    # On the CPU, cuEq must build its naive kernels: with cuequivariance_ops_torch importable it would
    # pick the CUDA-only uniform_1d kernel, and the first forward pass would fail (decision after T06).
    naive = _device_type(device) == "cpu" and backend == "cueq"
    if naive:
        from fastmace.model import compile as compiling
    naive_scope = compiling.naive_cueq() if naive else contextlib.nullcontext()
    with _fresh_cueq_basis_cache(), checkpoint_warning_filter(), scope, naive_scope, \
            capture_atomic_energies() as tables:
        from mace.calculators import MACECalculator
        try:
            if backend == "hybrid":
                with hybrid_keeps_last_layer_irreps():
                    calculator = MACECalculator(**kwargs)
            else:
                calculator = MACECalculator(**kwargs)
        except (ImportError, OSError, TypeError, RuntimeError) as error:
            if backend == "e3nn":
                raise
            raise BackendError(f"{backend} initialization failed: {error}") from error
        patches = _patches.applied()  # what was in force for the conversion

    heads = tuple(getattr(calculator, "available_heads", ()) or ())
    wanted = kwargs.get("head")
    if wanted is None and len(heads) > 1:  # MACE would take a head named "default"; SPEC §7.7 has none
        raise ValueError(f"{resolved.path.name} has several heads; pass head= one of {', '.join(heads)}")
    if wanted is not None and calculator.head != wanted:  # MACE falls back to the last head with a warning only
        raise ValueError(f"{resolved.path.name} has no head {wanted!r}; its heads are {', '.join(heads)}")
    if backend in ("hybrid", "cueq") and not _cueq_enabled(calculator):
        raise BackendError("the constructed MACE calculator did not enable cuEquivariance")
    model = calculator.models[0]
    if backend == "hybrid":
        if not getattr(calculator, "_enable_oeq", False):
            raise BackendError("the constructed MACE calculator did not enable OpenEquivariance")
        if not any(type(module).__module__.startswith("openequivariance") for module in model.modules()):
            raise BackendError("the hybrid conversion produced no OpenEquivariance modules")
    fused = require_fused_cueq(model, allow_naive=allow_naive_cueq)
    specs = check_embedding(model, entry, charge, spin_multiplicity)
    show_licence(entry, resolved.path, digest)
    functional = entry.functional if entry is not None else None
    return LoadedModel(
        calculator=calculator, path=resolved.path, sha256=digest, trust=trust, checkpoint_source=resolved.source,
        label=entry.label if entry is not None else None, head=calculator.head, heads=heads, model_type=model_type,
        family=entry.family if entry is not None else None, functional=functional,
        licence=entry.licence if entry is not None else None, backend=backend, dtype=dtype, device=device,
        cueq_conv_fusion=fusion, fused_cueq=fused, embedding_specs=specs, charge=charge,
        spin_multiplicity=spin_multiplicity, patches=patches, atomic_energies=tables[0] if tables else None)


def load_pristine(source, head=None, *, sha256=None, trusted=False, download=False, charge=None,
                  spin_multiplicity=None, checkpoint=None, environ=None):
    """``fm.load`` (SPEC §7.1): the checkpoint as a pristine model, e3nn in float64 on the CPU.

    ``source`` is a label (``"mh1-r2scan"``), an alias with ``head=`` (``"mh1"``), a
    path or a URL. A multi-head checkpoint needs ``head=`` or a label that names one;
    otherwise ``ValueError`` lists its heads. A path or URL is unpickled only with a
    matching ``sha256=`` or ``trusted=True``; a label's file must match the table
    (SPEC §7.6). ``download=True`` fetches a missing file into the models cache and
    checks it before it is kept. The trust decision and the hash are in
    ``provenance()``. The plan converts the model to its backend and precision; the
    pristine model itself is never changed (``fm.ForceField``). Before MACE is
    imported, the OEq runtime is prepared where the hybrid is possible
    (``prepare_hybrid_if_possible``; the outcome is ``oeq_runtime``). The expandable-segments
    allocator is configured first (``configure_allocator``, a no-op if ``PYTORCH_CUDA_ALLOC_CONF`` is
    set): preparing the hybrid can touch CUDA, and the allocator reads the variable only once (5-T07).
    """
    from fastmace.model.compile import configure_allocator

    configure_allocator(environ=environ)
    oeq_runtime = prepare_hybrid_if_possible()
    loaded = load(source, head, checkpoint=checkpoint, backend="e3nn", dtype="float64", device="cpu",
                  sha256=sha256, trusted=trusted, download=download, charge=charge,
                  spin_multiplicity=spin_multiplicity, environ=environ)
    return dataclasses.replace(loaded, oeq_runtime=oeq_runtime)


__all__ = [
    "BACKENDS", "BackendError", "CheckpointTrustError", "HybridUnavailableError", "LoadedModel",
    "ModelLicenceWarning", "NaiveCueqError", "ResolvedCheckpoint", "capture_atomic_energies", "check_trust",
    "download_checkpoint", "fused_cueq_problems", "hybrid_keeps_last_layer_irreps", "load", "load_pristine",
    "prepare_hybrid_if_possible", "require_fused_cueq", "resolve_checkpoint", "sha256_of",
]
