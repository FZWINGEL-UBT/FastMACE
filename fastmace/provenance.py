"""Provenance (SPEC §13): the schema-versioned ``metadata.json`` and the 1a run.json and metadata blocks.

Phase 1b (plan task 1b-T10) writes ``metadata.json`` version 1 (``SCHEMA``, layout in
``docs/design/metadata-schema.md``): ``metadata_record`` builds it from the run's records (the
plan with its revision chain, the force-field identity, the gate record, the reference record,
restart points with their output chunk indices, the setup stages, the report) and
``write_metadata`` writes it atomically. Every §13 item has a key (``SPEC_13_KEYS``); items a later
phase fills are an explicit ``null``. Generators appear only as hashes and cursors: full states and
prefetched random batches live in the restart file (§10.6). Phase 2 (plan task 2-T17) fills its
keys: the fallbacks tried and the revision each became, the per-record self-checks, the always-on
treatments, the gate's device, reduced cell and resource steps, the runner and its child records
(``subprocess``), and the force field's own record with the strain path it bound (``force_field``).

Phase 1a's blocks stay as they were, so that ``main.py`` on FastMACE (1a-T17) writes the same files:

- ``run_record``: run.json (``protocol``, ``created_utc``, ``code_sha256``), from
  ``workflow.main``;
- ``model_block``: metadata.json's ``model`` block, from the benchmark scripts'
  ``common.model_info`` and the keys ``workflow.main`` adds to it, plus the checkpoint ``trust``
  (1b-T08); in the version 1 record it nests unchanged under ``"model"``;

plus FastMACE's own block (``fastmace_block``, under the key ``"fastmace"`` in both):
package version and git commit, the process-wide patches in force, the file-lock mode
and kernel availability.

Collection never imports MACE unless a model is passed (package versions are read from
the installed distributions), and never touches CUDA unless a CUDA device was asked for
and CUDA is available.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import fastmace
from fastmace import _patches, kernels
from fastmace.io import durable

__all__ = ["SCHEMA", "SPEC_13_KEYS", "collect", "cueq_versions", "environment", "fastmace_block",
           "generator_digest", "git_commit", "metadata_record", "model_block", "run_record", "sha256_file",
           "subprocess_block",
           "write_metadata"]

# Environment variables the research engine's hybrid launcher recorded as ``oeq_wrapper``
# (workflow.HYBRID_WRAPPER_VARIABLES plus three more). FastMACE prepares the OEq build in-process
# (``_patches.prepare_oeq_runtime``); the variables are recorded as they are, for comparison.
OEQ_ENVIRONMENT_VARIABLES = ("OEQ_JIT_EXTENSION", "TORCH_EXTENSIONS_DIR", "OEQ_NVCC_ATTEMPTS", "CUDA_HOME",
                             "MACE_COMPILE_CACHE", "OEQ_EXTENSION_SEED")
# metadata.json key -> (config argument of model_block, field): the rewrite and engine options,
# in the order workflow.main writes them.
OPTION_KEYS = (
    ("factor_species_radial", "rewrites", "factor_species"),
    ("fold_radial_cutoff", "rewrites", "fold_cutoff"),
    ("density_before_linear", "rewrites", "density_first"),
    ("reassociate_first_interaction", "rewrites", "reassociate_first"),
    ("joint_radial_input", "rewrites", "joint_radial_input"),
    ("fused_folded_backward", "rewrites", "fused_folded_backward"),
    ("segmented_first_sum", "rewrites", "segmented_first_sum"),
    ("gpu_neighbour_list", "neighbours", "gpu_neighbour_list"),
    ("block_linear", "rewrites", "block_linear"),
    ("grouped_linear", "rewrites", "grouped_linear"),
    ("merged_residual", "rewrites", "merged_residual"),
    ("species_skip", "rewrites", "species_skip"),
)
# Packages whose versions go into the FastMACE block (distribution names; missing ones are left out).
PACKAGES = ("torch", "numpy", "ase", "mace-torch", "e3nn", "triton", "cuequivariance", "cuequivariance-torch",
            "cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13", "openequivariance", "cuda-bindings",
            "jinja2", "filelock")
_CHUNK = 1 << 20
# git status runs once per process and checkout, never in the step loop; over WSL's 9P it can take
# seconds, and the record then says "unknown" (1b-T28).
GIT_STATUS_TIMEOUT_S = 2.0
_OBJECT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")  # SHA-1 or SHA-256
_DIRTY = {}  # checkout -> dirty (bool, None or "unknown")
_SOFTWARE = {}  # device -> the static part of fastmace_block


# ------------------------------------------------------------------------ helpers


def sha256_file(path):
    """SHA-256 of a file, as hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(*names):
    """Version of the first installed distribution among ``names``, else None."""
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _ops_version():
    """Version and CUDA build of the installed cuequivariance-ops-torch package (cu12 or cu13)."""
    for name in ("cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13"):
        found = _distribution_version(name)
        if found is not None:
            return f"{found} ({name.rsplit('-', 1)[1]})"
    return None


def cueq_versions():
    """The cuEquivariance versions of model_info: cuequivariance, cuequivariance_torch, cuequivariance_ops_torch."""
    return {"cuequivariance": _distribution_version("cuequivariance"),
            "cuequivariance_torch": _distribution_version("cuequivariance-torch"),
            "cuequivariance_ops_torch": _ops_version()}


def _mace_version():
    mace = sys.modules.get("mace")  # already imported with a model: its own version, as model_info reads it
    found = getattr(mace, "__version__", None) if mace is not None else None
    return found if found is not None else _distribution_version("mace-torch")


def _uses_cuda(device):
    """Whether ``device`` is a CUDA device and CUDA can be used, without initialising CUDA otherwise."""
    if not str(device).startswith("cuda"):
        return False
    import torch

    return torch.cuda.is_available()


def _host_path(text, base):
    """A path written in a ``.git`` file or ``commondir``: absolute, or relative to ``base``. A Windows
    path (``D:/...``, a worktree made by Windows git) maps to ``/mnt/d/...`` under WSL."""
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match and os.name != "nt":
        return Path("/mnt") / match.group(1).lower() / match.group(2).replace("\\", "/")
    path = Path(text)
    return path if path.is_absolute() else base / path


def _read_ref(git_dir, common, ref):
    """The object a ref names: a loose ref file (the worktree's own, then the common one), else
    ``packed-refs``; None if neither has it."""
    for root in (git_dir, common):
        loose = root / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and line[0] not in "#^":
                value, _, name = line.partition(" ")
                if name.strip() == ref:
                    return value
    return None


def head_commit(directory):
    """The commit ``HEAD`` of the checkout at ``directory`` points to, read from its ``.git`` files
    without starting git (a branch, a detached HEAD, loose or packed refs, a worktree's ``.git``
    file); None when it cannot be read (no checkout, another ref storage)."""
    directory = Path(directory)
    dot = directory / ".git"
    try:
        if dot.is_file():
            text = dot.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            git_dir = _host_path(text[len("gitdir:"):].strip(), directory)
        else:
            git_dir = dot
        if not (git_dir / "HEAD").is_file():
            return None
        common = git_dir
        if (git_dir / "commondir").is_file():
            common = _host_path((git_dir / "commondir").read_text(encoding="utf-8").strip(), git_dir)
        value = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        for _ in range(5):  # symbolic refs may point to symbolic refs
            if not value.startswith("ref:"):
                return value if _OBJECT_ID.fullmatch(value) else None
            value = _read_ref(git_dir, common, value[len("ref:"):].strip())
            if value is None:
                return None
    except OSError:
        return None
    return None


def _dirty(directory):
    """Whether tracked files differ from HEAD: ``git status`` once per process and checkout, with a
    ``GIT_STATUS_TIMEOUT_S`` timeout. "unknown" when it runs out or git refuses (for example git's
    dubious-ownership check on NFS, which this never overrides: ``status`` on a foreign repository
    can run its configured programs); None without a git executable."""
    key = str(directory)
    if key not in _DIRTY:
        try:
            status = subprocess.run(["git", "-C", key, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=GIT_STATUS_TIMEOUT_S)
        except subprocess.TimeoutExpired:  # git status over WSL's 9P takes seconds (1b-T28)
            _DIRTY[key] = "unknown"
        except (OSError, subprocess.SubprocessError):  # no git executable
            _DIRTY[key] = None
        else:
            _DIRTY[key] = bool(status.stdout.strip()) if status.returncode == 0 else "unknown"
    return _DIRTY[key]


def git_commit(directory=None):
    """(commit, dirty) of the git checkout holding ``directory`` (default: FastMACE's own), else (None, None).

    The commit is read from the ``.git`` files (``head_commit``); a copy without ``.git``, such as
    the NAS checkout, gives the content of its ``.source_commit`` and ``dirty`` None. ``dirty``
    counts changes to tracked files only; the check runs once per process and checkout with a
    short timeout and is "unknown" when that runs out. Installed from a wheel, the package has no
    checkout around it and both are None.
    """
    directory = Path(fastmace.__file__).resolve().parent.parent if directory is None else Path(directory)
    if (directory / ".git").exists():  # a directory in a worktree has a .git file, a checkout a directory
        commit = head_commit(directory)
        return (None, None) if commit is None else (commit, _dirty(directory))
    source = directory / ".source_commit"
    if source.is_file():
        text = source.read_text(encoding="utf-8").strip()
        return (text or None), None
    return None, None


def _resolved(name, value):
    """A resolved Toggle ("on"/"off") or bool as bool; "auto" is refused (the planner resolves it, 1a-T15)."""
    if isinstance(value, bool):
        return value
    if value == "on":
        return True
    if value == "off":
        return False
    raise ValueError(f"{name} is {value!r}: provenance records resolved options; resolve 'auto' first")


def convolution_summary(model):
    """How the model's interactions convolve, for the run metadata: conv_tp classes and conv_fusion markers."""
    interactions = list(getattr(model, "interactions", ()))
    return {"conv_tp": sorted({type(interaction.conv_tp).__name__ for interaction in interactions
                               if hasattr(interaction, "conv_tp")}),
            "conv_fusion": [str(getattr(interaction, "conv_fusion", False)) for interaction in interactions]}


# ------------------------------------------------------------------------ blocks


def environment(device="cuda:0"):
    """The environment keys of model_info: torch, cuda_version, gpu, threads, mace_torch, ase, host."""
    import torch

    return {
        "torch": str(torch.__version__),  # a str subclass
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if _uses_cuda(device) else None,
        "threads": torch.get_num_threads(),
        "mace_torch": _mace_version(),
        "ase": _distribution_version("ase"),
        "host": platform.node(),
    }


def _software(device):
    """The parts of ``fastmace_block`` that do not change within a process: git commit, kernel
    probes, package versions and the GPU (1b-T28: about 0.13 s from ext4, seconds over 9P)."""
    commit, dirty = git_commit()
    probes = (("triton", kernels.triton_available), ("nvrtc", kernels.nvrtc_available),
              ("streaming", kernels.streaming_available))
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        found = _distribution_version(name)
        if found is not None:
            versions[name] = found
    gpu = None
    if _uses_cuda(device):
        import torch

        major, minor = torch.cuda.get_device_capability(device)
        gpu = {"name": torch.cuda.get_device_name(device), "compute_capability": f"{major}.{minor}"}
    return {"git_commit": commit, "git_dirty": dirty,
            "kernels": {name: dataclasses.asdict(probe()) for name, probe in probes}, "versions": versions, "gpu": gpu}


def fastmace_block(device="cuda:0"):
    """FastMACE's additions: version, git commit, patches in force, file locks, kernels, versions, GPU.

    The static part is built once per process and device (``_software``); the patches in force and
    the runtime are read at every call. Each call returns a new dictionary."""
    key = str(device)
    if key not in _SOFTWARE:
        _SOFTWARE[key] = _software(device)
    static = copy.deepcopy(_SOFTWARE[key])
    return {
        "version": fastmace.__version__,
        "git_commit": static["git_commit"],
        "git_dirty": static["git_dirty"],
        "patches": _patches.provenance(),
        "runtime": dataclasses.asdict(fastmace.runtime_info()),
        "kernels": static["kernels"],
        "versions": static["versions"],
        "gpu": static["gpu"],
    }


def collect(device="cuda:0"):
    """What can be collected without a model: ``{"environment": ..., "fastmace": ...}``."""
    return {"environment": environment(device), "fastmace": fastmace_block(device)}


def model_block(loaded, *, rewrites, engine, neighbours, polar, compute_stress, dispersion=False,
                auto_options_off=None):
    """metadata.json's ``model`` block for ``loaded`` (a ``fastmace.model.loading.LoadedModel``).

    ``rewrites``, ``engine``, ``neighbours`` and ``polar`` are the config objects with every
    Toggle resolved to "on" or "off". ``auto_options_off`` maps the options that auto turned
    off to the reason (``adapt_options_to_model``); the key is written only when it is not empty,
    as today. The keys and their order follow ``common.model_info`` and ``workflow.main`` at
    research-engine-baseline-20261017, without the benchmark-only ``setting`` and ``fmax_eV_A``.
    """
    cueq = loaded.backend in ("cueq", "hybrid")
    oeq = loaded.backend == "hybrid"
    backend = {"hybrid": "hybrid", "cueq": "cuequivariance"}.get(loaded.backend, "e3nn")
    entry = _label_entry(loaded)
    block = {
        "model": loaded.label,
        "mace_mp_name": entry.checkpoint.name if entry is not None else None,
        "checkpoint": str(loaded.path),
        "heads": list(loaded.heads),
        "head": loaded.head,
        "dtype": loaded.dtype,
        "device": loaded.device,
    }
    env = environment(loaded.device)
    block.update({key: env[key] for key in ("torch", "cuda_version", "gpu", "threads")})
    block["dispersion_d3"] = bool(dispersion)
    block.update({key: env[key] for key in ("mace_torch", "ase", "host")})
    block.update({"enable_cueq": cueq, "enable_oeq": oeq, "equivariance_backend": backend})
    block.update(cueq_versions() if cueq else dict.fromkeys(("cuequivariance", "cuequivariance_torch",
                                                              "cuequivariance_ops_torch")))
    block["openequivariance"] = _distribution_version("openequivariance", "OpenEquivariance") if oeq else None
    if auto_options_off:
        block["auto_options_off"] = dict(auto_options_off)
    block["compute_stress"] = bool(compute_stress)
    block["single_graph_strain"] = _resolved("single_graph_strain", rewrites.single_graph_strain)
    block["cueq_kernels"] = dict(loaded.fused_cueq) if loaded.fused_cueq is not None else None
    block["cueq_conv_fusion"] = _resolved("cueq_conv_fusion", engine.cueq_conv_fusion)
    block["file_locks"] = fastmace.runtime_info().file_locks  # "soft": no NFS lock requests
    block["convolution"] = convolution_summary(loaded.model)
    block["checkpoint_sha256"] = loaded.sha256
    block["trust"] = loaded.trust  # "label", "sha256" or "trusted" (SPEC §7.6, 1b-T08)
    block["md_backend"] = loaded.backend
    configs = {"rewrites": rewrites, "neighbours": neighbours}
    for key, config, name in OPTION_KEYS:
        block[key] = _resolved(name, getattr(configs[config], name))
    block["polar_uvu"] = "on" if rewrites.polar_uvu else "off"
    block["polar_fukui_regularisation"] = float(polar.fukui_regularisation)
    block["polar_dipole_cap"] = list(polar.dipole_cap) if polar.dipole_cap is not None else None
    block["streaming_conv"] = _resolved("streaming_conv", rewrites.streaming_conv)
    block["fused_radial_hidden"] = _resolved("fused_radial_hidden", rewrites.fused_radial_hidden)
    if oeq:
        block["oeq_wrapper"] = {name: os.environ.get(name) for name in OEQ_ENVIRONMENT_VARIABLES}
    block["fastmace"] = fastmace_block(loaded.device)
    return block


def _label_entry(loaded):
    if loaded.label is None:
        return None
    from fastmace.model import labels

    try:
        return labels.find(loaded.label, loaded.head)
    except (KeyError, ValueError):
        return None


def run_record(protocol, code_files=(), root=None, created_utc=None):
    """run.json: ``protocol`` (paths as strings), ``created_utc``, ``code_sha256`` and the FastMACE block.

    ``code_sha256`` maps each of ``code_files`` (relative to ``root`` when given) to its SHA-256,
    for the files that exist, as ``workflow.main`` does for its ``CODE_FILES``.
    """
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in dict(protocol).items()}
    code = {}
    for path in code_files:
        path = Path(path)
        if path.is_file():
            name = path.relative_to(root).as_posix() if root is not None else str(path)
            code[name] = sha256_file(path)
    created = created_utc if created_utc is not None else datetime.now(timezone.utc).isoformat()
    return {"protocol": protocol, "created_utc": created, "code_sha256": code, "fastmace": fastmace_block("cpu")}


# ------------------------------------------------------------------------ metadata.json version 1

SCHEMA = "fastmace.metadata/1"
# Every item of SPEC §13, then what plan task 1b-T10 adds: (item, dotted key path in the record).
# docs/design/metadata-schema.md describes each key; phase 2 freezes and validates the schema.
SPEC_13_KEYS = (
    ("model path", "checkpoint.path"),
    ("model SHA-256", "checkpoint.sha256"),
    ("model head", "checkpoint.head"),
    ("model licence", "checkpoint.licence"),
    ("force-field identity", "force_field_identity"),
    ("reference record", "reference_record"),
    ("plan tier", "plan.tier"),
    ("rewrites on/off", "plan.rewrites"),
    ("rewrite reasons", "plan.decisions"),
    ("force precision", "precision.force"),
    ("integration precision", "precision.integration"),
    ("TF32 flag", "precision.tf32"),
    ("backend", "plan.backend"),
    ("convolution fused (plain backend)", "plan.conv_fusion"),
    ("fallbacks tried", "plan.fallbacks_tried"),
    ("file-lock mode", "runtime.file_locks"),
    ("CUDA driver cache", "runtime.cuda_driver_cache"),
    ("package versions", "software.versions"),
    ("kernels", "software.kernels"),
    ("OEq extension build id", "software.oeq_extension"),
    ("Triton version", "software.triton"),
    ("GPU name", "hardware.gpu"),
    ("driver", "hardware.driver"),
    ("compute capability", "hardware.compute_capability"),
    ("seeds", "random.seed"),
    ("generator-state hashes", "random.generators"),
    ("cursors", "random.cursors"),
    ("integrator parameters", "integrator"),
    ("schedule", "schedule"),
    ("gate status", "gate.status"),
    ("self-check results", "gate.selfchecks"),
    ("re-capture counts", "counters.recaptures"),
    ("neighbour-list rebuilds, build times, skins", "counters.neighbour_lists"),
    ("overflows redone", "counters.overflows_redone"),
    ("tier switches", "counters.tier_switches"),
    ("MC counters", "counters.mc"),
    ("restart points with chunk indices", "restart_points"),
    ("timing summary", "timing"),
    ("package version", "code.version"),
    ("git hash", "code.git_commit"),
    # plan task 1b-T10
    ("schema version", "schema"),
    ("plan revision chain", "plan_revisions"),
    ("setup-time breakdown", "timing.setup_s"),
    ("run report (1b-T15)", "report"),
    ("1a model block", "model"),
    ("checkpoint trust (1b-T08)", "checkpoint.trust"),
    ("float64 energy accumulation (1b-T03)", "precision.float64_energy"),
    # plan task 2-T17: what phase 2 records
    ("plan runner (2-T07)", "plan.runner"),
    ("per-record self-checks (2-T04)", "gate.selfchecks"),
    ("always-on treatments (2-T04)", "software.treatments"),
    ("gate reference device (2-T05)", "gate.device"),
    ("gate reduced cell (2-T05)", "gate.reduced_cell"),
    ("gate resource steps (2-T05)", "gate.steps"),
    ("subprocess runner and its children (2-T07)", "subprocess"),
    ("strain path bound by the force field (2-T08b)", "force_field.strain_path"),
    ("compiled with dynamic edges (2-T13)", "plan.engine.dynamic_edges"),
    ("reproducible mode (2-T16)", "plan.workload.reproducible"),
)
# Integration is always float64 (SPEC §7.3); only the force precision is a choice.
INTEGRATION_PRECISION = "float64"
# D1: the gate is opt-in. Without a gate record the run says why it did not run.
GATE_SKIPPED = {"status": "skipped", "reason": 'opt-in: check="gate"', "selfchecks": None}
# Keys every gate record has (2-T05's resource fields): null where no gate ran, ``steps`` then [].
GATE_RECORD_KEYS = ("selfchecks", "device", "reduced_cell", "steps")
CHECKPOINT_KEYS = ("path", "sha256", "head", "licence", "label", "trust")
# The subprocess runner's block (2-T07, ``engine.subprocess.SubprocessRunner.record``) and its last child.
SUBPROCESS_KEYS = ("python", "reason", "directory", "children", "seconds", "failed", "last")
SUBPROCESS_CHILD_KEYS = ("action", "stage", "pid", "exit_code", "seconds", "steps")
# Counters the caller fills, and those a later phase fills (tier switches: phase 5, §9.6; MC: phase 3).
COUNTER_KEYS = ("recaptures", "neighbour_lists", "overflows_redone")
LATER_COUNTER_KEYS = ("tier_switches", "mc")
# A restart point: its step and the output chunk indices of io.restart.OUTPUT_KEYS (1b-T06).
RESTART_POINT_KEYS = ("step", "output_chunk", "output_rows", "output_frames")
# Arrays up to this size (a 3x3 tensor) go in as lists; anything larger is a state or a batch.
_LARGEST_ARRAY = 9
_IN_RESTART = "full generator states and random batches belong in the restart file (SPEC §10.6), not in metadata.json"


def _array_module(value):
    """"torch" or "numpy" for a tensor, array or generator of either (without importing them), else None."""
    module = type(value).__module__.split(".", 1)[0]
    return module if module in ("torch", "numpy") else None


def _plain(value, where):
    """``value`` as JSON-safe data; ``ValueError`` (naming ``where``) for states, batches and bytes."""
    if value is None:
        return value
    if isinstance(value, str):
        return str(value)  # a Reason as plain text
    if _array_module(value) is None and isinstance(value, (bool, int, float)):
        return bool(value) if isinstance(value, bool) else int(value) if isinstance(value, int) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: key {key!r} is not a string")
            out[key] = _plain(item, f"{where}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item, where) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{where}: raw bytes; {_IN_RESTART}")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value), where)
    module = _array_module(value)
    if module is not None and type(value).__name__ in ("Generator", "RandomState", "BitGenerator") or (
            module == "numpy" and hasattr(value, "random_raw")):
        raise ValueError(f"{where}: a generator; record it under generators= (a hash); {_IN_RESTART}")
    if module is not None and hasattr(value, "tolist"):
        size = value.numel() if module == "torch" else value.size
        if size > _LARGEST_ARRAY:
            raise ValueError(f"{where}: an array of {size} values; {_IN_RESTART}")
        if module == "torch":
            value = value.detach().cpu()
        return value.tolist()
    raise ValueError(f"{where}: {type(value).__name__} is not JSON-safe")


def _numpy_state_bytes(state):
    """Canonical bytes of a numpy bit-generator state (dicts of ints, strings and arrays)."""
    return json.dumps(state, sort_keys=True, default=lambda item: item.tolist()).encode()


def generator_digest(generator):
    """SHA-256 (hex) of a generator's state: what metadata.json records instead of the state.

    Takes a ``torch.Generator`` (its ``get_state()``), a state tensor or array (a CUDA generator's
    state is 16 bytes, a CPU one 5,056), a numpy ``Generator``, ``BitGenerator`` or ``RandomState``,
    raw bytes, or a digest already (64 lowercase hex digits, returned as it is).
    """
    if isinstance(generator, str):
        if len(generator) == 64 and all(char in "0123456789abcdef" for char in generator):
            return generator
        raise ValueError(f"generator digest {generator!r} is not 64 lowercase hex digits")
    module = _array_module(generator)
    if isinstance(generator, (bytes, bytearray, memoryview)):
        data = bytes(generator)
    elif module == "torch" and hasattr(generator, "get_state"):
        data = generator.get_state().numpy().tobytes()
    elif module == "torch" and hasattr(generator, "numpy"):
        data = generator.detach().cpu().contiguous().numpy().tobytes()
    elif module == "numpy" and hasattr(generator, "bit_generator"):
        data = _numpy_state_bytes(generator.bit_generator.state)
    elif module == "numpy" and hasattr(generator, "random_raw"):
        data = _numpy_state_bytes(generator.state)
    elif module == "numpy" and hasattr(generator, "get_state"):
        data = _numpy_state_bytes(generator.get_state(legacy=False))
    elif module == "numpy" and hasattr(generator, "tobytes"):
        data = generator.tobytes()
    else:
        raise ValueError(f"cannot hash a generator state of type {type(generator).__name__}")
    return hashlib.sha256(data).hexdigest()


def _record(value, where):
    """A record given as a mapping or as an object with ``to_metadata()`` or ``to_dict()``, as a dict."""
    for method in ("to_metadata", "to_dict"):
        if not isinstance(value, Mapping) and callable(getattr(value, method, None)):
            value = getattr(value, method)()
            break
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected a mapping or a record with to_metadata() or to_dict(), "
                         f"got {type(value).__name__}")
    return _plain(value, where)


def _checkpoint(checkpoint):
    if checkpoint is None:
        return dict.fromkeys(CHECKPOINT_KEYS)
    source = checkpoint.provenance() if callable(getattr(checkpoint, "provenance", None)) else checkpoint
    source = _record(source, "checkpoint")
    return {key: source.get(key) for key in CHECKPOINT_KEYS}


def _identity(identity):
    if identity is None:
        return None
    if isinstance(identity, Mapping):
        from fastmace.state import ForceFieldIdentity

        identity = ForceFieldIdentity.from_dict(identity)
    return {**_plain(identity.to_dict(), "force_field_identity"), "digest": identity.digest}


def _reference(reference_record, identity):
    if reference_record is None:
        return None
    record = (_record(reference_record, "reference_record")
              if isinstance(reference_record, Mapping) or hasattr(reference_record, "to_dict")
              else {"id": getattr(reference_record, "id", None)})
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError(f"reference_record needs an 'id' (SPEC §12.1), got {record!r}")
    expected = None if identity is None else identity.get("reference_record_id")
    if expected is not None and expected != record["id"]:
        raise ValueError(f"reference record {record['id']!r} is not the force-field identity's {expected!r}")
    return record


def _reason_fields(reason):
    return str(reason), getattr(reason, "code", None), getattr(reason, "rule", None)


def _plan(plan):
    """The active plan revision: tier, resolved options, every decision with its reason and code."""
    if plan is None:
        return None

    def config(part, name):
        return None if part is None else _plain(dataclasses.asdict(part), f"plan.{name}")

    decisions = []
    for key, choice, reason in plan.decisions:
        text, code, rule = _reason_fields(reason)
        decisions.append({"key": key, "choice": _plain(choice, f"plan.decisions.{key}"), "reason": text,
                          "code": code, "rule": rule})
    return {
        "revision": plan.revision,
        "tier": plan.tier,
        "reason": str(plan.reason),
        "step": plan.step,
        "description": plan.describe(),
        "backend": plan.engine.backend,
        "conv_fusion": plan.engine.cueq_conv_fusion,
        "rewrites": config(plan.rewrites, "rewrites"),
        "engine": config(plan.engine, "engine"),
        "neighbours": config(plan.neighbours, "neighbours"),
        "polar": config(plan.polar, "polar"),
        "integrator": config(plan.integrator, "integrator"),
        "decisions": decisions,
        "requested": _plain([list(entry) for entry in plan.requested], "plan.requested"),
        "auto_options_off": _plain([list(entry) for entry in plan.auto_options_off], "plan.auto_options_off"),
        "superset": list(plan.superset),
        "workload": config(plan.workload, "workload"),
        "software": None if plan.software is None else _plain(plan.software.to_dict(), "plan.software"),
        "allowances": list(plan.allowances),
        "fallbacks_tried": _plain(list(plan.fallbacks), "plan.fallbacks"),  # TriedCandidate, §9.4 (2-T15)
        "predicted_memory_bytes": plan.predicted_memory_bytes,
        "runner": plan.runner,
    }


def _plan_revisions(plan):
    """The revision chain, oldest first, the active revision last (SPEC §9.5)."""
    if plan is None:
        return []
    # ``fallback``: the tried candidate that became this revision (§9.4, 2-T15), else null.
    activated = {tried.revision: tried for tried in plan.fallbacks if tried.outcome == "activated"}
    return [{"revision": entry.revision, "tier": entry.tier, "step": entry.step, "reason": str(entry.reason),
             "description": entry.describe(), "runner": entry.runner,
             "fallback": _plain(activated.get(entry.revision), "plan.fallbacks")}
            for entry in (*plan.chain(), plan)]


def _gate(gate, plan):
    """The gate record with ``GATE_RECORD_KEYS``; its ``selfchecks`` default to the plan's (2-T04)."""
    if isinstance(gate, Mapping) or gate is None:
        record = dict(GATE_SKIPPED) if gate is None else _plain(gate, "gate")
    else:
        record = _record(gate, "gate")
    if "status" not in record:
        raise ValueError(f"gate record without a status (SPEC §12.3): {record!r}")
    for key in GATE_RECORD_KEYS:
        record.setdefault(key, None)
    if record["steps"] is None:
        record["steps"] = []
    if record["selfchecks"] is None and plan is not None:
        record["selfchecks"] = _plain(list(plan.selfchecks), "plan.selfchecks")
    return record


def _treatments(plan):
    """The always-on treatments' decisions (2-T04, ``rewrites.versions``); null without a plan (no MACE import)."""
    if plan is None:
        return None
    from fastmace.rewrites import versions

    return [_plain(item, "software.treatments") for item in versions.treatments()]


def _force_field(force_field):
    """The engine's force field record (its ``info()``): backend, rewrites applied, the strain path bound."""
    if force_field is None:
        return None
    if not isinstance(force_field, Mapping) and callable(getattr(force_field, "info", None)):
        force_field = force_field.info()
    if not isinstance(force_field, Mapping):
        raise ValueError(f"force_field: expected a mapping or a force field with info(), got "
                         f"{type(force_field).__name__}")
    return _plain(force_field, "force_field")


def subprocess_block(runner):
    """The ``subprocess`` block of ``metadata.json`` (2-T07): ``SUBPROCESS_KEYS``, checked.

    ``runner`` is a ``SubprocessRunner`` (its ``record()``) or the mapping it gives; ``last``, the
    last child, is null or has ``SUBPROCESS_CHILD_KEYS``. ``ValueError`` for a missing or unknown key.
    """
    record = _plain(runner.record() if callable(getattr(runner, "record", None)) else runner, "subprocess")
    if not isinstance(record, dict) or set(record) != set(SUBPROCESS_KEYS):
        found = sorted(record) if isinstance(record, dict) else type(record).__name__
        raise ValueError(f"subprocess block: expected the keys {SUBPROCESS_KEYS}, got {found}")
    last = record["last"]
    if last is not None and (not isinstance(last, dict) or set(last) != set(SUBPROCESS_CHILD_KEYS)):
        found = sorted(last) if isinstance(last, dict) else type(last).__name__
        raise ValueError(f"subprocess block: last child: expected the keys {SUBPROCESS_CHILD_KEYS}, got {found}")
    return {key: record[key] for key in SUBPROCESS_KEYS}


def _precision(plan, identity):
    if plan is not None:
        force, tf32, energy = plan.precision.dtype, plan.precision.tf32, plan.precision.float64_energy
    else:
        force, tf32, energy = (None if identity is None else identity["force_precision"]), None, None
    return {"force": force, "integration": INTEGRATION_PRECISION, "tf32": tf32, "float64_energy": energy}


def _hardware(device):
    hardware = {"gpu": None, "driver": None, "compute_capability": None}
    if _uses_cuda(device):
        import torch

        major, minor = torch.cuda.get_device_capability(device)
        hardware.update(gpu=torch.cuda.get_device_name(device), compute_capability=f"{major}.{minor}")
        try:
            version = torch._C._cuda_getDriverVersion()  # e.g. 12080 for CUDA 12.8
            hardware["driver"] = f"CUDA {version // 1000}.{version % 1000 // 10}"
        except (AttributeError, RuntimeError):
            pass
    return hardware


def _counters(counters, engine_info):
    given = dict(counters or {})
    for key in given:
        if key in LATER_COUNTER_KEYS:
            raise ValueError(f"counters: {key!r} is filled by a later phase (tier switches: phase 5, MC: phase 3)")
        if key not in COUNTER_KEYS:
            raise ValueError(f"counters: unknown key {key!r}; 1b records {COUNTER_KEYS}")
    out = {key: _plain(given.get(key), f"counters.{key}") for key in COUNTER_KEYS}
    out.update(dict.fromkeys(LATER_COUNTER_KEYS))
    out["engine"] = None if engine_info is None else _plain(engine_info, "counters.engine")
    return out


def _restart_points(points):
    out = []
    for index, point in enumerate(points):
        point = _plain(point, f"restart_points[{index}]")
        missing = [key for key in RESTART_POINT_KEYS if key not in point]
        if missing:
            raise ValueError(f"restart point {index} misses {', '.join(missing)} (SPEC §10.6)")
        for key in RESTART_POINT_KEYS:
            if isinstance(point[key], bool) or not isinstance(point[key], int) or point[key] < 0:
                raise ValueError(f"restart point {index}: {key} must be an integer >= 0, got {point[key]!r}")
        out.append({**{key: point[key] for key in RESTART_POINT_KEYS},
                    **{key: value for key, value in point.items() if key not in RESTART_POINT_KEYS}})
    return out


def _timing(setup, timing):
    stages = {}
    for stage, seconds in dict(setup or {}).items():
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds >= 0:
            raise ValueError(f"setup stage {stage!r}: seconds must be a number >= 0, got {seconds!r}")
        stages[str(stage)] = float(seconds)
    rest = _plain(dict(timing or {}), "timing")
    for key in ("setup_s", "setup_total_s"):
        if key in rest:
            raise ValueError(f"timing: {key!r} comes from setup=")
    return {"setup_s": stages, "setup_total_s": math.fsum(stages.values()) if stages else None, **rest}


def metadata_record(*, model=None, checkpoint=None, identity=None, reference_record=None, plan=None, seed=None,
                    generators=None, cursors=None, integrator=None, schedule=None, gate=None, counters=None,
                    engine_info=None, restart_points=(), setup=None, timing=None, report=None, created_utc=None,
                    device="cpu", force_field=None, subprocess=None):
    """``metadata.json`` version 1 (``SCHEMA``) of a run; every input is optional.

    - ``model``: the 1a ``model_block`` (nested unchanged under ``"model"``);
    - ``checkpoint``: a ``LoadedModel`` (its ``provenance()``) or a mapping with ``CHECKPOINT_KEYS``;
    - ``identity``: a ``state.ForceFieldIdentity`` or its ``to_dict()``; recorded with its digest;
    - ``reference_record``: the §12.1 record, a mapping or a record with ``to_dict()`` or ``id``;
      its id must equal the identity's ``reference_record_id`` when both are given;
    - ``plan``: the active ``planner.plan.Plan``; its revision chain goes to ``plan_revisions``;
    - ``seed``; ``generators``: {name: generator or state}, recorded as ``generator_digest``;
      ``cursors``: {stream: position} of the prefetched batches;
    - ``integrator``: the integrator parameters (default: the plan's ``IntegratorConfig``);
      ``schedule``: a mapping or a record with ``to_restart()`` (1b-T07's protocol);
    - ``gate``: the gate record (status and self-check results), a mapping or a record with
      ``to_dict()``; default ``GATE_SKIPPED`` (D1: the gate is opt-in);
    - ``counters``: ``COUNTER_KEYS``; ``engine_info``: the engine's ``info()`` as it is;
    - ``restart_points``: mappings with ``RESTART_POINT_KEYS`` (T06's chunk indices);
    - ``setup``: {stage: seconds} with §12.5's stage names; ``timing``: the timing summary;
    - ``report``: the run report (1b-T15), a mapping or a record with ``to_metadata()``;
    - ``force_field``: the engine's force field (its ``info()``) or a mapping: the rewrites applied
      and the strain path bound (2-T08b);
    - ``subprocess``: the subprocess runner (2-T07) or its ``record()``, checked by
      ``subprocess_block``; null for a plan that runs in this process.

    The gate's ``selfchecks`` default to the plan's per-record self-checks (2-T04); with a plan,
    the always-on treatments (2-T04) go to ``software.treatments``.

    ``device`` is where the run ran: CUDA is queried only for a CUDA device. A full generator
    state or random batch anywhere raises ``ValueError``.
    """
    block = fastmace_block(device)
    identity_record = _identity(identity)
    plan_record = _plan(plan)
    oeq = plan is not None and plan.engine.backend == "hybrid"
    gate_record = _gate(gate, plan)
    if schedule is not None and not isinstance(schedule, Mapping) and callable(getattr(schedule, "to_restart", None)):
        schedule = schedule.to_restart()
    if integrator is None:
        integrator_record = None if plan_record is None else plan_record["integrator"]
    else:
        integrator_record = _plain(integrator, "integrator")
    return {
        "schema": SCHEMA,
        "created_utc": created_utc if created_utc is not None else datetime.now(timezone.utc).isoformat(),
        "code": {"version": block["version"], "git_commit": block["git_commit"], "git_dirty": block["git_dirty"]},
        "checkpoint": _checkpoint(checkpoint),
        "model": None if model is None else _plain(model, "model"),
        "force_field_identity": identity_record,
        "reference_record": _reference(reference_record, identity_record),
        "plan": plan_record,
        "plan_revisions": _plan_revisions(plan),
        "precision": _precision(plan, identity_record),
        "runtime": block["runtime"],
        "software": {"versions": block["versions"], "kernels": block["kernels"], "patches": block["patches"],
                     "triton": block["versions"].get("triton"),
                     "oeq_extension": ({name: os.environ.get(name) for name in OEQ_ENVIRONMENT_VARIABLES}
                                       if oeq else None),
                     "treatments": _treatments(plan)},
        "hardware": _hardware(device),
        "random": {"seed": _plain(seed, "seed"),
                   "generators": None if generators is None else {
                       str(name): generator_digest(state) for name, state in dict(generators).items()},
                   "cursors": None if cursors is None else _plain(cursors, "cursors")},
        "integrator": integrator_record,
        "schedule": None if schedule is None else _plain(schedule, "schedule"),
        "gate": gate_record,
        "counters": _counters(counters, engine_info),
        "restart_points": _restart_points(restart_points),
        "timing": _timing(setup, timing),
        "report": None if report is None else _record(report, "report"),
        "force_field": _force_field(force_field),
        "subprocess": None if subprocess is None else subprocess_block(subprocess),
    }


def write_metadata(path, record):
    """Write ``record`` (a ``metadata_record``) to ``path`` durably and atomically (``io.durable.atomic_write``).

    Refuses a record of another schema and values JSON cannot hold (NaN, infinity); the file
    at ``path`` is then left as it was. Returns ``path`` as a ``Path``.
    """
    path = Path(path)
    if not isinstance(record, Mapping) or record.get("schema") != SCHEMA:
        raise ValueError(f"write_metadata writes {SCHEMA!r} records (metadata_record), got schema "
                         f"{record.get('schema') if isinstance(record, Mapping) else type(record).__name__!r}")
    try:
        text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata.json is not JSON-safe: {error}") from None
    return durable.atomic_write(path, text)
