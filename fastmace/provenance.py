"""Minimal provenance (SPEC §13, plan task 1a-T16): today's run.json and metadata.json blocks.

Phase 1a reproduces what the research engine writes, so that ``main.py`` on FastMACE
(1a-T17) writes the same files:

- ``run_record``: run.json (``protocol``, ``created_utc``, ``code_sha256``), from
  ``workflow.main``;
- ``model_block``: metadata.json's ``model`` block, from the benchmark scripts'
  ``common.model_info`` and the keys ``workflow.main`` adds to it;

plus FastMACE's own block (``fastmace_block``, under the key ``"fastmace"`` in both):
package version and git commit, the process-wide patches in force, the file-lock mode
and kernel availability. The schema-versioned metadata of SPEC §13 comes in phase 1b.

Collection never imports MACE unless a model is passed (package versions are read from
the installed distributions), and never touches CUDA unless a CUDA device was asked for
and CUDA is available.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace
from fastmace import _patches, kernels

__all__ = ["collect", "cueq_versions", "environment", "fastmace_block", "git_commit", "model_block",
           "run_record", "sha256_file"]

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


def git_commit(directory=None):
    """(commit, dirty) of the git checkout holding ``directory`` (default: FastMACE's own), else (None, None).

    ``dirty`` counts changes to tracked files only. Installed from a wheel, the package has no
    checkout around it and both are None.
    """
    directory = Path(fastmace.__file__).resolve().parent.parent if directory is None else Path(directory)
    if not (directory / ".git").exists():  # a directory in a worktree has a .git file, a checkout a directory
        return None, None
    try:
        head = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30)
        if head.returncode != 0:
            return None, None
        status = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):  # no git executable, or it hangs
        return None, None
    return head.stdout.strip(), (bool(status.stdout.strip()) if status.returncode == 0 else None)


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


def fastmace_block(device="cuda:0"):
    """FastMACE's additions: version, git commit, patches in force, file locks, kernels, versions, GPU."""
    commit, dirty = git_commit()
    probes = (("triton", kernels.triton_available), ("nvrtc", kernels.nvrtc_available),
              ("streaming", kernels.streaming_available))
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        found = _distribution_version(name)
        if found is not None:
            versions[name] = found
    block = {
        "version": fastmace.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "patches": _patches.provenance(),
        "runtime": dataclasses.asdict(fastmace.runtime_info()),
        "kernels": {name: dataclasses.asdict(probe()) for name, probe in probes},
        "versions": versions,
        "gpu": None,
    }
    if _uses_cuda(device):
        import torch

        major, minor = torch.cuda.get_device_capability(device)
        block["gpu"] = {"name": torch.cuda.get_device_name(device), "compute_capability": f"{major}.{minor}"}
    return block


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
