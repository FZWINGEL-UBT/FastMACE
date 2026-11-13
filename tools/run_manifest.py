"""Run manifests and the three configurations of the paper's experiment 1 (packed forward table consistency).

Configurations (OMAT-0 medium, float32), selected in a fresh process before anything is built:

  a   today's defaults: on the PACKED8 GPUs the spline forward reads the packed quadratic table, the w' reverse
      kernel the float32 cubic (values and slopes)
  b   a with ``spline_forward.PACKED8 = False``: forward and reverse read the same float32 cubic
  c   a with ``RewriteConfig(radial_tables="off")``: no radial tables, so no w' kernel and no spline; forces by
      autograd through the radial MLP (the control and the float32 noise floor)

``apply_config`` sets them; ``manifest`` writes what a record needs to be read later: the FastMACE revision, the
checkpoint SHA-256, the structure, the library versions, the resolved plan (rewrites, neighbour list, coordinate
path), every switch constant of the spline / w' kernels with its value on this device, and what each w' kernel
actually attached (spline grid, packed table). Callers add the timing schedule, start state and MD counters.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

CONFIGS = {
    "a": dict(constants={}, rewrites={}, text="today's defaults (PACKED8 forward, float32 cubic w')"),
    "b": dict(constants={"fastmace.kernels.oeq_streaming.spline_forward:PACKED8": False}, rewrites={},
              text="PACKED8 off: forward and w' read the same float32 cubic"),
    "c": dict(constants={}, rewrites={"radial_tables": "off"},
              text="radial tables off: no spline, no w' kernel (autograd through the radial MLP)"),
}

SWITCH_MODULES = ("fastmace.kernels.oeq_streaming.spline_forward", "fastmace.kernels.oeq_streaming.wprime_backward",
                  "fastmace.kernels.oeq_streaming.fused_forward", "fastmace.rewrites.radial",
                  "fastmace.rewrites.radial.spline", "fastmace.neighbors.skin")
PACKAGES = ("torch", "numpy", "ase", "mace-torch", "cuequivariance", "cuequivariance-torch",
            "cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13", "openequivariance", "triton")


def apply_config(name):
    """Set configuration ``name``'s module constants (before any build); returns its RewriteConfig overrides."""
    entry = CONFIGS[name]
    for target, value in entry["constants"].items():
        module, _, attribute = target.partition(":")
        mod = importlib.import_module(module)
        if not hasattr(mod, attribute):
            raise AttributeError(f"{target}: no such constant")
        setattr(mod, attribute, value)
    return dict(entry["rewrites"])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fastmace_revision():
    import subprocess

    import fastmace

    if os.environ.get("FASTMACE_SOURCE_COMMIT"):
        return os.environ["FASTMACE_SOURCE_COMMIT"]
    root = Path(fastmace.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = root / ".source_commit"
        return marker.read_text().strip() if marker.is_file() else None


def versions():
    from importlib import metadata

    out = dict(python=sys.version.split()[0], platform=platform.platform())
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def _plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return repr(value)[:300]


def switches(device):
    """{module: {constants: {NAME: value}, resolved: {fn: value on device}}} of the kernel switch modules."""
    out = {}
    for name in SWITCH_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError as error:
            out[name] = dict(error=str(error))
            continue
        constants = {k: _plain(v) for k, v in vars(mod).items()
                     if k.isupper() and not k.startswith("_") and isinstance(v, (bool, int, float, str, tuple,
                                                                                  type(None)))}
        resolved = {}
        for k, fn in vars(mod).items():
            if not (k.endswith("_default") and callable(fn) and getattr(fn, "__module__", None) == name):
                continue
            try:
                params = list(inspect.signature(fn).parameters)
                if params and params[0] == "device" and len(
                        [p for p in inspect.signature(fn).parameters.values() if p.default is p.empty]) == 1:
                    resolved[k] = _plain(fn(device))
            except Exception as error:  # noqa: BLE001  (a record, not a check)
                resolved[k] = f"error: {error}"[:200]
        out[name] = dict(constants=constants, resolved=resolved)
    return out


def attached():
    """What each live w' kernel (``WPrime``) attached: spline grid, packed table, staged edges, forward kernel."""
    try:
        from fastmace.kernels.oeq_streaming.wprime_backward import WPrime
    except ImportError:
        return []
    rows = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, WPrime):
                continue
        except ReferenceError:
            continue
        spline = getattr(obj, "spline", None)
        block = getattr(obj, "spline_block", None)
        forward = block[1] if isinstance(block, tuple) and len(block) > 1 else None
        rows.append(dict(
            spline=None if spline is None else dict(grid=_plain(spline.get("grid")), r_max=spline.get("r_max"),
                                                     r_lo=spline.get("r_lo"),
                                                     certificate=_plain((spline.get("certificate") or {}).get("worst"))),
            spline_packed=getattr(obj, "spline_packed", None) is not None,
            spline_block=block is not None,
            forward_packed8=None if forward is None else _plain(getattr(forward, "packed8", None)),
            forward_shared_locator=None if forward is None else _plain(getattr(forward, "shared_locator", None)),
            table_w=_plain(getattr(obj, "table_w", None)), horner=_plain(getattr(obj, "horner", None))))
    return rows


def spline_knots():
    """float64 numpy knots of the first attached spline (``rewrites.radial.spline.grid_knots``), or None."""
    from fastmace.rewrites.radial import spline as sp

    for row in attached():
        if row["spline"] is not None:
            return sp.grid_knots([tuple(g) for g in row["spline"]["grid"]]).numpy(), row["spline"]
    return None, None


def plan_record(plan):
    from fastmace.planner.fixed import auto_options_off, enabled_options

    record = dict(describe=plan.describe(full=True), enabled=list(enabled_options(plan)),
                  auto_options_off=auto_options_off(plan))
    for name in ("neighbours", "neighbors", "engine", "precision", "integrator"):
        part = getattr(plan, name, None)
        if part is not None:
            record[name] = _plain(part) if not hasattr(part, "__dataclass_fields__") else {
                f: _plain(getattr(part, f)) for f in part.__dataclass_fields__}
    return record


def manifest(*, config, checkpoint, structure, atoms, plan=None, extra=None):
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), host=platform.node(),
        argv=sys.argv, fastmace_revision=fastmace_revision(), config=config, config_text=CONFIGS[config]["text"],
        config_constants=CONFIGS[config]["constants"], config_rewrites=CONFIGS[config]["rewrites"],
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), structure=str(structure),
        natoms=len(atoms), formula=atoms.get_chemical_formula(), cell=atoms.cell.array.tolist(),
        versions=versions(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        cuda=torch.version.cuda, switches=switches(device), wprime_attached=attached(),
        environment={k: v for k, v in os.environ.items()
                     if k.startswith(("FASTMACE_", "TORCH", "PYTORCH", "CUDA_", "TRITON"))})
    if plan is not None:
        record["plan"] = plan_record(plan)
    record.update(extra or {})
    return record
