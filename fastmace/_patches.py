"""The registry of process-wide patches (SPEC §3.2 "Process-wide state", §13 provenance).

Some settings cannot live in a config object because they change something
the whole process shares: a function inside the installed MACE package, or the
environment that OpenEquivariance's extension build reads. Every such change
goes through this module, which

- applies it idempotently: asking again for the value in force changes nothing;
- keeps MACE's original object, so a patch can be undone (``restore``);
- records what it changed (``applied()``), for provenance and for the process
  constraints of SPEC §8.7, which refuse a second force field that needs a
  different value.

The installed package files are never modified. Entries so far:

- ``cueq_conv_fusion`` (plan task 1a-T05): MACE PR 1641's device rule for the
  fused cuEquivariance convolution in plain cuEq conversions.
- ``oeq_runtime`` (1a-T05): the in-process OpenEquivariance build environment
  (``fastmace.kernels._oeq_runtime.prepare_in_process``). It cannot be undone.
- ``polar_longrange`` (1a-T10, recorded here by 1a-T06): PolarMACE's class-level long-range
  wrappers, installed and recorded by ``fastmace.longrange.polar.install()``; this registry
  reports ``polar.installed()``. It is not undone from here.

Plan task 1a-T08 adds entries through the conductor.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass

# What is in force: name -> PatchRecord. The registry is the process-wide state this
# module exists to own; nothing else in fastmace keeps a copy.
_RECORDS: dict = {}
# MACE's own objects, before the first replacement: "module:attribute" -> object.
_ORIGINALS: dict = {}
# Objects that must stay alive while a patch is in force (the OEq runtime's preloaded libraries).
_KEEP_ALIVE: dict = {}


@dataclass(frozen=True)
class PatchRecord:
    """One process-wide patch in force."""

    name: str
    value: object  # the setting it implements (JSON-friendly)
    targets: tuple  # what it replaced or set: "module:attribute" or "env:VARIABLE"
    detail: str = ""


# Patches that their own module installs and records: (name, module whose installed() returns
# the PatchRecord fields or None). The module is not imported here; if it was never imported,
# its patch is not in force.
_RECORDED_ELSEWHERE = (("polar_longrange", "fastmace.longrange.polar"),)


def _recorded_elsewhere():
    records = []
    for name, module_name in _RECORDED_ELSEWHERE:
        module = sys.modules.get(module_name)
        fields = module.installed() if module is not None else None
        if fields:
            records.append(PatchRecord(fields["name"], fields["value"], tuple(fields["targets"]),
                                       fields.get("detail", "")))
    return records


def applied():
    """Every patch in force, in the order first applied (a tuple of PatchRecord); then those recorded elsewhere."""
    return tuple(_RECORDS.values()) + tuple(_recorded_elsewhere())


def record_of(name):
    """The PatchRecord of ``name``, or None."""
    if name in _RECORDS:
        return _RECORDS[name]
    return next((record for record in _recorded_elsewhere() if record.name == name), None)


def provenance():
    """``applied()`` as plain dictionaries, for metadata.json."""
    return [{"name": r.name, "value": r.value, "targets": list(r.targets), "detail": r.detail} for r in applied()]


def _original(module, attribute):
    key = f"{module.__name__}:{attribute}"
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(module, attribute)
    return _ORIGINALS[key]


def _put(module, attribute, value):
    _original(module, attribute)
    setattr(module, attribute, value)
    return f"{module.__name__}:{attribute}"


def restore(name):
    """Undo patch ``name`` (MACE's objects back in place) and drop its record; False if it was not applied.

    The OEq runtime changes the environment of a build that has happened; it cannot be undone.
    """
    if any(name == other for other, _ in _RECORDED_ELSEWHERE):
        if record_of(name) is None:
            return False
        raise RuntimeError(f"{name} is installed and kept by its own module; it is not undone from _patches")
    record = _RECORDS.get(name)
    if record is None:
        return False
    if name == "oeq_runtime":
        raise RuntimeError("the OEq runtime environment cannot be undone in a running process")
    for target in record.targets:
        module_name, attribute = target.split(":", 1)
        setattr(importlib.import_module(module_name), attribute, _ORIGINALS[target])
    del _RECORDS[name]
    return True


# ------------------------------------------------------------------------ cueq_conv_fusion

# mace.cli.convert_e3nn_cueq.run's MACE 0.3.16 parameters.
CUEQ_RUN_PARAMETERS = ("input_model", "output_model", "device", "return_model", "layout")


def conv_fusion_for(fused, device):
    """Whether a plain cuEquivariance conversion on ``device`` fuses the convolution.

    fused True: on any CUDA device (MACE PR 1641: torch.device(device).type);
    False: never; None: MACE 0.3.16's rule (only the device string "cuda").
    cuEq's fused kernel is CUDA-only.
    """
    import torch
    if fused is None:
        return device == "cuda"
    return bool(fused) and torch.device(device).type == "cuda"


def cueq_conv_fusion_run(fused, original):
    """MACE's convert_e3nn_cueq.run (``original``) with the convolution fused on CUDA (True) or never (False).

    MACE builds the model's CuEquivarianceConfig inside run; for the call the
    module's CuEquivarianceConfig is replaced by one that sets conv_fusion, and
    the converted model is checked: every interaction must carry MACE's
    conv_fusion marker exactly when fusion was asked for.
    """
    from mace.cli import convert_e3nn_cueq

    def run(input_model, output_model="_cueq.model", device="cpu", return_model=True, layout="ir_mul"):
        conv_fusion = conv_fusion_for(fused, device)
        config_class = convert_e3nn_cueq.CuEquivarianceConfig

        def config(**kwargs):
            kwargs["conv_fusion"] = conv_fusion
            return config_class(**kwargs)

        convert_e3nn_cueq.CuEquivarianceConfig = config
        try:
            model = original(input_model, output_model, device, return_model, layout)
        finally:
            convert_e3nn_cueq.CuEquivarianceConfig = config_class
        if model is not None:
            marked = [hasattr(interaction, "conv_fusion") for interaction in getattr(model, "interactions", ())]
            if any(flag != conv_fusion for flag in marked):
                raise RuntimeError(f"cuEquivariance conversion with conv_fusion={conv_fusion} gave interactions "
                                   f"with conv_fusion markers {marked}; installed MACE builds its config otherwise")
        return model

    run.conv_fusion = bool(fused)
    return run


def set_cueq_conv_fusion(enabled):
    """Make plain cuEquivariance conversions fuse the convolution (True), never fuse it (False) or follow MACE (None).

    None restores MACE 0.3.16's own rule (fused only for the device string
    "cuda"). Replaces ``run`` in mace.cli.convert_e3nn_cueq and
    ``run_e3nn_to_cueq`` in mace.calculators.mace (imported there by name), so
    call it before the calculator is built. The hybrid conversion does not use
    it. Refuses a MACE whose run has another signature. Returns the record in
    force (None after restoring MACE's rule).
    """
    if enabled is not None:
        enabled = bool(enabled)
    current = _RECORDS.get("cueq_conv_fusion")
    if enabled is None:
        restore("cueq_conv_fusion")
        return None
    if current is not None and current.value == enabled:
        return current
    from mace.cli import convert_e3nn_cueq
    calculator_module = importlib.import_module("mace.calculators.mace")
    original = _original(convert_e3nn_cueq, "run")
    parameters = tuple(inspect.signature(original).parameters)
    if parameters != CUEQ_RUN_PARAMETERS:
        raise RuntimeError(f"installed MACE's convert_e3nn_cueq.run takes {parameters}, "
                           f"not the 0.3.16 parameters {CUEQ_RUN_PARAMETERS}")
    replacement = cueq_conv_fusion_run(enabled, original)
    targets = [_put(convert_e3nn_cueq, "run", replacement)]
    if getattr(calculator_module, "run_e3nn_to_cueq", None) is not None:
        targets.append(_put(calculator_module, "run_e3nn_to_cueq", replacement))
    record = PatchRecord("cueq_conv_fusion", enabled, tuple(targets),
                         "MACE PR 1641: fused cuEq convolution on any CUDA device" if enabled
                         else "cuEq convolution never fused")
    _RECORDS.pop("cueq_conv_fusion", None)
    _RECORDS["cueq_conv_fusion"] = record
    return record


# ------------------------------------------------------------------------ oeq_runtime

def prepare_oeq_runtime(cache):
    """Prepare the OEq build environment in this process once (``_oeq_runtime.prepare_in_process``).

    Idempotent: a second call returns the first record, whatever ``cache`` is.
    Raises ``OEqRuntimeError`` when it is too late (OpenEquivariance or
    ``torch.utils.cpp_extension`` already imported) or the runtime is missing.
    """
    current = _RECORDS.get("oeq_runtime")
    if current is not None:
        return current
    from fastmace.kernels import _oeq_runtime
    runtime_plan, handles = _oeq_runtime.prepare_in_process(cache)
    _KEEP_ALIVE["oeq_runtime"] = handles
    record = PatchRecord("oeq_runtime", str(runtime_plan.root),
                         tuple(f"env:{name}" for name in sorted(runtime_plan.environment)),
                         "OEq host-extension build environment; preloaded "
                         + ", ".join(p.name for p in runtime_plan.preload))
    _RECORDS["oeq_runtime"] = record
    return record
