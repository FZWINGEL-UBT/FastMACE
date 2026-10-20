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
- ``capture_safe_mask_head``, ``first_derivative_only``, ``unfuse_full_addmm`` and
  ``single_graph_strain`` (1a-T08): the engine-level rewrites of ``fastmace.rewrites.engine``
  (MACE's readout mask, its force functions, Inductor's addmm rule with its cache key, and MACE
  PR 1498's one-graph strain). Each can be undone with ``restore``. Since 2-T08b no force field
  reads ``single_graph_strain``'s patch: each binds its strain path per model
  (``rewrites.engine.bind_strain_path``), and the patch remains for the reference calculators,
  which describe MACE's own calculator with or without it.

**Hash gates** (SPEC §6.3, plan task 2-T04). Every patch that replaces a function first checks
that the function it replaces is the one it was written for: the sha256 of its
``inspect.getsource`` must equal the pin in ``SOURCE_PINS`` (MACE 0.3.16 and torch 2.14.1, the
qualified versions). Otherwise nothing is replaced and ``UnqualifiedPatchError`` names the
source; the library never patches source it has not hashed. ``patch_problems(name)`` asks the
same question without patching.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import sys
from dataclasses import dataclass

from fastmace.errors import FastMACEError

# What is in force: name -> PatchRecord. The registry is the process-wide state this
# module exists to own; nothing else in fastmace keeps a copy.
_RECORDS: dict = {}
# MACE's own objects, before the first replacement: "module:attribute" -> object.
_ORIGINALS: dict = {}
# Objects that must stay alive while a patch is in force (the OEq runtime's preloaded libraries).
_KEEP_ALIVE: dict = {}
# Our replacements as installed: "module:attribute" -> object (for qualification, original_if_patched).
_INSTALLED: dict = {}


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
    _INSTALLED[f"{module.__name__}:{attribute}"] = value
    return f"{module.__name__}:{attribute}"


def original_if_patched(target, current):
    """MACE's preserved original for ``"module:attribute"`` when ``current`` is the replacement we installed.

    Qualification (SPEC §6.3) hashes this original instead of our replacement, so that a record
    stays qualified after its own patch is in place. Anything else, an unknown replacement
    included, returns None and is hashed as it is.
    """
    active = any(target in record.targets for record in _RECORDS.values())
    if active and target in _INSTALLED and current is _INSTALLED[target]:
        return _ORIGINALS[target]
    return None


# --------------------------------------------------------------------------- hash gates

# sha256 of inspect.getsource of every object a patch replaces, keyed "module:attribute" as
# replaced (MACE 0.3.16, torch 2.14.1). mace.modules.blocks imports mask_head from irreps_tools,
# mace.modules.models get_symmetric_displacement from utils, and mace.calculators.mace imports
# convert_e3nn_cueq.run as run_e3nn_to_cueq: the same source, so the same pin.
_MASK_HEAD = "bcb74869e70f9cd7ddfdfb7e133f4aacec058e8dd996b3c6eeb916e6e1b49e8e"
_STRAIN = "d75106300f967df0872e253c6d9c4607c0783a6dd8fe35045204e958a0b36437"
_PREPARE_GRAPH = "a369db1b449ebb4f4105cf7e42ad4cebea28950cd9d51a07f7b6cc8ec425c557"
_CUEQ_RUN = "5dde1426a0701a9bd9a4ced74fb810e3f82a849cff42b3778d96f8d7c4a4487c"
SOURCE_PINS = {
    "capture_safe_mask_head": {"mace.modules.blocks:mask_head": _MASK_HEAD},
    "first_derivative_only": {
        "mace.modules.utils:compute_forces": "b19d3e4acaf2c13eeff68562e22d6bb54cc03d73901336aba63e0b9a02c13565",
        "mace.modules.utils:compute_forces_virials":
            "8c1d2ab9d98ce99f3b8791c9c78f5f10d0d32ff5508ff48e0a641c174fed5ba0",
    },
    "single_graph_strain": {"mace.modules.utils:get_symmetric_displacement": _STRAIN,
                            "mace.modules.models:get_symmetric_displacement": _STRAIN,
                            # Not replaced: the per-model path reuses its code with other globals (2-T08b).
                            "mace.modules.utils:prepare_graph": _PREPARE_GRAPH},
    "cueq_conv_fusion": {"mace.cli.convert_e3nn_cueq:run": _CUEQ_RUN,
                         "mace.calculators.mace:run_e3nn_to_cueq": _CUEQ_RUN},
    # The addmm rule wraps torch's decision and reads its bias-like test: both are pinned.
    "unfuse_full_addmm": {
        "torch._inductor.fx_passes.post_grad:should_prefer_unfused_addmm":
            "5acf9ce3368b75519915c528ac2669fa6fac47741970b4324d8544c9236c10a1",
        "torch._inductor.fx_passes.post_grad:_is_bias_like_addmm_input":
            "a64f98f7d29f7398468b2bc2276dbff8ceacbe3ec120f07ee89179eff47ba321",
    },
}


class UnqualifiedPatchError(FastMACEError, RuntimeError):
    """A process-wide patch was refused: a source it replaces does not hash to its pin (SPEC §6.3).

    ``patch`` names the patch, ``problems`` lists each source and why.
    """

    default_next_step = ("Use the qualified MACE version (0.3.16); on another version the planner runs without "
                         "this patch (SPEC §18 decision 2).")

    def __init__(self, patch, problems):
        self.patch = patch
        self.problems = tuple(problems)
        super().__init__(f"process-wide patch {patch} refused: unqualified source: " + "; ".join(self.problems),
                         why="FastMACE never patches source it has not hashed (SPEC §6.3).")


def _present(target):
    """The object at ``"module:attribute"`` before any of our replacements (MACE's), or raises."""
    module_name, attribute = target.split(":", 1)
    module = importlib.import_module(module_name)
    key = f"{module.__name__}:{attribute}"
    if key in _ORIGINALS:
        return _ORIGINALS[key]
    return getattr(module, attribute)


def patch_problems(name):
    """Every source patch ``name`` would replace that does not hash to its pin (empty: qualified).

    A target module that does not exist is a problem; an attribute missing from a module that
    only re-imports it by name (``mace.modules.models``, ``mace.calculators.mace``) is not, since
    the patch skips it.
    """
    problems = []
    for target, expected in SOURCE_PINS[name].items():
        try:
            obj = _present(target)
            digest = hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
        except AttributeError as error:
            if target.split(":", 1)[0] in ("mace.modules.models", "mace.calculators.mace"):
                continue
            problems.append(f"{target}: {type(error).__name__}: {error}")
            continue
        except (ImportError, OSError, TypeError) as error:
            problems.append(f"{target}: {type(error).__name__}: {error}")
            continue
        if digest != expected:
            problems.append(f"{target}: source sha256 {digest[:12]} is not the qualified {expected[:12]}")
    return problems


def patch_applicable(name):
    """``(applicable, reason)``: whether patch ``name`` has anything to do on this torch and MACE.

    Only ``unfuse_full_addmm`` can be inapplicable: a torch without Inductor's bias-like addmm test
    (torch 2.13 and older) already keeps a full-matrix add unfused, so ``set_unfuse_full_addmm``
    changes nothing there. An inapplicable patch is not needed, which is not an unqualified source;
    its pins are not checked.
    """
    if name != "unfuse_full_addmm":
        return True, "applies"
    try:
        from torch._inductor.fx_passes import post_grad
    except ImportError as error:
        return False, f"not needed: Inductor is not importable ({error})"
    if not hasattr(post_grad, "_is_bias_like_addmm_input"):
        return False, ("not needed: this torch has no bias-like addmm test, so a full-matrix add already stays "
                       "unfused (torch 2.13 and older)")
    return True, "applies"


def _require_pinned(name):
    problems = patch_problems(name)
    if problems:
        raise UnqualifiedPatchError(name, problems)


def install_if_qualified(name):
    """Install engine-level patch ``name`` (``set_<name>(True)``) only if its sources hash; else log why.

    Returns the record in force, or None when the patch is not needed or not qualified. For the
    callers that can run with MACE's own function (SPEC §6.3: off with the reason logged); a
    caller that cannot (CUDA graph capture needs the capture-safe mask) calls the setter, which
    raises ``UnqualifiedPatchError``.
    """
    if name not in _RECORDS:
        if not patch_applicable(name)[0]:
            return None
        problems = patch_problems(name)
        if problems:
            import logging

            logging.getLogger("fastmace.rewrites").warning("%s not installed: unqualified source: %s", name,
                                                           "; ".join(problems))
            return None
    return globals()[f"set_{name}"](True)


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
    _require_pinned("cueq_conv_fusion")
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

@contextlib.contextmanager
def cueq_conv_fusion(enabled, device=None):
    """``set_cueq_conv_fusion(enabled)`` for the duration of the block, then the rule that was in force before.

    ``fastmace.model.loading.load`` converts each plain cuEq model inside this block, so its fusion
    setting acts on that conversion only and does not leak into later conversions in the process
    (a deep copy of a fused model loses MACE's convolution adapter). Yields the record in force
    inside the block (None for MACE's own rule).

    On a MACE whose converter does not hash (2-T04) nothing is patched when MACE's own rule gives
    the asked-for fusion on ``device`` (the block yields None); otherwise ``UnqualifiedPatchError``.
    """
    if enabled is not None and "cueq_conv_fusion" not in _RECORDS:
        problems = patch_problems("cueq_conv_fusion")
        if problems:
            if device is None or conv_fusion_for(enabled, device) != conv_fusion_for(None, device):
                raise UnqualifiedPatchError("cueq_conv_fusion", problems)
            yield None
            return
    before = _RECORDS.get("cueq_conv_fusion")
    record = set_cueq_conv_fusion(enabled)
    try:
        yield record
    finally:
        set_cueq_conv_fusion(None if before is None else before.value)


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


# ------------------------------------------------------------------------ engine-level rewrites (1a-T08)

# MACE 0.3.16 parameters of the functions the engine-level patches replace.
MASK_HEAD_PARAMETERS = ("x", "head", "num_heads")
COMPUTE_FORCES_PARAMETERS = ("energy", "positions", "training")
COMPUTE_FORCES_VIRIALS_PARAMETERS = ("energy", "positions", "displacement", "cell", "training", "compute_stress")
SYMMETRIC_DISPLACEMENT_PARAMETERS = ("positions", "unit_shifts", "cell", "edge_index", "num_graphs", "batch",
                                     "displacement")


def _check_parameters(function, expected, label):
    parameters = tuple(inspect.signature(function).parameters)
    if parameters != expected:
        raise RuntimeError(f"installed {label} takes {parameters}, not the MACE 0.3.16 parameters {expected}")


def _install(name, targets, detail):
    record = PatchRecord(name, True, tuple(targets), detail)
    _RECORDS[name] = record
    return record


def set_capture_safe_mask_head(enabled=True):
    """Put ``rewrites.engine.capture_safe_mask_head`` into MACE's readout blocks (True) or MACE's own back (False).

    MACE's mask_head builds the multi-head mask with indexed assignment, which CUDA graph capture
    rejects; mace.modules.blocks imports it by name. Returns the record in force (None after False).

    Applied again, it puts the function back if something replaced it since (the research engine
    installs its own copy before every capture). Compiled models guard on the function in
    ``blocks.mask_head``; with another one in place, the next eager call of a compiled model,
    such as the warm-up of a capture at a new edge size, would compile again (1a-T13b).
    """
    name = "capture_safe_mask_head"
    if not enabled:
        restore(name)
        return None
    from mace.modules import blocks

    from fastmace.rewrites import engine
    if name in _RECORDS:
        if blocks.mask_head is not engine.capture_safe_mask_head:
            _put(blocks, "mask_head", engine.capture_safe_mask_head)
        return _RECORDS[name]
    _require_pinned(name)
    _check_parameters(_original(blocks, "mask_head"), MASK_HEAD_PARAMETERS, "mace.modules.blocks.mask_head")
    return _install(name, [_put(blocks, "mask_head", engine.capture_safe_mask_head)],
                    "multi-head readout mask without indexed assignment (CUDA graph capture)")


def set_first_derivative_only(enabled=True):
    """MACE's compute_forces(_virials) without the second-derivative graph for training=True (True), or MACE's (False).

    Only calls with training=True change (compiled cuEquivariance models); training=False goes to
    MACE's own function. Returns the record in force (None after False).
    """
    name = "first_derivative_only"
    if not enabled:
        restore(name)
        return None
    if name in _RECORDS:
        return _RECORDS[name]
    from mace.modules import utils

    from fastmace.rewrites import engine
    _require_pinned(name)
    _check_parameters(_original(utils, "compute_forces"), COMPUTE_FORCES_PARAMETERS,
                      "mace.modules.utils.compute_forces")
    _check_parameters(_original(utils, "compute_forces_virials"), COMPUTE_FORCES_VIRIALS_PARAMETERS,
                      "mace.modules.utils.compute_forces_virials")
    targets = [_put(utils, "compute_forces", engine.first_derivative_forces),
               _put(utils, "compute_forces_virials", engine.first_derivative_forces_virials)]
    return _install(name, targets, "force and virial gradients with create_graph=False for training=True")


def set_unfuse_full_addmm(enabled=True):
    """Keep Inductor's ``mm + full-matrix add`` unfused on the GPU (True), or torch's own rule (False).

    Replaces torch._inductor.fx_passes.post_grad.should_prefer_unfused_addmm (looked up per match)
    and sets a no-op post_grad_custom_pre_pass whose uuid keys the compiler cache; both are undone
    together. On a torch without the newer bias-like check nothing is needed and nothing changes
    (returns None). Refuses when another custom pre pass is in use. Returns the record in force.
    """
    name = "unfuse_full_addmm"
    if not enabled:
        restore(name)
        return None
    if name in _RECORDS:
        return _RECORDS[name]
    try:
        from torch._inductor import config as inductor_config
        from torch._inductor.fx_passes import post_grad
    except ImportError:
        return None
    if not hasattr(post_grad, "_is_bias_like_addmm_input"):
        return None  # torch 2.5 behaviour already
    from fastmace.rewrites import engine
    _require_pinned(name)
    current = inductor_config.post_grad_custom_pre_pass
    if current is not None and not (hasattr(current, "uuid") and current.uuid() == engine.UNFUSED_ADDMM_UUID):
        raise RuntimeError("Inductor's post_grad_custom_pre_pass is already in use; cannot key the addmm choice")
    rule = engine.unfused_full_addmm_rule(_original(post_grad, "should_prefer_unfused_addmm"), post_grad)
    targets = [_put(post_grad, "should_prefer_unfused_addmm", rule)]
    if current is None:
        targets.append(_put(inductor_config, "post_grad_custom_pre_pass", engine.unfused_addmm_key()))
    return _install(name, targets, f"mm + full-matrix add unfused on the GPU; cache key {engine.UNFUSED_ADDMM_UUID}")


def set_single_graph_strain(enabled=True):
    """Put MACE PR 1498's one-graph get_symmetric_displacement into MACE (True), or MACE's own back (False).

    For the reference calculators only (``fastmace.reference.calculators``), which describe MACE's
    own calculator with or without this patch. No ``MACEForceField`` reads it: each binds its
    strain path per model (``fastmace.rewrites.engine.bind_strain_path``, 2-T08b).

    Replaces the function in mace.modules.utils, which prepare_graph calls (MACE, ScaleShiftMACE,
    PolarMACE), and in mace.modules.models, which imports it by name (EnergyDipolesMACE).
    torch.compile traces whichever function is in place, so call this before a model is
    compiled. Refuses a MACE whose function has another signature. Returns the record in force
    (None after False).
    """
    name = "single_graph_strain"
    if not enabled:
        restore(name)
        return None
    if name in _RECORDS:
        return _RECORDS[name]
    from mace.modules import models, utils

    from fastmace.rewrites import engine
    _require_pinned(name)
    _check_parameters(_original(utils, "get_symmetric_displacement"), SYMMETRIC_DISPLACEMENT_PARAMETERS,
                      "mace.modules.utils.get_symmetric_displacement")
    targets = [_put(module, "get_symmetric_displacement", engine.single_graph_symmetric_displacement)
               for module in (utils, models) if hasattr(module, "get_symmetric_displacement")]
    return _install(name, targets, "MACE PR 1498: one-graph strain without per-edge cell gathers")
